# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
REST API server exposing Lyra2 single-image -> zoom video generation.

This wraps the same pipeline as ``lyra2_zoomgs_inference.py`` (zoom-in + zoom-out
exploration video) behind a small HTTP API so a GUI / CLI can POST an image and get
an mp4 back. The heavy model is loaded ONCE at startup and reused for every request;
GPU work is serialized with a lock (single GPU).

Run (MUST go through the lyra2 conda env so its activation hooks set the
LD_PRELOAD libcudart.so.13 shim from INSTALL_BLACKWELL.md — launching the env's
python binary directly skips them and DA3 aborts with "Multiple libcudart
libraries found"):
    PYTHONPATH=. conda run --no-capture-output -n lyra2 \\
        python -m lyra_2._src.inference.demo_server \\
        --checkpoint_dir checkpoints/model --experiment lyra2 --use_dmd --port 8080

Endpoints:
    GET  /health        -> {"status": "ok", "model_loaded": true}
    GET  /resolutions   -> {"default": "480p", "presets": [{label,height,width}, ...]}
    POST /generate      -> streams an mp4 (multipart form field `image`, plus optional
                           `prompt`, `resolution`, `trajectory`, frame/strength fields)

The original CLI (``lyra2_zoomgs_inference.py``) is left completely untouched; this
module imports its leaf helpers so no inference logic is duplicated.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lyra_2._ext.imaginaire.utils import log, misc
from lyra_2._ext.imaginaire.visualize.video import save_img_or_video
from lyra_2._src.utils.model_loader import load_model_from_checkpoint

# Leaf helpers reused verbatim from the zoom CLI (no duplication of inference logic).
from lyra_2._src.inference.lyra2_zoomgs_inference import (
    _apply_dmd_defaults,
    _da3_infer_depth_intrinsics_single,
    _fit_ground_normal_from_depth,
    _generate_one_direction,
)
from lyra_2._src.inference.camera_traj_utils import CAMERA_TRAJECTORY_CHOICES

# Camera-move directions the trajectory builder accepts.
DIRECTION_CHOICES = ("left", "right", "up", "down")
DEFAULT_TRAJECTORY = "horizontal_zoom"
DEFAULT_DIRECTION = "right"

torch.enable_grad(False)
torch.backends.cudnn.enabled = False


# ---------------------------------------------------------------------------
# Resolution presets (dropdown). H and W must be multiples of 16 (video VAE
# 8x spatial compression x patch 2). Smaller = faster + less VRAM.
# ---------------------------------------------------------------------------
RESOLUTION_PRESETS = {
    "480p": (480, 832),
    "360p": (368, 640),
    "320p": (320, 576),
    "240p": (256, 448),
}
DEFAULT_RESOLUTION = "240p"


def resolve_resolution(value: Optional[str]) -> str:
    """Accept a preset label ('480p') or a raw 'H,W' string; return validated 'H,W'.

    Both dimensions must be positive multiples of 16. Raises ValueError otherwise.
    """
    if not value:
        value = DEFAULT_RESOLUTION
    value = str(value).strip()
    if value in RESOLUTION_PRESETS:
        h, w = RESOLUTION_PRESETS[value]
    else:
        try:
            parts = [int(x) for x in value.replace("x", ",").split(",")]
            if len(parts) != 2:
                raise ValueError
            h, w = parts
        except ValueError:
            raise ValueError(
                f"Invalid resolution '{value}'. Use a preset "
                f"({', '.join(RESOLUTION_PRESETS)}) or 'H,W' (e.g. '480,832')."
            )
    if h <= 0 or w <= 0 or h % 16 != 0 or w % 16 != 0:
        raise ValueError(
            f"Resolution {h},{w} invalid: height and width must be positive "
            f"multiples of 16."
        )
    return f"{h},{w}"


def validate_num_frames(n: int, name: str) -> int:
    """Frame counts must be of the form 1 + 80k (81, 161, 241, ...)."""
    n = int(n)
    if n < 81 or (n - 1) % 80 != 0:
        raise ValueError(
            f"{name}={n} invalid: must be 1 + 80k (e.g. 81, 161, 241, 321)."
        )
    return n


def validate_trajectory(name: str) -> str:
    if name not in CAMERA_TRAJECTORY_CHOICES:
        raise ValueError(
            f"Invalid trajectory '{name}'. Choices: "
            f"{', '.join(CAMERA_TRAJECTORY_CHOICES)}."
        )
    return name


def validate_direction(d: str) -> str:
    if d not in DIRECTION_CHOICES:
        raise ValueError(
            f"Invalid direction '{d}'. Use one of {', '.join(DIRECTION_CHOICES)}."
        )
    return d


# ---------------------------------------------------------------------------
# Default args namespace mirroring the zoomgs CLI defaults. The server overrides
# a few fields per request (resolution, prompt, frame counts, strengths).
# ---------------------------------------------------------------------------
def build_default_args(server_args: argparse.Namespace) -> argparse.Namespace:
    a = argparse.Namespace(
        input_image_path="",
        num_samples=1,
        sample_start_idx=0,
        sample_id=None,
        prompt="",
        prompt_dir=None,
        prompt_suffix="",
        experiment=server_args.experiment,
        checkpoint_dir=server_args.checkpoint_dir,
        output_path=server_args.output_dir,
        guidance=5.0,
        shift=5.0,
        num_sampling_step=50,
        seed=1,
        fps=16,
        num_frames=81,
        # Single-trajectory controls (one camera move per request).
        trajectory=server_args.default_trajectory,
        direction=DEFAULT_DIRECTION,
        strength=0.5,
        resolution=server_args.default_resolution,
        context_parallel_size=1,
        lora_paths=None,
        # NOTE: None (not [0.4, 0.4]) so that --use_dmd yields a single matching
        # (path, weight) pair via _apply_dmd_defaults instead of a length mismatch.
        lora_weights=None,
        offload=False,
        offload_when_prompt=False,
        use_moge_scale=True,
        ground_plane_align=False,
        ground_plane_bottom_frac=0.4,
        depth_backend="da3",
        da3_model_name="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        da3_model_path_custom="checkpoints/recon/model.pt",
        da3_frame_interval=8,
        da3_max_history_frames=10,
        da3_include_ar_chunk_last_frames=False,
        da3_use_predicted_pose=False,
        da3_predicted_pose_continuation=False,
        use_dmd=server_args.use_dmd,
        ablate_same_t5=False,
        use_dmd_scheduler=False,
        warp_chunk_size=None,
        num_retrieval_views=1,
        disable_cache_update=False,
        multiview_ids=None,
        offload_da3_diffusion=False,
    )
    _apply_dmd_defaults(a)
    return a


# ---------------------------------------------------------------------------
# Pipeline load (once) — mirrors lyra2_zoomgs_inference.__main__ setup.
# ---------------------------------------------------------------------------
def load_pipeline(args: argparse.Namespace) -> dict:
    misc.set_random_seed(seed=args.seed, by_rank=True)

    negative_prompt_data = torch.load(
        "checkpoints/text_encoder/negative_prompt.pt", map_location="cpu", weights_only=False
    )

    experiment_opts = [
        "model.config.use_mp_policy_fsdp=False",
        "model.config.keep_original_net_dtype=False",
    ]
    if args.lora_paths:
        experiment_opts += ["model.config.net.postpone_checkpoint=True"]

    model, config = load_model_from_checkpoint(
        config_file="lyra_2/_src/configs/config.py",
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint_dir,
        enable_fsdp=False,
        instantiate_ema=False,
        load_ema_to_reg=False,
        experiment_opts=experiment_opts,
    )
    if args.lora_paths:
        lora_names = []
        for lora_path in args.lora_paths:
            lora_name = model.load_lora_weights(lora_path)
            lora_names.append(lora_name)
        model.set_weights_and_activate_adapters(lora_names, args.lora_weights)
        if hasattr(model, "net") and hasattr(model.net, "enable_selective_checkpoint"):
            model.net.enable_selective_checkpoint(model.net.sac_config, model.net.blocks)

    desired_dtype = model.tensor_kwargs.get("dtype", None)
    desired_device = model.tensor_kwargs.get("device", None)
    if desired_dtype is not None:
        model.net = model.net.to(device=desired_device, dtype=desired_dtype)
        log.info(f"Casted model.net to dtype={desired_dtype}", rank0_only=True)

    assert getattr(model.config, "important_start", True) is True
    assert getattr(model.config, "encode_video_from_start", True) is True
    assert not getattr(model.config, "use_hd_map_cond", False)

    model.eval()
    if args.warp_chunk_size is not None:
        model.config.warp_chunk_size = args.warp_chunk_size
        model.warp_chunk_size = args.warp_chunk_size

    from lyra_2._src.inference.depth_utils import load_da3_model
    da3_device = model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    da3_model = load_da3_model(
        da3_model_name=args.da3_model_name,
        da3_model_path_custom=args.da3_model_path_custom,
        device=da3_device,
    )
    da3_model.eval()

    moge_model = None
    if args.use_moge_scale:
        from lyra_2._src.inference.depth_utils import load_moge_model
        moge_device = model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        moge_model = load_moge_model(moge_device)
        moge_model.eval()
        log.info("MoGe model loaded for depth scale alignment.", rank0_only=True)

    log.info("Pipeline loaded and ready.", rank0_only=True)
    return {
        "model": model,
        "config": config,
        "da3_model": da3_model,
        "moge_model": moge_model,
        "negative_prompt_data": negative_prompt_data,
        "desired_device": desired_device,
        "desired_dtype": desired_dtype,
    }


# ---------------------------------------------------------------------------
# Single generation — ONE camera trajectory per call (no zoom-in/zoom-out pair).
# Returns (mp4_path, last_frame_png_path); the last frame is the end of the
# camera move, handy for chaining a follow-up call that continues the scene.
# ---------------------------------------------------------------------------
def run_single(
    ctx: dict,
    args: argparse.Namespace,
    image_path: str,
    prompt: str,
    output_path: str,
) -> tuple[str, str]:
    model = ctx["model"]
    da3_model = ctx["da3_model"]
    moge_model = ctx["moge_model"]
    negative_prompt_data = ctx["negative_prompt_data"]
    desired_device = ctx["desired_device"]
    desired_dtype = ctx["desired_dtype"]
    process_group = None

    target_h, target_w = [int(x) for x in args.resolution.split(",")]

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    videos_dir = os.path.join(output_path, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"{base_name}.mp4")
    last_frame_path = os.path.join(videos_dir, f"{base_name}_last.png")

    misc.set_random_seed(seed=args.seed, by_rank=True)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_t = torch.from_numpy(rgb)  # H,W,3 uint8

    # Step 1: Depth & intrinsics
    log.info("Running DA3 single-image depth...", rank0_only=True)
    image_chw01, depth_hw, K_33, mask_hw = _da3_infer_depth_intrinsics_single(
        da3_model=da3_model,
        img_rgb_uint8=rgb_t,
        target_hw=(target_h, target_w),
    )

    # Step 1b: Optionally align DA3 depth to MoGe scale
    if args.use_moge_scale and moge_model is not None:
        log.info("Aligning DA3 depth to MoGe scale...", rank0_only=True)
        from lyra_2._src.inference.depth_utils import moge_infer_depth_intrinsics

        moge_model.to(desired_device)
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            _, moge_depth_hw, _, moge_mask_hw = moge_infer_depth_intrinsics(
                moge_model,
                rgb_t,
                depth_pred_hw=(target_h, target_w),
                target_hw=(target_h, target_w),
            )

        da3_d = depth_hw.to(moge_depth_hw.device)
        da3_m = mask_hw.to(moge_mask_hw.device)
        valid_mask = (da3_m > 0.5) & (moge_mask_hw > 0.5)
        if valid_mask.sum() > 10:
            inv_da3 = 1.0 / (da3_d[valid_mask] + 1e-6)
            inv_moge = 1.0 / (moge_depth_hw[valid_mask] + 1e-6)
            numerator = (inv_da3 * inv_moge).sum()
            denominator = (inv_da3 * inv_da3).sum()
            if denominator > 1e-8:
                scale = numerator / denominator
                log.info(f"Global inverse-depth scale factor: {scale.item()}", rank0_only=True)
                if scale > 1e-6:
                    depth_hw = depth_hw / scale.to(depth_hw.device)
                else:
                    log.warning(f"Scale too small ({scale.item()}), skipping alignment.", rank0_only=True)
            else:
                log.warning("Denominator too small for LS scale alignment.", rank0_only=True)
        else:
            log.warning("Not enough overlapping valid pixels for scale alignment.", rank0_only=True)

        moge_model.cpu()
        del moge_depth_hw, moge_mask_hw, da3_d, da3_m
        torch.cuda.empty_cache()
        gc.collect()

    img_bchw = image_chw01.to(device=desired_device) * 2.0 - 1.0  # [-1,1]

    # Step 2: Caption
    caption = prompt.strip() if prompt and prompt.strip() else "a high quality scenic photo"
    log.info(f"Using prompt: {caption}", rank0_only=True)
    if args.prompt_suffix:
        caption = caption.rstrip() + " " + args.prompt_suffix

    # Step 2b: T5 embeddings
    from lyra_2._src.inference.get_t5_emb import get_umt5_embedding, get_umt5_embedding_offloaded
    if args.offload_when_prompt:
        t5 = get_umt5_embedding_offloaded(caption, device=desired_device).to(dtype=desired_dtype)
    else:
        t5 = get_umt5_embedding(caption, device=desired_device).to(dtype=desired_dtype)
    if t5.dim() == 2:
        t5 = t5.unsqueeze(0)
    elif t5.dim() == 3 and t5.shape[0] != 1:
        t5 = t5[:1]
    neg_t5 = misc.to(negative_prompt_data["t5_text_embeddings"], **model.tensor_kwargs)

    N = int(args.num_frames)

    # Step 2c: Optionally fit ground plane
    ground_normal = None
    if args.ground_plane_align:
        ground_normal = _fit_ground_normal_from_depth(
            depth_hw, K_33, mask_hw, bottom_frac=args.ground_plane_bottom_frac,
        )
        if ground_normal is None:
            log.warning("Ground plane fitting failed, using original trajectory.", rank0_only=True)

    # Per-trajectory strength normalization so `strength` feels consistent across
    # presets. The rotate-in-place presets feed `rotation_angle = 1.0 * strength`
    # in DEGREES (camera_traj_utils), whereas orbits feed `angle = strength` in
    # RADIANS. Left as-is, strength=0.5 → a 0.5° pan (invisible). Scale by
    # 180/pi (~57°/unit) so strength=0.5 ≈ 29° like the orbits.
    import math as _math
    effective_strength = args.strength
    if args.trajectory in ("rotate_spot", "rotate_spot_noise"):
        effective_strength = args.strength * (180.0 / _math.pi)

    # Step 3: Generate the single camera trajectory.
    log.info(
        f"=== GENERATE ({args.trajectory} {args.direction} "
        f"str={args.strength}→{effective_strength:.2f}, N={N}) ===", rank0_only=True)
    result = _generate_one_direction(
        model=model, args=args, img_bchw=img_bchw, depth_hw=depth_hw, mask_hw=mask_hw,
        K_33=K_33, t5_embeddings=t5, neg_t5_embeddings=neg_t5,
        trajectory=args.trajectory, direction=args.direction,
        strength=effective_strength, N=N, da3_model=da3_model,
        process_group=process_group, log_prefix=base_name,
        ground_normal_cam=ground_normal,
    )
    if result is None:
        raise RuntimeError(f"Generation failed for {image_path}")

    # Save the clip: result["video"] is [B, C, T, H, W] in [-1, 1].
    video01 = (result["video"][0].clamp(-1, 1) * 0.5 + 0.5).float().cpu()  # [C,T,H,W] in [0,1]
    save_img_or_video(video01, video_path.replace(".mp4", ""), fps=args.fps)
    log.info(f"Saved video ({video01.shape[1]} frames): {video_path}", rank0_only=True)

    # Extract the LAST frame (end of the camera move) as a png so a follow-up
    # call can continue the scene from where this clip left off.
    last_rgb = (video01[:, -1].permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    cv2.imwrite(last_frame_path, cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR))
    log.info(f"Saved last frame: {last_frame_path}", rank0_only=True)

    del video01
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if not os.path.exists(video_path):
        raise RuntimeError(f"Expected output not found: {video_path}")
    return video_path, last_frame_path


# ===========================================================================
# HTTP server (FastAPI)
# ===========================================================================
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# Module-level state populated at startup.
STATE: dict = {"ctx": None, "args": None, "server_args": None}
GPU_LOCK = threading.Lock()
# job_id -> {"video": <mp4 path>, "last_frame": <png path>} for follow-up fetches.
JOBS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    server_args = STATE["server_args"]
    base_args = build_default_args(server_args)
    log.info("Loading Lyra2 pipeline (this can take a while)...", rank0_only=True)
    STATE["ctx"] = load_pipeline(base_args)
    STATE["args"] = base_args
    log.info("Server ready.", rank0_only=True)
    yield
    STATE["ctx"] = None


app = FastAPI(title="Lyra2 demo server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["ctx"] is not None}


@app.get("/resolutions")
def resolutions():
    return {
        "default": DEFAULT_RESOLUTION,
        "presets": [
            {"label": k, "height": v[0], "width": v[1]}
            for k, v in RESOLUTION_PRESETS.items()
        ],
    }


@app.get("/trajectories")
def trajectories():
    """Camera-motion options for the GUI dropdown (one trajectory per call)."""
    return {
        "default": DEFAULT_TRAJECTORY,
        "default_direction": DEFAULT_DIRECTION,
        "directions": list(DIRECTION_CHOICES),
        "trajectories": list(CAMERA_TRAJECTORY_CHOICES),
    }


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    resolution: str = Form(DEFAULT_RESOLUTION),
    trajectory: str = Form(DEFAULT_TRAJECTORY),
    direction: str = Form(DEFAULT_DIRECTION),
    num_frames: int = Form(81),
    strength: float = Form(0.5),
    fps: int = Form(16),
    seed: int = Form(1),
):
    """Generate ONE camera trajectory from the uploaded image and stream the mp4.

    The response carries an ``X-Job-Id`` header; ``GET /last_frame/{job_id}``
    then returns the final frame of the clip, which you can POST back as the
    image for a follow-up call to continue the scene.
    """
    if STATE["ctx"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate request params before doing any heavy work.
    try:
        res = resolve_resolution(resolution)
        n = validate_num_frames(num_frames, "num_frames")
        traj = validate_trajectory(trajectory)
        dirn = validate_direction(direction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    job_id = uuid.uuid4().hex[:12]
    server_args = STATE["server_args"]
    job_dir = os.path.join(server_args.output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    # Validate the upload is a decodable image, but persist the ORIGINAL bytes so
    # run_single's cv2.imread reads exactly what we just verified.
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
    suffix = os.path.splitext(image.filename or "")[1].lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        suffix = ".png"
    image_path = os.path.join(job_dir, f"input{suffix}")
    with open(image_path, "wb") as f:
        f.write(data)

    # Per-request args (copy of the loaded defaults with overrides).
    req_args = argparse.Namespace(**vars(STATE["args"]))
    req_args.resolution = res
    req_args.num_frames = n
    req_args.trajectory = traj
    req_args.direction = dirn
    req_args.strength = float(strength)
    req_args.fps = int(fps)
    req_args.seed = int(seed)

    # Serialize GPU work (single GPU); run the blocking job in a worker thread.
    import anyio

    def _job():
        with GPU_LOCK:
            return run_single(STATE["ctx"], req_args, image_path, prompt, job_dir)

    try:
        mp4_path, last_frame_path = await anyio.to_thread.run_sync(_job)
    except ValueError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 - surface a clean error, keep server up
        log.error(f"Generation failed for job {job_id}: {e}")
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    JOBS[job_id] = {"video": mp4_path, "last_frame": last_frame_path}
    return FileResponse(
        mp4_path, media_type="video/mp4", filename=f"{job_id}.mp4",
        headers={"X-Job-Id": job_id},
    )


@app.get("/last_frame/{job_id}")
def last_frame(job_id: str):
    """Return the final frame of a previous /generate job as a PNG.

    POST this back as the `image` of a new /generate call to continue the scene.
    Note: each call re-grounds (fresh depth + identity pose), so continuity is
    visual, not a single consistent 3D frame — drift accumulates when chaining.
    """
    job = JOBS.get(job_id)
    if not job or not os.path.exists(job.get("last_frame", "")):
        raise HTTPException(status_code=404, detail=f"No last frame for job_id {job_id!r}.")
    return FileResponse(
        job["last_frame"], media_type="image/png", filename=f"{job_id}_last.png"
    )


def parse_server_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lyra2 zoom video REST server")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/model")
    p.add_argument("--experiment", type=str, default="lyra2")
    p.add_argument("--use_dmd", action=argparse.BooleanOptionalAction, default=True,
                   help="Use the 4-step DMD distillation LoRA (fast). --no-use_dmd for full quality.")
    p.add_argument("--default-resolution", dest="default_resolution", type=str,
                   default=RESOLUTION_PRESETS[DEFAULT_RESOLUTION][0].__str__() + "," +
                           RESOLUTION_PRESETS[DEFAULT_RESOLUTION][1].__str__())
    p.add_argument("--default-trajectory", dest="default_trajectory", type=str,
                   default="horizontal_zoom")
    p.add_argument("--output-dir", dest="output_dir", type=str, default="outputs/demo_server")
    return p.parse_args()


if __name__ == "__main__":
    import uvicorn

    server_args = parse_server_args()
    server_args.default_resolution = resolve_resolution(server_args.default_resolution)
    os.makedirs(server_args.output_dir, exist_ok=True)
    STATE["server_args"] = server_args
    uvicorn.run(app, host=server_args.host, port=server_args.port)
