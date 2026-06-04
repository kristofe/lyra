"""FastAPI service for Lyra-2 custom-trajectory video generation.

The model is loaded ONCE at startup (FastAPI lifespan) and kept warm, so each
request only does per-image work (depth, embedding, AR sampling) — no weight
reloading. Endpoints: POST /generate, POST /sequence/generate, plus download/info
routes; see each handler and api/README.md for the request fields.

Per-request fields (resolution, fps, num_frames, prompt, seed, guidance) only
affect each data batch; model identity (experiment, checkpoint, DMD, MoGe) is
fixed at startup via the LYRA_* env vars below. Launch with api/run_server_cu12.sh
so the right CUDA env is set before torch imports.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent

# Each /generate call persists a self-contained run folder here:
#   outputs/api_runs/<YYYYmmdd-HHMMSS>_<id>/{input.*, trajectory.npz, output.mp4, metadata.json}
RUNS_DIR = BASE_DIR / "outputs" / "api_runs"

# Each sequential session persists here:
#   outputs/api_sessions/<session_id>/{session.json, seed_input.*, gen_000/, ..., _state/}
SESSIONS_DIR = BASE_DIR / "outputs" / "api_sessions"

DEFAULT_PROMPT = "A high-quality, photorealistic scene with smooth, stable camera motion."

# Fixed-at-startup model identity (override via env before launching).
ENGINE_CONFIG = dict(
    experiment=os.environ.get("LYRA_EXPERIMENT", "lyra2"),
    checkpoint_dir=os.environ.get("LYRA_CHECKPOINT_DIR", "checkpoints/model"),
    use_dmd=os.environ.get("LYRA_USE_DMD", "0") == "1",
    use_moge_scale=os.environ.get("LYRA_USE_MOGE", "1") == "1",
    buffer_selection=os.environ.get("LYRA_BUFFER_SELECTION", "last_history"),
    # Per-block torch.compile of the DiT is ON by default (~1.2x on ar_inference).
    # The compiled kernels are warmed at startup (see lifespan) so the first real
    # request is already fast. Set LYRA_COMPILE_DIT=0 to disable.
    compile_dit=os.environ.get("LYRA_COMPILE_DIT", "1") == "1",
)

# Startup warmup: trigger DiT compilation once at boot so no client request pays
# the (one-time, per input-shape) compile cost. The warmup SYNTHESIZES its own
# throwaway image + trajectory (see _make_warmup_inputs) — it needs no on-disk
# assets, and the content is meaningless: it exists only to push one generation
# through the model at the target resolution. torch.compile(dynamic=False)
# specializes on input shape, which depends on RESOLUTION (not num_frames —
# FramePack generates fixed-size chunks), so the warmup covers all num_frames at
# this resolution; requests at a different resolution recompile once.
WARMUP_ON_STARTUP = os.environ.get("LYRA_WARMUP_ON_STARTUP", "1") == "1"
WARMUP_RESOLUTION = os.environ.get("LYRA_WARMUP_RESOLUTION", "240,416")  # 240p default
WARMUP_NUM_FRAMES = int(os.environ.get("LYRA_WARMUP_NUM_FRAMES", "81"))  # 1 chunk = cheapest

# Populated in the lifespan handler.
state: dict = {"engine": None}


# --- Authentication -------------------------------------------------------
# This service runs on a Studio that also holds team GitHub/S3 credentials, and
# it is reachable unauthenticated over the internal network (a direct hit to the
# Studio IP bypasses any Lightning proxy token). So auth is enforced HERE, in the
# app, on every path: a request must carry `Authorization: Bearer <LYRA_API_TOKEN>`.
#
# Fail closed: if LYRA_API_TOKEN is unset/empty the server refuses to start,
# rather than silently running open. /health is intentionally left unauthenticated
# so liveness probes work without sharing the token.
API_TOKEN = os.environ.get("LYRA_API_TOKEN", "")


def require_token(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency: reject any request without a valid bearer token."""
    scheme, _, token = (authorization or "").partition(" ")
    # Constant-time compare to avoid leaking the token via timing.
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, API_TOKEN):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _make_warmup_inputs(dir_path: str, resolution: tuple[int, int], num_frames: int):
    """Write a throwaway image + trajectory .npz for the startup warmup.

    The output is intentionally generic — a smooth RGB gradient and a gentle
    forward dolly — and only has to be *valid* enough to run one generation so the
    DiT compiles. No bundled sample files are involved. Returns ``(image_path,
    trajectory_path)``.
    """
    import cv2  # local import: only needed at warmup, and engine already pulls cv2 in

    h, w = int(resolution[0]), int(resolution[1])

    # Smooth gradient frame (avoids the degenerate depth a flat image could give).
    xx = np.broadcast_to(np.linspace(0, 255, w, dtype=np.float32)[None, :], (h, w))
    yy = np.broadcast_to(np.linspace(0, 255, h, dtype=np.float32)[:, None], (h, w))
    rgb = np.stack([xx, yy, np.full((h, w), 128.0, np.float32)], axis=-1).astype(np.uint8)
    image_path = os.path.join(dir_path, "warmup.png")
    cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Trajectory: identity rotation + small incremental forward translation.
    n = int(num_frames)
    w2c = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    w2c[:, 2, 3] = np.arange(n, dtype=np.float32) * 0.02
    f = float(max(h, w))
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    intrinsics = np.tile(K, (n, 1, 1))
    trajectory_path = os.path.join(dir_path, "warmup_traj.npz")
    np.savez(trajectory_path, w2c=w2c, intrinsics=intrinsics, image_height=h, image_width=w)

    return image_path, trajectory_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail closed: refuse to start (and don't waste minutes loading the model) if no
    # token is configured, so the API is never accidentally served unauthenticated.
    if not API_TOKEN:
        raise RuntimeError(
            "LYRA_API_TOKEN is not set. This service is reachable unauthenticated over "
            "the internal network, so a token is required. Set LYRA_API_TOKEN before launch "
            "(see api/run_server_cu12.sh)."
        )

    # Import here so torch/lyra only load under the proper CUDA env (run_server_cu13.sh).
    from api.engine import LyraEngine

    print(f"[startup] Loading Lyra-2 engine: {ENGINE_CONFIG}", flush=True)
    t0 = time.perf_counter()
    engine = LyraEngine(base_dir=str(BASE_DIR), **ENGINE_CONFIG)
    engine.load()
    print(f"[startup] Engine ready in {time.perf_counter() - t0:.1f}s "
          f"(timings: {engine.load_timings})", flush=True)

    # Warm up the compiled DiT so the first client request is already fast.
    if ENGINE_CONFIG.get("compile_dit") and WARMUP_ON_STARTUP:
        try:
            wt = time.perf_counter()
            res_hw = _parse_resolution(WARMUP_RESOLUTION)
            with tempfile.TemporaryDirectory(prefix="lyra_warmup_") as wd:
                img_path, traj_path = _make_warmup_inputs(wd, res_hw, WARMUP_NUM_FRAMES)
                r = engine.generate(
                    input_image_path=img_path,
                    trajectory_path=traj_path,
                    output_path=wd,
                    prompt=DEFAULT_PROMPT,
                    num_frames=WARMUP_NUM_FRAMES,
                    fps=30,
                    resolution=res_hw,
                    seed=1,
                )
            print(f"[startup] DiT warmup done in {time.perf_counter() - wt:.1f}s "
                  f"(synthetic inputs, resolution {res_hw}, "
                  f"ar_inference={r['timings'].get('ar_inference')}s) "
                  f"— compiled kernels cached for this resolution.", flush=True)
        except Exception as e:
            print(f"[startup] WARNING: DiT warmup failed ({type(e).__name__}: {e}); "
                  f"first request will pay the compile cost.", flush=True)

    state["engine"] = engine
    yield
    state["engine"] = None


app = FastAPI(title="Lyra-2 Inference API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health():
    engine = state["engine"]
    return {
        "status": "ready" if engine is not None else "loading",
        "config": ENGINE_CONFIG,
        "load_timings": engine.load_timings if engine is not None else None,
    }


def _validate_num_frames(n: int) -> None:
    if n < 1 or (n - 1) % 80 != 0:
        raise HTTPException(
            status_code=400,
            detail=f"num_frames must satisfy (N-1) % 80 == 0 (e.g. 81, 161, 241, 321). Got {n}.",
        )


# session_id is used to build filesystem paths (os.path.join in the engine), so it
# must be a single safe path component. A bare "/"/"\\" filter is NOT enough — it lets
# ".." through, enabling one-level path traversal. Require a strict allowlist AND that
# it survives Path(...).name unchanged (rejects ".", "..", and any separator).
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id) or Path(session_id).name != session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id.")


def _parse_resolution(resolution: str) -> tuple[int, int]:
    parts = resolution.split(",")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise HTTPException(status_code=400, detail=f"resolution must be 'H,W' (e.g. '480,832'). Got {resolution!r}.")
    return int(parts[0]), int(parts[1])


def _validate_trajectory(npz_path: Path, num_frames: int) -> None:
    try:
        data = np.load(npz_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read trajectory .npz: {e}")
    missing = {"w2c", "intrinsics"} - set(data.files)
    if missing:
        raise HTTPException(status_code=400, detail=f"Trajectory .npz missing keys: {sorted(missing)}. Found: {data.files}")
    if data["w2c"].shape[0] < num_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Trajectory has {data['w2c'].shape[0]} poses but num_frames={num_frames} requested.",
        )


@app.post("/generate", dependencies=[Depends(require_token)])
def generate(
    image: UploadFile = File(..., description="Single input image"),
    trajectory: UploadFile = File(..., description="Camera trajectory .npz (w2c, intrinsics, image_height, image_width)"),
    resolution: str = Form("480,832"),
    fps: int = Form(30),
    num_frames: int = Form(241),
    prompt: str = Form(DEFAULT_PROMPT),
    seed: int | None = Form(None),
    guidance: float | None = Form(None),
):
    engine = state["engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine still loading; retry shortly.")

    _validate_num_frames(num_frames)
    res_hw = _parse_resolution(resolution)
    if fps < 1:
        raise HTTPException(status_code=400, detail="fps must be >= 1.")

    # Persistent, self-contained run folder: outputs/api_runs/<timestamp>_<id>/
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{os.urandom(3).hex()}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        img_suffix = Path(image.filename or "input.png").suffix or ".png"
        img_path = run_dir / f"input{img_suffix}"
        with img_path.open("wb") as f:
            shutil.copyfileobj(image.file, f)

        traj_path = run_dir / "trajectory.npz"
        with traj_path.open("wb") as f:
            shutil.copyfileobj(trajectory.file, f)
        _validate_trajectory(traj_path, num_frames)

        # engine.generate writes <output_path>/output.mp4
        try:
            result = engine.generate(
                input_image_path=str(img_path),
                trajectory_path=str(traj_path),
                output_path=str(run_dir),
                prompt=prompt,
                num_frames=num_frames,
                fps=fps,
                resolution=res_hw,
                seed=seed,
                guidance=guidance,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation error: {e}")

        mp4 = Path(result["video_path"])
        if not mp4.exists():
            raise HTTPException(status_code=500, detail="Inference produced no video.")

        # Write a sidecar metadata.json so each run folder is self-describing.
        metadata = {
            "run_id": run_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "request": {
                "image_filename": image.filename,
                "trajectory_filename": trajectory.filename,
                "prompt": prompt,
                "num_frames": num_frames,
                "fps": fps,
                "resolution": list(res_hw),
                "seed": seed,
                "guidance": guidance,
            },
            "engine_config": ENGINE_CONFIG,
            "result": {
                "video": mp4.name,
                "input": img_path.name,
                "trajectory": traj_path.name,
                "num_frames": result.get("num_frames"),
                "fps": result.get("fps"),
                "resolution": result.get("resolution"),
                "timings": result["timings"],
            },
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Expose per-step timings + the saved run location as response headers.
        headers = {f"X-Timing-{k}": str(v) for k, v in result["timings"].items()}
        headers["X-Run-Id"] = run_id
        headers["X-Run-Dir"] = str(run_dir)
        return FileResponse(
            path=str(mp4),
            media_type="video/mp4",
            filename=f"{run_id}.mp4",
            headers=headers,
        )
    except BaseException:
        # Only successful, complete runs persist; clean up partial/failed ones.
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


@app.post("/sequence/generate", dependencies=[Depends(require_token)])
def sequence_generate(
    trajectory: UploadFile = File(..., description="Camera trajectory .npz for this generation"),
    session_id: str | None = Form(None, description="Omit to start a new session; pass to continue one"),
    image: UploadFile | None = File(None, description="Seed image (required only for a new session)"),
    resolution: str = Form("480,832"),
    fps: int = Form(30),
    num_frames: int = Form(81),
    prompt: str = Form(DEFAULT_PROMPT),
    seed: int | None = Form(None),
    guidance: float | None = Form(None),
    export_pointcloud: bool = Form(False),
):
    """Generate one video chunk within a session, accumulating persistent spatial memory.

    The first call (no ``session_id``) starts a session from ``image`` ("point A").
    Each later call passes the returned ``session_id`` (and a new trajectory); the
    spatial memory is loaded and accumulated while the temporal memory is reset, so
    every generation branches from point A. See api/sequential.py for the design.
    """
    engine = state["engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine still loading; retry shortly.")

    _validate_num_frames(num_frames)
    res_hw = _parse_resolution(resolution)
    if fps < 1:
        raise HTTPException(status_code=400, detail="fps must be >= 1.")

    if session_id is not None:
        _validate_session_id(session_id)
        if not (SESSIONS_DIR / session_id / "session.json").is_file():
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    elif image is None:
        raise HTTPException(status_code=400, detail="A new session requires an 'image'.")

    stage = Path(tempfile.mkdtemp(prefix="lyra_seq_"))
    try:
        traj_path = stage / "trajectory.npz"
        with traj_path.open("wb") as f:
            shutil.copyfileobj(trajectory.file, f)
        _validate_trajectory(traj_path, num_frames)

        img_path = None
        img_suffix = ".png"
        if image is not None:
            img_suffix = Path(image.filename or "input.png").suffix or ".png"
            img_path = stage / f"input{img_suffix}"
            with img_path.open("wb") as f:
                shutil.copyfileobj(image.file, f)

        try:
            result = engine.generate_sequential(
                sessions_root=str(SESSIONS_DIR),
                session_id=session_id,
                trajectory_path=str(traj_path),
                input_image_path=str(img_path) if img_path is not None else None,
                prompt=prompt,
                num_frames=num_frames,
                fps=fps,
                resolution=res_hw,
                seed=seed,
                guidance=guidance,
                export_pointcloud=export_pointcloud,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Sequential generation error: {e}")

        sid = result["session_id"]
        gen_idx = result["generation_index"]
        mp4 = Path(result["video_path"])
        if not mp4.exists():
            raise HTTPException(status_code=500, detail="Inference produced no video.")
        gen_dir = mp4.parent

        # Persist the request inputs alongside the generation for reproducibility.
        shutil.copyfile(traj_path, gen_dir / "trajectory.npz")
        if image is not None and session_id is None:
            # Save the seed image once, at session creation.
            shutil.copyfile(img_path, SESSIONS_DIR / sid / f"seed_input{img_suffix}")

        pts = Path(result["pointcloud_path"]) if result.get("pointcloud_path") else None
        gen_name = gen_dir.name

        metadata = {
            "session_id": sid,
            "generation_index": gen_idx,
            "created": datetime.now().isoformat(timespec="seconds"),
            "request": {
                "continued_session": session_id is not None,
                "trajectory_filename": trajectory.filename,
                "image_filename": image.filename if image is not None else None,
                "prompt": prompt,
                "num_frames": num_frames,
                "fps": fps,
                "resolution": list(res_hw),
                "seed": seed,
                "guidance": guidance,
            },
            "engine_config": ENGINE_CONFIG,
            "result": {
                "video": mp4.name,
                "pointcloud": pts.name if pts is not None else None,
                "num_cache_entries": result.get("num_cache_entries"),
                "num_frames": result.get("num_frames"),
                "fps": result.get("fps"),
                "resolution": result.get("resolution"),
                "timings": result["timings"],
            },
        }
        (gen_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        headers = {f"X-Timing-{k}": str(v) for k, v in result["timings"].items()}
        headers["X-Session-Id"] = sid
        headers["X-Generation-Index"] = str(gen_idx)
        headers["X-Run-Dir"] = str(gen_dir)
        headers["X-Cache-Entries"] = str(result.get("num_cache_entries"))
        if pts is not None:
            headers["X-Pointcloud-Url"] = f"/sessions/{sid}/{gen_name}/{pts.name}"
        return FileResponse(
            path=str(mp4),
            media_type="video/mp4",
            filename=f"{sid}_{gen_name}.mp4",
            headers=headers,
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)


@app.get("/sequence/{session_id}", dependencies=[Depends(require_token)])
def session_info(session_id: str):
    """Summarize a sequential session: metadata + the per-generation artifacts."""
    _validate_session_id(session_id)
    sdir = (SESSIONS_DIR / session_id).resolve()
    if sdir.parent != SESSIONS_DIR.resolve() or not (sdir / "session.json").is_file():
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    meta = json.loads((sdir / "session.json").read_text())
    generations = []
    for gen_dir in sorted(sdir.glob("gen_*")):
        if gen_dir.is_dir():
            generations.append({
                "name": gen_dir.name,
                "artifacts": sorted(p.name for p in gen_dir.iterdir() if p.is_file()),
            })
    return {"session": meta, "generations": generations}


@app.delete("/sequence/{session_id}", dependencies=[Depends(require_token)])
def delete_session(session_id: str):
    """Reset/erase a session: delete its persisted spatial memory + all generations.

    To start fresh instead of erasing, simply omit ``session_id`` on the next
    ``POST /sequence/generate`` (a new session begins with empty spatial memory).
    """
    _validate_session_id(session_id)
    sdir = (SESSIONS_DIR / session_id).resolve()
    if sdir.parent != SESSIONS_DIR.resolve() or not sdir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    shutil.rmtree(sdir, ignore_errors=True)
    return {"deleted": session_id}


@app.get("/sessions/{session_id}/{gen}/{artifact}", dependencies=[Depends(require_token)])
def download_session_artifact(session_id: str, gen: str, artifact: str):
    """Download a per-generation artifact (video, .pts, metadata.json, ...)."""
    _validate_session_id(session_id)
    # gen/artifact must each be a single safe path component (rejects "", ".", "..", separators).
    for p in (gen, artifact):
        if not p or Path(p).name != p:
            raise HTTPException(status_code=400, detail="Invalid path component.")
    gen_dir = (SESSIONS_DIR / session_id / gen).resolve()
    file_path = (gen_dir / artifact).resolve()
    sessions_root = SESSIONS_DIR.resolve()
    if (sessions_root not in gen_dir.parents) or (gen_dir not in file_path.parents):
        raise HTTPException(status_code=400, detail="Invalid path.")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Not found: {session_id}/{gen}/{artifact}")
    media_type = _ARTIFACT_MEDIA_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(path=str(file_path), media_type=media_type, filename=artifact)


# Pick a sensible content-type for the small set of artifacts a run produces.
_ARTIFACT_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".npz": "application/octet-stream",
    ".json": "application/json",
    ".png": "image/png",
    ".pts": "text/plain",
    ".pt": "application/octet-stream",
}


@app.get("/runs/{run_id}/{artifact}", dependencies=[Depends(require_token)])
def download_artifact(run_id: str, artifact: str):
    """Download a file from a persisted run folder (e.g. output.mp4, metadata.json)."""
    # Guard against path traversal: names only, must resolve inside the run dir.
    if "/" in run_id or "/" in artifact or "\\" in run_id or "\\" in artifact:
        raise HTTPException(status_code=400, detail="Invalid run_id or artifact name.")
    run_dir = (RUNS_DIR / run_id).resolve()
    file_path = (run_dir / artifact).resolve()
    if run_dir.parent != RUNS_DIR.resolve() or run_dir not in file_path.parents:
        raise HTTPException(status_code=400, detail="Invalid path.")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Not found: {run_id}/{artifact}")
    media_type = _ARTIFACT_MEDIA_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(path=str(file_path), media_type=media_type, filename=artifact)
