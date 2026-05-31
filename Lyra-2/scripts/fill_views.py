# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Headless gap-fill: synthesize views between two camera poses in a splat scene.

Given an existing reconstruction (Inria 3DGS .ply) and two camera poses (a START
pose, e.g. an existing scene camera, and a TARGET pose at an under-covered region),
this:
  1. builds an N-frame lerp+slerp trajectory from start → target (splat world frame),
  2. renders the START view as the generation seed image,
  3. renders a few anchor views (RGB + metric depth + poses) to ground the scale,
  4. POSTs everything to the demo_server's /generate_custom endpoint,
  5. saves the returned mp4.

The mp4 can then be appended back into the scene (GUI "append video", or
lyra2 reconstruction) — the trajectory only steers WHERE new views are created.

Run the server first (see DEMO_README.md), then e.g.:

    PYTHONPATH=. conda run --no-capture-output -n lyra2 python scripts/fill_views.py \\
        --ply outputs/scene/splats.ply \\
        --start-w2c start.npy --target-w2c target.npy \\
        --fov 60 --height 256 --width 448 \\
        --num-frames 81 --server-url http://localhost:8000/generate \\
        --out /tmp/fill.mp4

Poses are world-to-camera 4x4 (OpenCV convention), as .npy or .json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


def _ensure_cuda_include_path() -> None:
    """Put the conda env's CUDA dev headers on CPATH before gsplat is imported.

    gsplat JIT-compiles its CUDA kernels on first use; in a cold standalone
    process nvcc can't find ``cuda_runtime_api.h`` unless the include dirs are on
    the path (the running GUI avoids this because it already has the kernels
    compiled/loaded). This only mutates THIS process's env — it does not touch the
    conda environment on disk. No-op if the dirs aren't found.
    """
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return
    candidates = [
        os.path.join(prefix, "lib", "python3.10", "site-packages",
                     "nvidia", "cuda_runtime", "include"),
        os.path.join(prefix, "targets", "x86_64-linux", "include"),
        os.path.join(prefix, "lib", "python3.10", "site-packages",
                     "nvidia", "cudnn", "include"),
        os.path.join(prefix, "include"),
    ]
    dirs = [d for d in candidates if os.path.isdir(d)]
    if not dirs:
        return
    existing = os.environ.get("CPATH", "")
    parts = dirs + ([existing] if existing else [])
    os.environ["CPATH"] = ":".join(parts)


_ensure_cuda_include_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Make the visergui modules importable (nbv_trajectory, splat loader).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "visergui"))

import nbv_trajectory as nbv          # noqa: E402
import video_api                      # noqa: E402
from splat_trainer import load_inria_ply  # noqa: E402


class _SceneShim:
    """Minimal stand-in for viewer.SceneState that nbv_trajectory can render.

    Holds ACTIVATED gsplat inputs (scales = exp(log_s), opacities = sigmoid(logit_o)).
    """

    def __init__(self, means, quats, scales, opacities, sh):
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.sh = sh


def _load_mat(path: str, shape) -> torch.Tensor:
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        with open(path) as f:
            arr = np.array(json.load(f), dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr)


def _K_from_fov(fov_deg: float, H: int, W: int) -> torch.Tensor:
    fov = math.radians(fov_deg)
    fy = 0.5 * H / math.tan(0.5 * fov)
    fx = fy
    cx, cy = 0.5 * W, 0.5 * H
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                        dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Gap-fill views between two splat-world camera poses.")
    p.add_argument("--ply", required=True, help="Inria 3DGS .ply of the existing scene.")
    p.add_argument("--start-w2c", required=True, help="START pose, 4x4 w2c (.npy/.json).")
    p.add_argument("--target-w2c", required=True, help="TARGET pose, 4x4 w2c (.npy/.json).")
    p.add_argument("--K", default=None, help="3x3 intrinsics (.npy/.json). Or use --fov.")
    p.add_argument("--fov", type=float, default=60.0, help="Vertical FOV degrees if --K omitted.")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--num-frames", type=int, default=81, help="1 + 80k (81, 161, …).")
    p.add_argument("--anchor-w2c", default=None,
                   help="Optional (K,4,4) w2c anchors (.npy). Default: just the start pose.")
    p.add_argument("--no-anchors", action="store_true",
                   help="Skip multiview anchors (scale grounding). Faster, less consistent.")
    p.add_argument("--server-url", default="http://localhost:8000/generate")
    p.add_argument("--prompt", default="")
    p.add_argument("--resolution", default=None, help="Preset label or 'H,W' (default: server default).")
    p.add_argument("--pose-scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="/tmp/fill.mp4")
    args = p.parse_args()

    device = torch.device(args.device)
    H, W = int(args.height), int(args.width)

    loaded = load_inria_ply(args.ply, device=str(device))
    # load_inria_ply returns a dict of RAW params: means (M,3), sh0 (M,1,3),
    # shN (M,R,3), log_s (M,3) [log-scale], logit_o (M,) [logit-opacity], quats (M,4).
    # gsplat.rasterization wants ACTIVATED scales/opacities and full SH coeffs.
    sh = torch.cat([loaded["sh0"], loaded["shN"]], dim=1) \
        if loaded["shN"].shape[1] > 0 else loaded["sh0"]
    scene = _SceneShim(
        means=loaded["means"],
        quats=loaded["quats"],
        scales=torch.exp(loaded["log_s"]),
        opacities=torch.sigmoid(loaded["logit_o"]).reshape(-1),
        sh=sh,
    )

    start_w2c = _load_mat(args.start_w2c, (4, 4)).to(device)
    target_w2c = _load_mat(args.target_w2c, (4, 4)).to(device)
    K = (_load_mat(args.K, (3, 3)) if args.K else _K_from_fov(args.fov, H, W)).to(device)

    # 1. trajectory
    traj_payload = nbv.build_lerp_trajectory(start_w2c, target_w2c, K, args.num_frames, H, W)
    traj_path = os.path.splitext(args.out)[0] + "_trajectory.npz"
    nbv.save_trajectory_npz(traj_payload, traj_path)
    print(f"[fill] trajectory: {args.num_frames} poses → {traj_path}")

    # 2. seed image (start view)
    seed_rgb = nbv.render_seed_image(scene, start_w2c, K, H, W)  # (H,W,3) uint8
    import cv2
    seed_path = os.path.splitext(args.out)[0] + "_seed.png"
    cv2.imwrite(seed_path, cv2.cvtColor(seed_rgb, cv2.COLOR_RGB2BGR))
    with open(seed_path, "rb") as f:
        seed_bytes = f.read()
    print(f"[fill] seed image → {seed_path}")

    # 3. anchors (optional)
    mv_bytes = None
    if not args.no_anchors:
        if args.anchor_w2c:
            anchor_w2c = _load_mat(args.anchor_w2c, (-1, 4, 4)).to(device)
        else:
            anchor_w2c = start_w2c.reshape(1, 4, 4)
        anchor_K = K.reshape(1, 3, 3).repeat(anchor_w2c.shape[0], 1, 1)
        mv_payload = nbv.render_anchor_views(scene, anchor_w2c, anchor_K, H, W)
        mv_path = os.path.splitext(args.out)[0] + "_multiview.npz"
        nbv.save_multiview_npz(mv_payload, mv_path)
        with open(mv_path, "rb") as f:
            mv_bytes = f.read()
        print(f"[fill] {anchor_w2c.shape[0]} anchor view(s) → {mv_path}")

    # 4. POST to /generate_custom
    print(f"[fill] requesting generation from {args.server_url} …")
    video_bytes = video_api.request_custom_video(
        args.server_url, seed_bytes, "seed.png",
        trajectory_npz_bytes=open(traj_path, "rb").read(),
        multiview_npz_bytes=mv_bytes,
        prompt=args.prompt, resolution=args.resolution, pose_scale=args.pose_scale,
    )

    # 5. save
    with open(args.out, "wb") as f:
        f.write(video_bytes)
    print(f"[fill] saved {len(video_bytes):,} bytes → {args.out}")


if __name__ == "__main__":
    main()
