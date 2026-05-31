# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Next-best-view trajectory + anchor builder for gap-filling an existing splat scene.

Workflow this supports (no lyra core changes):
  1. A scene is already reconstructed (splats + a set of cameras).
  2. The user picks a START camera (an existing scene view) and a TARGET pose (e.g.
     the live viewer camera looking at an under-covered region).
  3. We interpolate N camera poses (lerp translation + slerp rotation) from start to
     target — all in the SPLAT WORLD frame — and write them as a `trajectory.npz`
     that ``lyra2_custom_traj_inference.load_trajectory`` understands.
  4. We render a few existing scene cameras (RGB + metric depth + poses) as
     ``multiview.npz`` anchors so the generation is grounded in the splat world scale.
  5. The start view's rendered RGB is the generation seed image.

The generated video is then registered back via the GUI's existing ``append_video``
(DA3 re-poses + aligns), so the trajectory only STEERS where new views appear.

Pose convention: world-to-camera (w2c) 4x4, OpenCV (look=+Z, up=-Y, right=+X) — the
same convention the viewer's ``viser_camera_to_opencv_viewmat`` and lyra's
``camera_utils.look_at_matrix`` use, so poses pass straight through.

Reuses ``camera_utils.{slerp, matrix_to_quaternion, quaternion_to_matrix}``. NOTE:
``camera_utils.interpolate`` is intentionally NOT used — it never returns a value.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

import gsplat

from lyra_2._src.inference.camera_utils import (
    slerp,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


def _interp_pose_w2c(w2c_a: torch.Tensor, w2c_b: torch.Tensor, t: float,
                     device) -> torch.Tensor:
    """Interpolate between two w2c 4x4 poses at fraction t in [0,1].

    Interpolation is done in CAMERA->WORLD space (c2w): lerp the camera CENTER and
    slerp the orientation, then invert back to w2c. Interpolating w2c translation
    directly would lerp ``-R@C`` which bends the path; interpolating centers is the
    geometrically correct camera move.
    """
    c2w_a = torch.linalg.inv(w2c_a)
    c2w_b = torch.linalg.inv(w2c_b)

    center_a = c2w_a[:3, 3]
    center_b = c2w_b[:3, 3]
    interp_center = (1.0 - t) * center_a + t * center_b

    q_a = matrix_to_quaternion(c2w_a, device)
    q_b = matrix_to_quaternion(c2w_b, device)
    q = slerp(q_a, q_b, torch.as_tensor(float(t), device=device))

    c2w = torch.eye(4, device=device, dtype=torch.float32)
    c2w[:3, :3] = quaternion_to_matrix(q, device)[:3, :3]
    c2w[:3, 3] = interp_center
    return torch.linalg.inv(c2w)


def build_lerp_trajectory(
    start_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    K: torch.Tensor,
    n_frames: int,
    H: int,
    W: int,
) -> dict:
    """Build a lerp+slerp w2c trajectory from start to target (inclusive endpoints).

    Args:
        start_w2c, target_w2c: (4,4) world-to-camera, splat world frame.
        K: (3,3) intrinsics (shared across frames).
        n_frames: number of poses; must be 1 + 80k (81, 161, 241, ...).
        H, W: image size the intrinsics refer to.

    Returns an npz-ready payload dict:
        w2c (N,4,4) float32, intrinsics (N,3,3) float32, image_height, image_width.
    Frame 0 == start_w2c, frame N-1 == target_w2c.
    """
    if (n_frames - 1) % 80 != 0 or n_frames < 81:
        raise ValueError(
            f"n_frames={n_frames} invalid: must be 1 + 80k (81, 161, 241, ...).")
    device = start_w2c.device
    start_w2c = start_w2c.to(torch.float32)
    target_w2c = target_w2c.to(torch.float32)

    poses = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        poses.append(_interp_pose_w2c(start_w2c, target_w2c, t, device))
    w2c = torch.stack(poses, dim=0)  # (N,4,4)
    Ks = K.to(torch.float32).unsqueeze(0).repeat(n_frames, 1, 1)  # (N,3,3)

    return {
        "w2c": w2c.detach().cpu().numpy().astype(np.float32),
        "intrinsics": Ks.detach().cpu().numpy().astype(np.float32),
        "image_height": int(H),
        "image_width": int(W),
    }


def _render_rgbd(scene, w2c: torch.Tensor, K: torch.Tensor, H: int, W: int,
                 device) -> tuple[torch.Tensor, torch.Tensor]:
    """Render one view from the splats. Returns (rgb HWC in [0,1], depth HW metric).

    Uses gsplat RGB+ED — the same call the viewer uses — so depth is in the splats'
    world units (the same frame as the poses).
    """
    viewmats = w2c.to(torch.float32, copy=False).reshape(1, 4, 4).to(device)
    Ks = K.to(torch.float32, copy=False).reshape(1, 3, 3).to(device)
    sh = scene.sh
    sh_degree_eff = None if sh is None else int(round(sh.shape[1] ** 0.5)) - 1
    out, _alpha, _info = gsplat.rasterization(
        means=scene.means,
        quats=scene.quats,
        scales=scene.scales,
        opacities=scene.opacities,
        colors=sh,
        viewmats=viewmats,
        Ks=Ks,
        width=int(W),
        height=int(H),
        sh_degree=sh_degree_eff,
        render_mode="RGB+ED",
    )
    rgb = out[0, :, :, :3].clamp(0.0, 1.0)        # (H,W,3)
    depth = out[0, :, :, 3]                         # (H,W) metric
    return rgb, depth


def render_seed_image(scene, start_w2c: torch.Tensor, K: torch.Tensor,
                      H: int, W: int) -> np.ndarray:
    """Render the START view as a uint8 RGB image (H,W,3) to use as the gen seed."""
    device = scene.means.device
    rgb, _ = _render_rgbd(scene, start_w2c, K, H, W, device)
    return (rgb.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)


def render_anchor_views(
    scene,
    anchor_w2c: torch.Tensor,
    anchor_K: torch.Tensor,
    H: int,
    W: int,
) -> dict:
    """Render K existing-camera views as multiview anchors for the Sparse3DCache.

    Args:
        anchor_w2c: (K,4,4) world-to-camera, splat world frame.
        anchor_K:   (K,3,3) intrinsics.

    Returns an npz-ready payload matching lyra's --multiview_path schema:
        video [1,3,K,H,W] in [-1,1], depth [1,K,H,W] (metric, splat units),
        camera_w2c [1,K,4,4], intrinsics [1,K,3,3], image_height, image_width.
    """
    device = scene.means.device
    Kn = int(anchor_w2c.shape[0])
    rgbs, depths = [], []
    for i in range(Kn):
        rgb, depth = _render_rgbd(scene, anchor_w2c[i], anchor_K[i], H, W, device)
        rgbs.append(rgb.permute(2, 0, 1))   # (3,H,W)
        depths.append(depth)                # (H,W)
    video = torch.stack(rgbs, dim=1).unsqueeze(0)        # (1,3,K,H,W) in [0,1]
    video = (video * 2.0 - 1.0).clamp(-1.0, 1.0)          # → [-1,1]
    depth = torch.stack(depths, dim=0).unsqueeze(0)       # (1,K,H,W)

    return {
        "video": video.detach().cpu().numpy().astype(np.float32),
        "depth": depth.detach().cpu().numpy().astype(np.float32),
        "camera_w2c": anchor_w2c.detach().cpu().numpy().reshape(1, Kn, 4, 4).astype(np.float32),
        "intrinsics": anchor_K.detach().cpu().numpy().reshape(1, Kn, 3, 3).astype(np.float32),
        "image_height": int(H),
        "image_width": int(W),
    }


def nearest_cameras(
    data,
    target_w2c: torch.Tensor,
    count: int,
    include: Sequence[int] = (),
) -> list[int]:
    """Indices of the `count` existing cameras whose centers are closest to the
    target camera center, always including any `include` indices first.

    `data` is a VideoData (has `.c2w (N,4,4)`). Used to pick anchor views near the
    region being filled.
    """
    c2w = data.c2w.to(torch.float32)
    centers = c2w[:, :3, 3]                                  # (N,3)
    target_center = torch.linalg.inv(target_w2c.to(torch.float32))[:3, 3]
    d = torch.linalg.norm(centers - target_center.to(centers), dim=1)  # (N,)
    order = torch.argsort(d).tolist()
    picked = list(dict.fromkeys(list(include) + order))     # de-dup, keep order
    return picked[:max(count, len(include))]


def save_trajectory_npz(payload: dict, path: str) -> str:
    np.savez(
        path,
        w2c=payload["w2c"],
        intrinsics=payload["intrinsics"],
        image_height=payload["image_height"],
        image_width=payload["image_width"],
    )
    return path


def save_multiview_npz(payload: dict, path: str) -> str:
    np.savez(
        path,
        video=payload["video"],
        depth=payload["depth"],
        camera_w2c=payload["camera_w2c"],
        intrinsics=payload["intrinsics"],
        image_height=payload["image_height"],
        image_width=payload["image_width"],
    )
    return path
