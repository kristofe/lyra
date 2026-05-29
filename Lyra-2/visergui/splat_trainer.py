"""
Barebones splat trainer scaffold. Phase 1: video -> VIPE cameras.

This script does only the preprocessing step today: read an mp4, run VIPE
on it, save the per-frame poses + intrinsics + depth to
`outputs/<name>/vipe_predictions.npz`. No training loop yet.

The class shape (`SplatTrainer.step()` placeholder) is forward-compatible
with the Phase 5 `Trainer` seam in `visergui/training.py` so this file
can grow into a real trainer without restructuring.

Run:
    python visergui/splat_trainer.py path/to/video.mp4 \
        [--name X] [--out-dir outputs] [--fps F] [--no-fast]
    python visergui/splat_trainer.py  outputs/zoomgs/videos/14.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

import torch.nn.functional as F
import matplotlib.pyplot as plt
import lpips
from tqdm.auto import tqdm
from plyfile import PlyData, PlyElement
from gsplat import rasterization, rasterization_2dgs, DefaultStrategy
from gsplat.utils import depth_to_normal

if TYPE_CHECKING:
    from viewer import SceneState


# --------------------------------------------------------------------------- #
# Data bundles
# --------------------------------------------------------------------------- #


@dataclass
class VideoData:
    """Output of `_preprocess_video`: everything derived from the input mp4."""
    rgb:   torch.Tensor              # (N,H,W,3) float in [0,1]
    depth: torch.Tensor              # (N,H,W) f32
    K:     torch.Tensor              # (N,3,3)
    w2c:   torch.Tensor              # (N,4,4)
    c2w:   torch.Tensor              # (N,4,4)
    conf:  torch.Tensor | None       # (N,H,W) f32, or None if DA3 didn't return it
    sky:   torch.Tensor | None       # (N,H,W) bool, or None
    N: int
    H: int
    W: int
    # Per-frame epoch tag. Frames from the original `_preprocess_video` are
    # epoch 0; frames added later via `append_frame` get a higher epoch so
    # Phase 3 sampling + Phase 5 frustum coloring can tell them apart. Use a
    # plain Python list (length == N) — these are tiny per-frame integers,
    # accessed by viewer code and rarely by step().
    frame_epoch: list[int] | None = None

    def __post_init__(self) -> None:
        if self.frame_epoch is None:
            self.frame_epoch = [0] * int(self.N)

    def append_frame(self, rgb: torch.Tensor, depth: torch.Tensor,
                     K: torch.Tensor, c2w: torch.Tensor,
                     conf: torch.Tensor | None = None,
                     sky: torch.Tensor | None = None,
                     epoch: int | None = None) -> int:
        """Append one frame (H, W, 3) RGB / (H, W) depth / (3, 3) K / (4, 4) c2w.

        Casts to the existing tensors' device + dtype, recomputes w2c, and
        bumps N. Returns the new frame index.

        Resolution must match (self.H, self.W). Caller can pass conf/sky;
        if either field already has rows, the matching argument is required
        (we fill with ones / zeros if None to keep the row count aligned).

        `epoch` tags the new frame's session (Phase 3 sampling / Phase 5
        visual debug). Defaults to the current max epoch (so back-to-back
        appends collect into one session); callers starting a new session
        (e.g. `append_video`, `append_supplied_frames`) pass an explicit
        `max(frame_epoch) + 1`.
        """
        if rgb.shape[:2] != (self.H, self.W) or depth.shape != (self.H, self.W):
            raise ValueError(f"frame shape must be ({self.H}, {self.W}); got rgb={tuple(rgb.shape)}, depth={tuple(depth.shape)}")
        rgb_new   = rgb.to(self.rgb).reshape(1, self.H, self.W, 3)
        depth_new = depth.to(self.depth).reshape(1, self.H, self.W)
        K_new     = K.to(self.K).reshape(1, 3, 3)
        c2w_new   = c2w.to(self.c2w).reshape(1, 4, 4)
        w2c_new   = torch.linalg.inv(c2w_new)

        self.rgb   = torch.cat([self.rgb,   rgb_new],   dim=0)
        self.depth = torch.cat([self.depth, depth_new], dim=0)
        self.K     = torch.cat([self.K,     K_new],     dim=0)
        self.c2w   = torch.cat([self.c2w,   c2w_new],   dim=0)
        self.w2c   = torch.cat([self.w2c,   w2c_new],   dim=0)
        if self.conf is not None:
            conf_row = conf.to(self.conf).reshape(1, self.H, self.W) if conf is not None \
                else torch.ones((1, self.H, self.W), device=self.conf.device, dtype=self.conf.dtype)
            self.conf = torch.cat([self.conf, conf_row], dim=0)
        if self.sky is not None:
            sky_row = sky.to(self.sky).reshape(1, self.H, self.W) if sky is not None \
                else torch.zeros((1, self.H, self.W), device=self.sky.device, dtype=self.sky.dtype)
            self.sky = torch.cat([self.sky, sky_row], dim=0)

        if self.frame_epoch is None:
            self.frame_epoch = [0] * self.N
        ep = int(epoch) if epoch is not None \
            else (max(self.frame_epoch) if self.frame_epoch else 0)
        self.frame_epoch.append(ep)

        self.N += 1
        return self.N - 1


@dataclass
class GaussianInit:
    """Output of `_build_initial_gaussians`: initial splat params + loss mask."""
    means:   torch.Tensor
    quats:   torch.Tensor
    log_s:   torch.Tensor
    logit_o: torch.Tensor
    sh:      torch.Tensor            # (M, 1, 3)
    train_mask:  torch.Tensor        # (N,H,W) bool — per-pixel supervision mask
    scene_scale: float
    conf_thresh: float | None
    voxel: float                     # voxel edge length used for init downsampling
    # Whether sky pixels were excluded from the init mask. Stored so that
    # incremental-append flows can rebuild a consistent per-frame mask for
    # the new frames without re-running the original init.
    remove_sky: bool = True

    def append_train_mask(self, mask: torch.Tensor) -> None:
        """Append a per-pixel mask (H, W) bool for one new frame, keeping
        train_mask shape in lock-step with VideoData.N after `append_frame`."""
        m = mask.to(device=self.train_mask.device, dtype=self.train_mask.dtype)
        if m.shape != self.train_mask.shape[1:]:
            raise ValueError(f"mask shape {tuple(m.shape)} != train_mask frame shape "
                             f"{tuple(self.train_mask.shape[1:])}")
        self.train_mask = torch.cat([self.train_mask, m.unsqueeze(0)], dim=0)


@dataclass
class TrainState:
    """Trainable splat parameters + per-key optimizers + optional densify
    strategy.

    `params` keys: means, scales, quats, opacities, sh0, shN. shN has
    shape (M, K_REST, 3) with K_REST=0 when sh_max_deg=0 — torch.cat with
    a zero-band tensor is a no-op so `step()` doesn't need to branch.

    One Adam per parameter is required by `gsplat.DefaultStrategy`, which
    needs to rebuild per-param state when M changes during clone/split/prune.
    When `strategy is None` the per-key dict still works identically."""
    params:          torch.nn.ParameterDict
    opts:            dict
    strategy:        object | None = None
    strategy_state:  dict | None = None


# --------------------------------------------------------------------------- #
# Free helpers — video preprocessing
# --------------------------------------------------------------------------- #


def _extract_frames(video_path: Path, out_dir: Path, max_frames: int) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total // max_frames) if max_frames > 0 else 1
    if max_frames <= 0:
        max_frames = total
    frame_paths: list[str] = []
    for i in tqdm(range(0, max_frames), desc="Extracting frames", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * stride)
        ok, bgr = cap.read()
        if not ok:
            break
        p = f"{out_dir}/f{i:04d}.png"
        cv2.imwrite(p, bgr)
        frame_paths.append(p)
    cap.release()
    print(f"{len(frame_paths)} frames @ stride {stride} from {total} total")
    return frame_paths


def _da3_inference_on_paths(frame_paths: Sequence[str],
                            device: torch.device) -> VideoData:
    """Run DA3 on an already-extracted list of frame paths and pack the
    output into a `VideoData`. Used both by `_preprocess_video` (single
    clip) and by `append_video`'s joint pass (original + appended paths
    in one DA3 inference call → single shared coordinate frame). Each
    call pays the DA3 model load cost (~few seconds + VRAM)."""
    model = DepthAnything3.from_pretrained(
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    ).to(device).eval()
    pred = model.inference(
        image=list(frame_paths),
        process_res=504,                        # DA3 default — here for clarity
        process_res_method="upper_bound_resize",
    )

    imgs  = torch.from_numpy(pred.processed_images).to(device)   # (N,H,W,3) uint8
    depth = torch.from_numpy(pred.depth).to(device)              # (N,H,W) f32
    K     = torch.from_numpy(pred.intrinsics).to(device)         # (N,3,3)
    w2c34 = torch.from_numpy(pred.extrinsics).to(device)         # (N,3,4) OpenCV w2c

    conf_np = getattr(pred, "conf", None)
    if conf_np is None:
        conf_np = getattr(pred, "confidence", None)
    conf = torch.from_numpy(conf_np).to(device) if conf_np is not None else None

    sky_np = getattr(pred, "sky", None)
    sky = torch.from_numpy(sky_np).to(device).bool() if sky_np is not None else None

    N, H, W, _ = imgs.shape
    w2c = torch.eye(4, device=device).expand(N, 4, 4).clone()
    w2c[:, :3, :4] = w2c34
    c2w = torch.linalg.inv(w2c)
    rgb = imgs.float() / 255.0

    del model, imgs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"DA3 over {N} frames: H={H} W={W}; depth range [{depth.min():.3f}, {depth.max():.3f}]")

    return VideoData(rgb=rgb, depth=depth, K=K, w2c=w2c, c2w=c2w,
                     conf=conf, sky=sky, N=N, H=H, W=W)


def _preprocess_video(video_path: Path, out_dir: Path, device: torch.device,
                      max_frames: int) -> tuple[VideoData, list[str]]:
    """Extract frames from `video_path` to disk and run DA3 on them.

    Returns `(VideoData, frame_paths)` — paths are kept around by the
    trainer so incremental-append flows can re-run DA3 jointly over the
    original frame paths plus the appended ones (one shared coordinate
    frame across clips).
    """
    frame_paths = _extract_frames(video_path, out_dir, max_frames)
    data = _da3_inference_on_paths(frame_paths, device)
    return data, frame_paths


def _average_se3(Ts: torch.Tensor) -> torch.Tensor:
    """Chordal-L2 mean of a stack of (N, 4, 4) SE(3) transforms.
    Translation: arithmetic mean of the (N, 3) translation column.
    Rotation: SVD-based projection of `sum_i R_i` onto SO(3) — this is
    the standard closed-form mean rotation that minimises the chordal
    Frobenius error and includes the reflection fix so the result is a
    proper rotation (det = +1).

    Why this over Kabsch-on-centers in `append_video`: Kabsch on camera
    *centers* fits position alignment but ignores camera *orientations*.
    Joint DA3 can produce poses with subtle angular drift from the
    original single-pass run (a "twist" around shared centers). Per-frame
    T_i = c2w_old[i] @ inv(c2w_joint[i]) and then averaging in SE(3)
    captures *both* position and orientation evidence — small angular
    residual collapses to ~0 instead of being silently absorbed by
    Kabsch's position-only fit.
    """
    assert Ts.ndim == 3 and Ts.shape[1] == 4 and Ts.shape[2] == 4, Ts.shape
    Rs = Ts[:, :3, :3]                                    # (N, 3, 3)
    ts = Ts[:, :3, 3]                                     # (N, 3)
    M = Rs.sum(dim=0)                                     # (3, 3)
    U, _, Vt = torch.linalg.svd(M)
    R = U @ Vt
    if float(torch.det(R).item()) < 0:
        Vt_fixed = Vt.clone()
        Vt_fixed[-1, :] = -Vt_fixed[-1, :]
        R = U @ Vt_fixed
    t = ts.mean(dim=0)
    T = torch.eye(4, device=Ts.device, dtype=Ts.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _kabsch_se3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Classical Kabsch / Umeyama rigid alignment — find the 4x4 SE(3)
    transform T such that `(T @ A^T)^T ≈ B` for paired point sets
    `A, B: (N, 3)`. Used by `append_video` to bring the joint-DA3 poses
    back into the original world frame (so existing splats stay valid):
    feed in joint-pass camera centers as `A` and the old single-pass
    camera centers as `B`; the returned `T` left-multiplies every
    `c2w` in the joint VideoData.

    Returns a 4x4 tensor on A's device + dtype. Requires N ≥ 3; for
    fewer points the alignment is under-determined and the caller
    should fall back to a simpler approach (per-frame copy of the
    original c2w when available).
    """
    assert A.shape == B.shape and A.ndim == 2 and A.shape[1] == 3, (A.shape, B.shape)
    A_mean = A.mean(dim=0)
    B_mean = B.mean(dim=0)
    Ac = A - A_mean
    Bc = B - B_mean
    H = Ac.T @ Bc                                         # (3, 3)
    U, _, Vt = torch.linalg.svd(H)
    # Reflection fix: ensure a proper rotation (det = +1).
    d = float(torch.sign(torch.det(Vt.T @ U.T)).item())
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=A.device, dtype=A.dtype))
    R = Vt.T @ D @ U.T
    t = B_mean - R @ A_mean
    T = torch.eye(4, device=A.device, dtype=A.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# --------------------------------------------------------------------------- #
# Free helpers — gaussian init
# --------------------------------------------------------------------------- #


def _scatter_mean(vals: torch.Tensor, inv: torch.Tensor, g: int) -> torch.Tensor:
    """Per-group mean over the `inv` partition. vals: (t,) or (t,d); returns (g,) or (g,d)."""
    ones = torch.ones(vals.shape[0], device=vals.device)
    counts = torch.zeros(g, device=vals.device).scatter_add_(0, inv, ones).clamp_min(1)
    if vals.ndim == 1:
        s = torch.zeros(g, device=vals.device).scatter_add_(0, inv, vals)
        return s / counts
    D = vals.shape[1]
    s = torch.zeros(g, D, device=vals.device).scatter_add_(0, inv[:, None].expand(-1, D), vals)
    return s / counts[:, None]


def _to_tensor(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Accept a numpy array OR a torch tensor and return a torch tensor on
    `device` with `dtype`. No-op cast if already matching. Used by
    `append_supplied_frames` to be lenient about caller payloads."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def _resize_frame_to(rgb: torch.Tensor, depth: torch.Tensor, K: torch.Tensor,
                     conf: torch.Tensor | None, sky: torch.Tensor | None,
                     H: int, W: int) -> tuple | None:
    """Resize a single (H, W, 3) RGB + (H, W) depth + (3, 3) K + optional
    masks to match (H, W). Same trick the inpainter uses at inpainter.py:919
    — INTER_AREA for RGB, INTER_NEAREST for depth/sky/conf, K rescaled by
    the resize ratios. Returns the resized tuple, or None if the input is
    malformed (caller treats None as a per-frame skip)."""
    try:
        in_H, in_W = int(rgb.shape[0]), int(rgb.shape[1])
        if (in_H, in_W) == (H, W):
            return rgb, depth, K, conf, sky
        rgb_np = rgb.detach().cpu().numpy().astype(np.float32)
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        rgb_r = cv2.resize(rgb_np, (W, H), interpolation=cv2.INTER_AREA)
        depth_r = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_NEAREST)
        sx = W / float(in_W)
        sy = H / float(in_H)
        K_r = K.clone()
        K_r[0, 0] *= sx
        K_r[1, 1] *= sy
        K_r[0, 2] *= sx
        K_r[1, 2] *= sy
        conf_r = None
        if conf is not None:
            conf_np = conf.detach().cpu().numpy().astype(np.float32)
            conf_r = cv2.resize(conf_np, (W, H), interpolation=cv2.INTER_NEAREST)
            conf_r = torch.from_numpy(conf_r).to(device=conf.device, dtype=conf.dtype)
        sky_r = None
        if sky is not None:
            sky_np = sky.detach().cpu().numpy().astype(np.uint8)
            sky_r = cv2.resize(sky_np, (W, H), interpolation=cv2.INTER_NEAREST)
            sky_r = torch.from_numpy(sky_r).to(device=sky.device).bool()
        return (
            torch.from_numpy(rgb_r).to(device=rgb.device, dtype=rgb.dtype),
            torch.from_numpy(depth_r).to(device=depth.device, dtype=depth.dtype),
            K_r,
            conf_r,
            sky_r,
        )
    except Exception as e:
        print(f"_resize_frame_to failed: {e}")
        return None


def _per_frame_train_mask(depth: torch.Tensor,
                          sky: torch.Tensor | None,
                          conf: torch.Tensor | None,
                          conf_thresh: float | None,
                          remove_sky: bool) -> torch.Tensor:
    """Build the per-pixel supervision mask for a single frame (or batch).
    Same gates `_build_initial_gaussians` writes into `init.train_mask` at
    init time, factored out so `append_video` / `append_supplied_frames` can
    rebuild it for newly-appended frames consistently.

    Accepts (H, W) or (N, H, W) tensors — output shape matches `depth`.
    """
    mask = depth > 0
    if sky is not None and remove_sky:
        mask &= ~sky
    if conf is not None and conf_thresh is not None:
        mask &= conf > conf_thresh
    return mask


def _build_initial_gaussians(data: VideoData, max_points: int,
                             confidence: float, remove_sky: bool) -> GaussianInit:
    """Unproject RGBD pixels into world-space gaussians, then voxel-downsample
    so multi-frame overlap doesn't produce redundant co-located splats. Writes
    splats_init.ply / splats_voxel.ply and renders preview snapshots as a
    side-effect (init-time diagnostics)."""
    device = data.rgb.device

    ii, jj = torch.meshgrid(
        torch.arange(data.H, device=device),
        torch.arange(data.W, device=device), indexing="ij")
    uv1 = torch.stack([jj, ii, torch.ones_like(ii)], -1).float()              # (H,W,3)

    Kinv = torch.linalg.inv(data.K)                                            # (N,3,3)
    cam_pts = torch.einsum("nij,hwj->nhwi", Kinv, uv1) * data.depth[..., None]  # (N,H,W,3)
    R, t = data.c2w[:, :3, :3], data.c2w[:, :3, 3]
    world = torch.einsum("nij,nhwj->nhwi", R, cam_pts) + t[:, None, None, :]

    valid = data.depth > 0
    print(f'masking out sky = {"False" if remove_sky else "True"}')
    if data.sky is not None and remove_sky:
        valid &= ~data.sky

    conf_thresh: float | None = None
    if data.conf is not None:
        conf_flat = data.conf.flatten()
        if conf_flat.numel() > 16_000_000:
            idx = torch.randint(0, conf_flat.numel(), (16_000_000,), device=conf_flat.device)
            conf_thresh = float(conf_flat[idx].quantile(confidence))
            print('WARNING total unfiltered splats more than 16 Million, Capping')
        else:
            conf_thresh = float(conf_flat.quantile(confidence))
        valid &= data.conf > conf_thresh

    fx_nhw    = data.K[:, 0, 0][:, None, None].expand(data.N, data.H, data.W)
    pts_full  = world[valid]
    cols_full = data.rgb[valid]
    z_full    = data.depth[valid]
    fx_full   = fx_nhw[valid]

    total_valid = pts_full.shape[0]
    if total_valid > max_points:
        sel = torch.randperm(total_valid, device=device)[:max_points]
        pts, cols, z_sel, fx_sel = pts_full[sel], cols_full[sel], z_full[sel], fx_full[sel]
        print(f'WARNING: Too many initial gaussians {total_valid} subsampling down to {max_points}')
    else:
        pts, cols, z_sel, fx_sel = pts_full, cols_full, z_full, fx_full
    M = pts.shape[0]
    subsample_ratio = max(1.0, total_valid / M)
    print(f'gaussian setup: subsample ratio {subsample_ratio}')

    # Per-point scale ≈ one texel at its depth, inflated by sqrt(subsample_ratio)
    tex = (z_sel / fx_sel * (subsample_ratio ** 0.5)).clamp_min(1e-4)

    C0 = 0.28209479177387814  # SH DC constant
    means_init   = pts.clone()
    f_dc_init    = (cols - 0.5) / C0
    log_s_init   = torch.log(tex[:, None].expand(M, 3).contiguous())
    logit_o_init = torch.full((M,), 2.1972, device=device)                    # sigmoid⁻¹(0.9)
    quats_init   = torch.zeros((M, 4), device=device); quats_init[:, 0] = 1.0
    scene_scale  = (pts.std(dim=0).mean()).item()
    print(f"M={M} gaussians (from {total_valid} valid samples), scene_scale={scene_scale:.3f}")

    sh_init = f_dc_init[:, None, :]
    save_inria_ply("splats_init.ply", means_init, sh_init, log_s_init, logit_o_init, quats_init)
    print(f"Saved splats_init.ply: {M} gaussians")
    render_and_show(data, means_init, quats_init, log_s_init, logit_o_init, sh_init, tag="init gaussians")

    # voxel init: typical reduction 3–10× vs v0 with comparable coverage; uses
    # the *full* valid-sample tensors, not the random-subsampled ones.
    voxel_frac = 0.005
    voxel = max(scene_scale * voxel_frac, 1e-4)
    print(f"voxel size: {voxel:.4f}  (scene_scale={scene_scale:.3f})")

    keys = torch.floor(pts_full / voxel).long()                               # (t, 3)
    uniq_keys, inv = torch.unique(keys, dim=0, return_inverse=True)
    G = uniq_keys.shape[0]

    means_vox = _scatter_mean(pts_full,  inv, G)
    cols_vox  = _scatter_mean(cols_full, inv, G)
    z_vox     = _scatter_mean(z_full,    inv, G)
    fx_vox    = _scatter_mean(fx_full,   inv, G)

    # scale: larger of (one texel at this depth, inflated for remaining sparsity)
    # and (half the voxel edge, so neighbouring voxels overlap).
    inflate = max(1.0, (total_valid / G) ** 0.5)
    tex_vox = (z_vox / fx_vox * inflate).clamp_min(voxel * 0.5)

    means_vx   = means_vox.clone()
    f_dc_vx    = (cols_vox - 0.5) / C0
    log_s_vx   = torch.log(tex_vox[:, None].expand(G, 3).contiguous())
    logit_o_vx = torch.full((G,), 2.1972, device=device)
    quats_vx   = torch.zeros((G, 4), device=device); quats_vx[:, 0] = 1.0
    print(f"m_voxel={G} gaussians  ({100.0 * G / M:.1f}% of v0's {M}, inflate={inflate:.2f})")

    sh_vx = f_dc_vx[:, None, :]
    save_inria_ply("splats_voxel.ply", means_vx, sh_vx, log_s_vx, logit_o_vx, quats_vx)
    print("saved splats_voxel.ply")
    render_and_show(data, means_vx, quats_vx, log_s_vx, logit_o_vx, sh_vx, tag="v0 voxel init")

    # Per-pixel loss mask, indexed per-frame by step() as train_mask[idx].
    train_mask = _per_frame_train_mask(
        data.depth, data.sky, data.conf, conf_thresh, remove_sky,
    )
    print(f"loss mask: {train_mask.float().mean().item()*100:.1f}% pixels kept across {data.N} frames")

    return GaussianInit(
        means=means_vx, quats=quats_vx, log_s=log_s_vx, logit_o=logit_o_vx,
        sh=sh_vx, train_mask=train_mask,
        scene_scale=scene_scale, conf_thresh=conf_thresh,
        voxel=voxel, remove_sky=bool(remove_sky),
    )


# --------------------------------------------------------------------------- #
# Free helpers — train state, rendering, PLY save
# --------------------------------------------------------------------------- #


def _make_train_state(init: GaussianInit, sh_max_deg: int = 0,
                      use_densify: bool = False,
                      densify_total_steps: int = 7000,
                      mode: str = "3dgs") -> TrainState:
    """Build the ParameterDict + per-key Adam optimizers + (optional)
    DefaultStrategy.

    `sh_max_deg > 0` allocates shN of shape (M, (sh_max_deg+1)²−1, 3) with
    its own Adam at 1/20 of the sh0 LR. `use_densify=True` wires up the
    gsplat clone/split/prune strategy; `means` LR is scaled by scene_scale
    to match the v3 notebook recipe. `mode="2dgs"` switches the densify
    strategy's gradient key to `gradient_2dgs` and disables opacity reset
    (2DGS is very sensitive to it — matches the notebook recipe)."""
    torch.set_grad_enabled(True)
    device = init.means.device
    M = init.means.shape[0]
    K_rest = max(0, (sh_max_deg + 1) ** 2 - 1)

    params = torch.nn.ParameterDict({
        "means":     torch.nn.Parameter(init.means.clone()),
        "scales":    torch.nn.Parameter(init.log_s.clone()),
        "quats":     torch.nn.Parameter(init.quats.clone()),
        "opacities": torch.nn.Parameter(init.logit_o.clone()),
        "sh0":       torch.nn.Parameter(init.sh.clone()),
        "shN":       torch.nn.Parameter(
            torch.zeros((M, K_rest, 3), device=device, dtype=init.means.dtype)
        ),
    }).to(device)

    # One Adam per parameter — required by DefaultStrategy so it can rebuild
    # per-param Adam state when M changes during clone/split/prune.
    means_lr = 1.6e-4 * init.scene_scale if use_densify else 1.6e-4
    lr_table = {
        "means":     means_lr,
        "scales":    5e-3,
        "quats":     1e-3,
        "opacities": 5e-2,
        "sh0":       2.5e-3,
        "shN":       2.5e-3 / 20.0,
    }
    opts = {
        k: torch.optim.Adam([{"params": [params[k]], "lr": lr}])
        for k, lr in lr_table.items()
    }

    strategy = None
    strategy_state = None
    if use_densify:
        if mode == "2dgs":
            # 2DGS recipe (matches notebook cell 7): opacity reset disabled,
            # gradient key switched to the 2DGS-specific accumulator.
            strategy = DefaultStrategy(
                refine_start_iter=500,
                refine_stop_iter=max(500, densify_total_steps - 500),
                reset_every=densify_total_steps + 1,
                refine_every=100,
                key_for_gradient="gradient_2dgs",
                verbose=False,
            )
        else:
            # Aggressive 3DGS schedule: start ~2.5x sooner, densify 2x more
            # often, reset opacity 2x more often, refine longer at the tail.
            strategy = DefaultStrategy(
                refine_start_iter=200,                                  # was 500
                refine_stop_iter=max(500, densify_total_steps - 200),   # was -500
                reset_every=1500,                                       # was 3000
                refine_every=50,                                        # was 100
                verbose=False,
            )
        strategy.check_sanity(params, opts)
        strategy_state = strategy.initialize_state(scene_scale=init.scene_scale)

    return TrainState(params=params, opts=opts,
                      strategy=strategy, strategy_state=strategy_state)


def render_view(data: VideoData, means, quats, log_s, logit_o, sh_all, cam_idx,
                mode: str = "3dgs"):
    sh_deg = int(round(sh_all.shape[1] ** 0.5)) - 1
    if mode == "2dgs":
        out, *_ = rasterization_2dgs(
            means, quats, torch.exp(log_s), torch.sigmoid(logit_o), sh_all,
            data.w2c[cam_idx:cam_idx+1], data.K[cam_idx:cam_idx+1], data.W, data.H,
            sh_degree=sh_deg, packed=False, render_mode="RGB")
        return out[0].clamp(0, 1)
    out, _, _ = rasterization(
        means, quats, torch.exp(log_s), torch.sigmoid(logit_o), sh_all,
        data.w2c[cam_idx:cam_idx+1], data.K[cam_idx:cam_idx+1], data.W, data.H,
        sh_degree=sh_deg, packed=False)
    return out[0].clamp(0, 1)


def render_and_show(data: VideoData, means, quats, log_s, logit_o, sh_all,
                    tag: str, n_views: int = 4, mode: str = "3dgs"):
    idxs = torch.linspace(0, data.N - 1, n_views).round().long().tolist()
    fig, ax = plt.subplots(2, n_views, figsize=(4 * n_views, 8))
    with torch.no_grad():
        for c, i in enumerate(idxs):
            r = render_view(data, means, quats, log_s, logit_o, sh_all, i,
                            mode=mode).cpu().numpy()
            ax[0, c].imshow(data.rgb[i].cpu().numpy()); ax[0, c].set_title(f"gt frame {i}"); ax[0, c].axis("off")
            ax[1, c].imshow(r);                         ax[1, c].set_title(f"{tag} cam {i}"); ax[1, c].axis("off")
    plt.suptitle(tag)
    plt.tight_layout()
    plt.savefig(f"{tag}.png", dpi=150, bbox_inches='tight')
    #plt.show()


def render_mask_diagnostic(data: VideoData, init: GaussianInit,
                           means, quats, log_s, logit_o, sh_all,
                           tag: str, n_views: int = 4, mode: str = "3dgs") -> None:
    """Per-frame diagnostic: shows what splats are doing inside the regions
    that `train_mask` filtered out. For `n_views` evenly-spaced frames,
    plots one row of: GT | GT+mask (red on excluded) | render | render·mask
    (what L1 sees) | leak (render·~mask on grey — splats inside excluded
    regions). Saves `{tag}_mask_diag.png`."""
    idxs = torch.linspace(0, data.N - 1, n_views).round().long().tolist()
    titles = ["GT", "GT + mask (red = excluded)", "render",
              "render · mask (what L1 sees)", "leak  (render · ~mask)"]
    n_cols = len(titles)
    fig, ax = plt.subplots(n_views, n_cols, figsize=(4 * n_cols, 4 * n_views))
    if n_views == 1:
        ax = ax[None, :]
    grey = 0.5
    with torch.no_grad():
        for r, i in enumerate(idxs):
            gt = data.rgb[i].cpu().numpy()
            mask = init.train_mask[i].cpu().numpy().astype(bool)
            rend = render_view(data, means, quats, log_s, logit_o, sh_all, i,
                               mode=mode).cpu().numpy()

            overlay = gt.copy()
            alpha = 0.45
            overlay[~mask, 0] = (1 - alpha) * overlay[~mask, 0] + alpha * 1.0
            overlay[~mask, 1] = (1 - alpha) * overlay[~mask, 1]
            overlay[~mask, 2] = (1 - alpha) * overlay[~mask, 2]

            rend_masked = rend.copy()
            rend_masked[~mask] = 0.0

            leak = np.full_like(rend, grey)
            leak[~mask] = rend[~mask]

            panels = [gt, overlay, rend, rend_masked, leak]
            for c, img in enumerate(panels):
                ax[r, c].imshow(np.clip(img, 0, 1))
                if r == 0:
                    ax[r, c].set_title(titles[c])
                if c == 0:
                    ax[r, c].set_ylabel(f"frame {i}", rotation=90, labelpad=10)
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    kept_pct = init.train_mask.float().mean().item() * 100.0
    plt.suptitle(f"{tag} — mask diagnostic  (train_mask keeps {kept_pct:.1f}% of pixels)")
    plt.tight_layout()
    plt.savefig(f"{tag}_mask_diag.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def splat_stats_diagnostic(init: GaussianInit, means, log_s, logit_o,
                           tag: str, mode: str = "3dgs",
                           log_scale_max: float | None = None) -> None:
    """Histograms of splat shape stats + a printed top-10 by max-axis size,
    so big flat-faced splats can be located by index. Saves
    `{tag}_splat_stats.png`."""
    with torch.no_grad():
        scales = torch.exp(log_s.detach())               # (M, 3)
        opa    = torch.sigmoid(logit_o.detach())         # (M,)
        m      = means.detach()
    voxel = max(init.voxel, 1e-8)
    max_axis = scales.max(dim=1).values
    max_axis_vx = (max_axis / voxel).cpu().numpy()

    if mode == "2dgs":
        pair = scales[:, :2]
    else:
        pair = scales.topk(2, dim=1).values
    pair_sorted, _ = torch.sort(pair, dim=1)
    needle = (pair_sorted[:, 1] / pair_sorted[:, 0].clamp_min(1e-8)).cpu().numpy()
    opa_np = opa.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(max_axis_vx, bins=80, log=True)
    axes[0].set_xlabel("max axis scale (× voxel)")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("Splat size")
    if log_scale_max is not None:
        clamp_vx = math.exp(log_scale_max) / voxel
        axes[0].axvline(clamp_vx, color="r", ls="--",
                        label=f"clamp = {clamp_vx:.2f}× voxel")
        axes[0].legend()

    axes[1].hist(opa_np, bins=80, range=(0, 1))
    axes[1].set_xlabel("opacity (sigmoid)")
    axes[1].set_title("Opacity")

    axes[2].hist(needle, bins=80, log=True)
    axes[2].set_xlabel("needle-ness (larger / smaller in-plane axis)")
    axes[2].set_title("Disk anisotropy" if mode == "2dgs"
                      else "3DGS top-2-axes aniso")

    M = int(scales.shape[0])
    plt.suptitle(f"{tag} — splat stats  (M={M}, voxel={voxel:.4f}, "
                 f"scene_scale={init.scene_scale:.3f})")
    plt.tight_layout()
    plt.savefig(f"{tag}_splat_stats.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    top = torch.topk(max_axis, k=min(10, max_axis.numel()))
    idxs = top.indices.cpu().numpy()
    print(f"[{tag}] top-{len(idxs)} splats by max-axis scale (voxel units):")
    print(f"  {'idx':>7s}  {'max_vx':>8s}  {'opa':>6s}  {'needle':>8s}  "
          f"{'x':>8s} {'y':>8s} {'z':>8s}")
    for gi in idxs:
        x, y, z = m[gi].cpu().tolist()
        print(f"  {gi:>7d}  {max_axis_vx[gi]:>8.2f}  {opa_np[gi]:>6.3f}  "
              f"{needle[gi]:>8.2f}  {x:>8.3f} {y:>8.3f} {z:>8.3f}")


def save_inria_ply(path, means, sh_all, log_s, logit_o, quats):
    """sh_all: (M, K, 3) where K = (sh_deg+1)**2. Writes Inria 3DGS PLY with matching f_rest fields."""
    n = means.shape[0]
    K_sh = sh_all.shape[1]
    K_rest = K_sh - 1
    base_fields = [("x","f4"),("y","f4"),("z","f4"),
                   ("nx","f4"),("ny","f4"),("nz","f4"),
                   ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4")]
    rest_fields = [(f"f_rest_{i}", "f4") for i in range(3 * K_rest)]
    tail_fields = [("opacity","f4"),
                   ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
                   ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4")]
    arr = np.zeros(n, dtype=base_fields + rest_fields + tail_fields)
    m = means.detach().cpu().numpy()
    sh_np = sh_all.detach().cpu().numpy()
    s = log_s.detach().cpu().numpy()
    q = quats.detach().cpu().numpy()
    o = logit_o.detach().cpu().numpy()
    arr["x"], arr["y"], arr["z"] = m[:,0], m[:,1], m[:,2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = sh_np[:,0,0], sh_np[:,0,1], sh_np[:,0,2]
    if K_rest > 0:
        rest = sh_np[:, 1:, :].transpose(0, 2, 1).reshape(n, 3 * K_rest)
        for i in range(3 * K_rest):
            arr[f"f_rest_{i}"] = rest[:, i]
    arr["opacity"] = o
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = s[:,0], s[:,1], s[:,2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = q[:,0], q[:,1], q[:,2], q[:,3]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)


def load_inria_ply(path: str | Path, device: torch.device) -> dict:
    """Inverse of `save_inria_ply` for trainer re-ingestion.

    Returns the *raw, pre-activation* tensors that go straight back into
    `train.params` — opacity stays as logit, scales stay in log-space, quats
    are not normalized, no coordinate flip. This is intentionally different
    from `viewer.PlyLoader.load`, which post-activates and flips for display.

    Returned dict:
      means:    (M, 3)
      log_s:    (M, 3)
      quats:    (M, 4)  wxyz
      logit_o:  (M,)
      sh0:      (M, 1, 3)
      shN:      (M, K_rest, 3)  K_rest = (sh_max_deg+1)**2 - 1, may be 0
      sh_max_deg: int
    """
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    prop_names = v.dtype.names

    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    logit_o = v["opacity"].astype(np.float32)
    log_s = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    sh0 = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)[:, None, :]

    rest_props = sorted(
        (n for n in prop_names if n.startswith("f_rest_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_props:
        assert len(rest_props) % 3 == 0
        K_minus_1 = len(rest_props) // 3
        K = K_minus_1 + 1
        sh_max_deg = int(round(math.sqrt(K))) - 1
        assert (sh_max_deg + 1) ** 2 == K, f"f_rest count {len(rest_props)} not a valid SH degree"
        rest_np = np.stack([v[n] for n in rest_props], axis=-1).astype(np.float32)
        # save_inria_ply wrote rest = sh_np[:, 1:, :].transpose(0,2,1).reshape(n, 3*K_rest)
        # so to invert: reshape (n, 3, K_rest) then transpose back to (n, K_rest, 3).
        shN = rest_np.reshape(-1, 3, K_minus_1).transpose(0, 2, 1)
    else:
        sh_max_deg = 0
        shN = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

    return {
        "means":    torch.from_numpy(means).to(device),
        "log_s":    torch.from_numpy(log_s).to(device),
        "quats":    torch.from_numpy(quats).to(device),
        "logit_o":  torch.from_numpy(logit_o).to(device),
        "sh0":      torch.from_numpy(sh0).to(device),
        "shN":      torch.from_numpy(shN).to(device),
        "sh_max_deg": sh_max_deg,
    }


_COMPONENT_ABBR = {
    "l1": "l1", "void": "void", "lpips": "lpips",
    "distortion": "dist", "normal_consistency": "nrm",
    "depth_sup": "dep", "da3_normal": "da3",
}


def format_loss_components(components: dict[str, float]) -> str:
    """Compact `k=v` rendering of a per-term loss dict, sorted by magnitude.
    Used by the trainer's tqdm postfix and the GUI loss-breakdown panel."""
    if not components:
        return "(no components)"
    items = sorted(components.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return " ".join(f"{_COMPONENT_ABBR.get(k, k)}={v:.4g}" for k, v in items)


# --------------------------------------------------------------------------- #
# SplatTrainer: lifecycle + orchestration + scene publishing
# --------------------------------------------------------------------------- #


class SplatTrainer:
    """Barebones trainer. Holds three lifecycle bundles
    (`data` / `init` / `train`) and orchestrates them via `prepare_and_init`
    + `step` + `reset`. The class shape matches the `Trainer`-protocol seam
    in `visergui/training.py` so `step` can be driven by
    `BackgroundTrainingThread`."""

    def __init__(
        self,
        output_root: Path = Path("outputs"),
        device: str = "cuda",
        max_points: int = 1_000_000,
        scene: "SceneState | None" = None,
        publish_every: int = 25,
    ) -> None:
        self.output_root = Path(output_root)
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.max_points = max_points

        # Live-viewer bridge. `scene` is the shared SceneState whose splat
        # tensors the viewer reads; `publish_every` throttles how often we
        # detach + activate working tensors into it.
        self.scene = scene
        self.publish_every = int(publish_every)
        self._step_count = 0
        self._initialized = False
        self._last_video: tuple[Path, int] | None = None
        self._lpips_net: torch.nn.Module | None = None

        # Loss / SH config. Defaults reproduce v1 behavior (sh=0, L1-only,
        # static splat count). Live tunables set via `prepare_and_init`.
        self.sh_max_deg: int = 0
        self.sh_ramp_steps_per_band: int = 1000
        self.l1_weight: float = 1.0
        self.lpips_weight: float = 0.0
        # Penalty on rendered alpha inside `~train_mask` — drives splats to
        # render nothing in filtered regions, removing the asymmetric "free
        # growth" at mask boundaries. 0 disables.
        self.void_weight: float = 0.5
        self.use_densify: bool = False
        self.densify_total_steps: int = 7000
        # Max gaussian axis-scale as a multiple of the init voxel edge. The
        # log-space cap is recomputed in prepare_and_init from init.voxel.
        self.scale_clamp_voxel_mult: float = 2.0
        self._log_scale_max: float | None = None

        # 2DGS mode and its extra loss terms. All weights are ignored in
        # "3dgs" mode. Defaults match the notebook recipe in
        # video_to_2dsplats.ipynb cell 7.
        self.mode: str = "3dgs"                  # "3dgs" or "2dgs"
        self.distortion_weight: float = 1.0
        self.normal_consistency_weight: float = 0.05
        self.depth_sup_weight: float = 0.5
        self.da3_normal_weight: float = 0.05
        self.dist_warmup_steps: int = 700
        self.normal_warmup_steps: int = 1600
        self._da3_normals: torch.Tensor | None = None

        # Last step's per-term loss breakdown — populated by `step()` and
        # read by the tqdm postfix + GUI status panel.
        self._last_loss_components: dict[str, float] = {}

        self.data:  VideoData    | None = None
        self.init:  GaussianInit | None = None
        self.train: TrainState   | None = None

        # Per-splat session tag — length M, written by `prepare_and_init`
        # (all zeros) and extended by `_seed_splats_for_new_frames` (Phase 2)
        # with the appended frames' epoch. Filtered through `prune_splats`
        # and reset by `load_checkpoint` / `reset`. Used by Phase 5's
        # voxel-overlap / coverage-by-epoch visualizations.
        self._splat_epoch: torch.Tensor | None = None

        # Disk paths of the frames currently in `self.data`, in row order.
        # Populated by `prepare_and_init` (length N0) and extended by
        # `append_video` after extracting the new clip; needed so we can
        # re-run DA3 over the union (original + appended) on each append
        # and get a single consistent coordinate frame across clips.
        self._frame_paths: list[str] | None = None

        # Phase 3: per-step frame sampler. "uniform" (default) matches the
        # original `torch.randint(0, N, 1)` behaviour. "stratified" samples
        # epoch first (latest epoch gets weight `_new_frame_weight`, every
        # other epoch gets 1.0), then a frame within that epoch. "scheduled"
        # additionally interpolates from "sample only the newest epoch" at
        # step 0 to the stratified target by `_sampling_horizon` — biases
        # toward freshly-appended frames early so the new region integrates
        # before the gradient mixes back to the rest of the scene.
        self._sampling_mode: str = "uniform"
        self._new_frame_weight: float = 1.0
        self._sampling_horizon: int = 1000

        # Phase 2 tuning: coarse-voxel dedup radius for `_seed_splats_…`.
        # The init voxel is ~0.5% of scene scale; joint DA3 pose noise is
        # often a few voxels, so an exact voxel-key match misses near
        # duplicates. We dedup at `init.voxel * _seed_dedup_multiplier` and
        # only voxel-downsample (the compactness step) at `init.voxel`.
        # Default 2.0 = "occupied if there's an existing splat within ~2
        # init voxels". Bump higher (3-4) if you still see new splats
        # piling onto existing geometry after an append; lower (1.0) if
        # dedup is over-rejecting candidates in low-overlap regions.
        self._seed_dedup_multiplier: float = 2.0

        # Phase 4: per-splat freeze mask. True = trainable (gradient + Adam
        # step applied; densify may clone/split). False = frozen (gradient
        # zeroed before optim.step). `_freeze_mode == "off"` (default) means
        # no freezing — `_frozen_mask` stays None and step() behaves
        # identically to pre-Phase-4. When a freeze mode is active and the
        # mask is computed, densify is skipped too so splat-count changes
        # don't desync the mask. Caller must click "Recompute freeze mask"
        # after every append (or when they want to refresh).
        self._freeze_mode: str = "off"
        self._frozen_mask: torch.Tensor | None = None

    # ---- External-compat forwarders ------------------------------------- #
    # Returned references stay valid across densify swaps because they go
    # through `self.train.params[...]` on each access — DefaultStrategy
    # replaces the Parameter object in the dict in-place.
    @property
    def means_t(self) -> torch.Tensor:
        if self.train is None: raise AttributeError("means_t")
        return self.train.params["means"]
    @property
    def log_s_t(self) -> torch.Tensor:
        if self.train is None: raise AttributeError("log_s_t")
        return self.train.params["scales"]
    @property
    def quats_t(self) -> torch.Tensor:
        if self.train is None: raise AttributeError("quats_t")
        return self.train.params["quats"]
    @property
    def logit_o_t(self) -> torch.Tensor:
        if self.train is None: raise AttributeError("logit_o_t")
        return self.train.params["opacities"]
    @property
    def sh0_t(self) -> torch.Tensor:
        if self.train is None: raise AttributeError("sh0_t")
        return self.train.params["sh0"]

    # save_inria_ply stays callable as a method for train_and_view.py.
    save_inria_ply = staticmethod(save_inria_ply)

    # ---- Orchestration -------------------------------------------------- #

    def prepare_and_init(self, video: Path, max_frames: int,
                         confidence: float, remove_sky: bool,
                         name: str | None = None,
                         sh_max_deg: int = 0,
                         lpips_weight: float = 0.0,
                         l1_weight: float = 1.0,
                         void_weight: float = 0.5,
                         sh_ramp_steps_per_band: int = 1000,
                         use_densify: bool = False,
                         densify_total_steps: int = 7000,
                         mode: str = "3dgs") -> None:
        """One-call init: preprocess video (skipped if same video already
        processed), build initial gaussians, build train state, and publish
        to the scene.

        `sh_max_deg` > 0 enables SH-band progression (0→sh_max_deg) ramped
        every `sh_ramp_steps_per_band` steps. `lpips_weight` > 0 adds an
        LPIPS perceptual term to the L1 loss. `use_densify` wires up the
        gsplat clone/split/prune strategy; `densify_total_steps` sets when
        the strategy stops refining (refine_stop_iter = total - 500).
        `mode="2dgs"` switches the rasterizer to `gsplat.rasterization_2dgs`
        and adds the notebook's 2DGS loss recipe (distortion, normal
        consistency, DA3 depth + normal supervision)."""
        if mode not in ("3dgs", "2dgs"):
            raise ValueError(f"mode must be '3dgs' or '2dgs', got {mode!r}")
        self.mode = mode
        self.sh_max_deg = int(sh_max_deg)
        self.lpips_weight = float(lpips_weight)
        self.l1_weight = float(l1_weight)
        self.void_weight = float(void_weight)
        self.sh_ramp_steps_per_band = max(1, int(sh_ramp_steps_per_band))
        self.use_densify = bool(use_densify)
        self.densify_total_steps = int(densify_total_steps)

        video = Path(video)
        signature = (video, int(max_frames))
        self.name = name or video.stem
        out_dir = self.output_root / self.name
        if self._last_video != signature or self.data is None:
            self.data, frame_paths = _preprocess_video(
                video, out_dir, self.device, max_frames,
            )
            self._frame_paths = list(frame_paths)
            self._last_video = signature
        self.init = _build_initial_gaussians(
            self.data, self.max_points,
            confidence=confidence, remove_sky=remove_sky,
        )
        self.train = _make_train_state(
            self.init, sh_max_deg=self.sh_max_deg,
            use_densify=self.use_densify,
            densify_total_steps=self.densify_total_steps,
            mode=self.mode,
        )
        # Precompute per-pixel world-space normals from DA3 depth for the
        # 2DGS DA3 normal-supervision term. Same convention as gsplat's
        # surf_normals (depth_to_normal of rendered depth), so the (1-cos)
        # term pins splat orientations to actual surfaces.
        if self.mode == "2dgs" and self.da3_normal_weight > 0.0:
            with torch.no_grad():
                self._da3_normals = depth_to_normal(
                    self.data.depth[..., None], self.data.c2w, self.data.K,
                )
        else:
            self._da3_normals = None
        # Cap each gaussian axis-scale at `scale_clamp_voxel_mult * voxel`.
        # Clamping is applied after each optimizer step in log-space. Skipped
        # in 2DGS mode — 2DGS disks routinely need scales an order of
        # magnitude larger than voxel to cover smooth surfaces, and the
        # 3DGS-tuned clamp prevents L1 from converging (use the post-train
        # Prune Splats button to catch outliers instead, matching the
        # notebook recipe).
        if self.mode == "2dgs":
            self._log_scale_max = None
        else:
            self._log_scale_max = math.log(self.scale_clamp_voxel_mult * self.init.voxel)
        # Phase 2: all initial splats belong to epoch 0.
        M0 = int(self.train.params["means"].shape[0])
        self._splat_epoch = torch.zeros(M0, dtype=torch.int32, device=self.device)
        self._publish_to_scene()
        self._initialized = True

    def _current_sh_degree(self) -> int:
        """SH band currently being optimized. Ramps 0→sh_max_deg, one band
        every `sh_ramp_steps_per_band` steps."""
        if self.sh_max_deg <= 0:
            return 0
        return min(self.sh_max_deg, self._step_count // self.sh_ramp_steps_per_band)

    def _get_lpips_net(self) -> torch.nn.Module:
        """Lazy-construct the LPIPS net the first time it's needed. Params
        are frozen — we only use it as a fixed perceptual metric."""
        if self._lpips_net is None:
            import lpips as _lpips_pkg
            net = _lpips_pkg.LPIPS(net="vgg", verbose=False).to(self.device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._lpips_net = net
        return self._lpips_net

    def set_seed_dedup_multiplier(self, mult: float) -> None:
        """Phase 2 tunable: set the coarse dedup radius for the next call to
        `_seed_splats_for_new_frames`. Units are multiples of `init.voxel`.
        Cheap (single attribute assign); safe to call any time."""
        self._seed_dedup_multiplier = max(0.1, float(mult))

    def set_scale_clamp(self, voxel_mult: float) -> None:
        """Update the per-axis scale cap as a multiple of the init voxel edge.
        Takes effect on the next `step()`. Safe to call before init (the cap
        will be re-derived from `init.voxel` once prepare_and_init runs).
        In 2DGS mode the clamp is disabled — see `prepare_and_init`."""
        self.scale_clamp_voxel_mult = float(voxel_mult)
        if self.init is not None and self.mode != "2dgs":
            self._log_scale_max = math.log(self.scale_clamp_voxel_mult * self.init.voxel)

    def render_diagnostics(self, tag: str) -> None:
        """Render mask + splat-stats diagnostics for the current params.
        See `render_mask_diagnostic` and `splat_stats_diagnostic`. Safe to
        call any time after `prepare_and_init`."""
        if self.train is None or self.data is None or self.init is None:
            return
        p = self.train.params
        sh_all = (p["sh0"] if p["shN"].shape[1] == 0
                  else torch.cat([p["sh0"], p["shN"]], dim=1))
        render_mask_diagnostic(self.data, self.init, p["means"], p["quats"],
                               p["scales"], p["opacities"], sh_all,
                               tag=tag, mode=self.mode)
        splat_stats_diagnostic(self.init, p["means"], p["scales"],
                               p["opacities"], tag=tag, mode=self.mode,
                               log_scale_max=self._log_scale_max)

    def save_current(self, path: str) -> None:
        """Save the current splats as an Inria-format PLY. Concatenates
        sh0 + shN automatically when higher SH bands are trained."""
        if self.train is None:
            return
        p = self.train.params
        sh_all = (p["sh0"] if p["shN"].shape[1] == 0
                  else torch.cat([p["sh0"], p["shN"]], dim=1))
        save_inria_ply(path, p["means"], sh_all,
                       p["scales"], p["opacities"], p["quats"])

    # ---- Phase 1: incremental training checkpoints --------------------- #

    def save_checkpoint(self, path: str | Path) -> dict:
        """Save trained splats as an Inria PLY plus a sidecar JSON capturing
        the bits `load_checkpoint` needs to rebuild a consistent
        `GaussianInit` / `TrainState` shell (step count, voxel, scene_scale,
        conf_thresh, rasterizer mode, sh_max_deg, scale-clamp setting).

        The sidecar lives next to the PLY (same stem, `.json` suffix). Adam
        moments and densify-strategy state are NOT saved — they rebuild
        fresh on load (Phase 6 will add full optimizer-state checkpointing).
        Returns the metadata dict that was written.
        """
        if self.train is None or self.init is None:
            raise RuntimeError("call prepare_and_init before save_checkpoint")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # PLY
        self.save_current(str(path))
        # Sidecar
        meta = {
            "schema_version": 1,
            "step_count":     int(self._step_count),
            "scene_scale":    float(self.init.scene_scale),
            "voxel":          float(self.init.voxel),
            "conf_thresh":    (None if self.init.conf_thresh is None
                               else float(self.init.conf_thresh)),
            "remove_sky":     bool(self.init.remove_sky),
            "sh_max_deg":     int(self.sh_max_deg),
            "mode":           str(self.mode),
            "scale_clamp_voxel_mult": float(self.scale_clamp_voxel_mult),
            "use_densify":    bool(self.use_densify),
            "densify_total_steps":    int(self.densify_total_steps),
            "splat_count":    int(self.train.params["means"].shape[0]),
            "frame_count":    int(self.data.N if self.data is not None else 0),
        }
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps(meta, indent=2))
        print(f"saved checkpoint: {path} (+ {sidecar.name}) — "
              f"{meta['splat_count']:,} splats @ step {meta['step_count']}")
        return meta

    def load_checkpoint(self, ply_path: str | Path,
                        sidecar_path: str | Path | None = None) -> dict:
        """Replace the current trained splats with those in `ply_path`.

        Requires `self.data` to already be loaded (i.e. `prepare_and_init`
        was called on the original video). The sidecar JSON supplies the
        `scene_scale` / `voxel` / `conf_thresh` / `remove_sky` so that the
        existing per-frame loss mask + scale-clamp stay consistent with how
        the saved splats were trained.

        Adam state is NOT restored — splats will wobble for the first few
        hundred steps after resume. Densify strategy state is re-initialised.
        Sidecar is optional: if missing, we fall back to the trainer's
        current settings (you'll get a warning if voxel was inferred).
        """
        if self.data is None:
            raise RuntimeError(
                "load_checkpoint requires data to be loaded — call "
                "prepare_and_init on the source video first"
            )
        ply_path = Path(ply_path)
        if sidecar_path is None:
            sidecar_path = ply_path.with_suffix(".json")
        sidecar_path = Path(sidecar_path)

        loaded = load_inria_ply(ply_path, self.device)
        meta: dict = {}
        if sidecar_path.exists():
            meta = json.loads(sidecar_path.read_text())
        else:
            print(f"warning: sidecar {sidecar_path} not found — using current trainer settings")

        sh_max_deg = int(meta.get("sh_max_deg", loaded["sh_max_deg"]))
        scene_scale = float(meta.get("scene_scale",
                                     self.init.scene_scale if self.init else 1.0))
        voxel = float(meta.get("voxel",
                               self.init.voxel if self.init else 1e-2))
        conf_thresh = meta.get("conf_thresh",
                               self.init.conf_thresh if self.init else None)
        if conf_thresh is not None:
            conf_thresh = float(conf_thresh)
        remove_sky = bool(meta.get("remove_sky",
                                   self.init.remove_sky if self.init else True))
        mode = str(meta.get("mode", self.mode))
        scale_clamp = float(meta.get("scale_clamp_voxel_mult",
                                     self.scale_clamp_voxel_mult))
        use_densify = bool(meta.get("use_densify", self.use_densify))
        densify_total = int(meta.get("densify_total_steps",
                                     self.densify_total_steps))

        # Trainer-level switches that affect rasterizer + step()
        self.mode = mode
        self.sh_max_deg = sh_max_deg
        self.use_densify = use_densify
        self.densify_total_steps = densify_total

        # Rebuild the per-frame loss mask against the current `self.data`
        # using the checkpoint's mask gates (so the supervision matches what
        # produced the saved splats).
        train_mask = _per_frame_train_mask(
            self.data.depth, self.data.sky, self.data.conf,
            conf_thresh, remove_sky,
        )
        self.init = GaussianInit(
            means=loaded["means"], quats=loaded["quats"],
            log_s=loaded["log_s"], logit_o=loaded["logit_o"],
            sh=loaded["sh0"], train_mask=train_mask,
            scene_scale=scene_scale, conf_thresh=conf_thresh,
            voxel=voxel, remove_sky=remove_sky,
        )
        # Build a fresh TrainState from the loaded params. Adam moments are
        # zero — see method docstring. We then patch in the higher SH bands
        # (`shN`) from the checkpoint since `_make_train_state` always
        # allocates a *zeroed* shN of the requested capacity.
        self.train = _make_train_state(
            self.init, sh_max_deg=sh_max_deg,
            use_densify=use_densify, densify_total_steps=densify_total,
            mode=mode,
        )
        if loaded["shN"].shape[1] > 0:
            with torch.no_grad():
                shN_capacity = self.train.params["shN"].shape[1]
                copy_bands = min(shN_capacity, loaded["shN"].shape[1])
                if copy_bands > 0:
                    self.train.params["shN"][:, :copy_bands, :] = loaded["shN"][:, :copy_bands, :]

        # Restore step count + scale clamp + DA3 normals. Splat-epoch tags
        # are not in the v1 sidecar (Phase 1 saves only PLY), so we reset to
        # all-zeros for the loaded splat count — they'll get bumped again
        # the next time `_seed_splats_for_new_frames` runs.
        M_loaded = int(self.train.params["means"].shape[0])
        self._splat_epoch = torch.zeros(M_loaded, dtype=torch.int32, device=self.device)
        # Phase 4: any freeze mask from before the load is now stale (the
        # loaded splats are a different set entirely). Caller can recompute.
        self._frozen_mask = None
        self._step_count = int(meta.get("step_count", 0))
        self.scale_clamp_voxel_mult = scale_clamp
        if self.mode == "2dgs":
            self._log_scale_max = None
        else:
            self._log_scale_max = math.log(self.scale_clamp_voxel_mult * self.init.voxel)
        if self.mode == "2dgs" and self.da3_normal_weight > 0.0:
            with torch.no_grad():
                self._da3_normals = depth_to_normal(
                    self.data.depth[..., None], self.data.c2w, self.data.K,
                )
        else:
            self._da3_normals = None

        self._publish_to_scene()
        self._initialized = True
        info = {
            "splat_count":  int(self.train.params["means"].shape[0]),
            "step_count":   int(self._step_count),
            "sh_max_deg":   sh_max_deg,
            "mode":         mode,
            "frame_count":  int(self.data.N),
            "sidecar_used": sidecar_path.exists(),
        }
        print(f"loaded checkpoint: {ply_path} — {info['splat_count']:,} splats "
              f"@ step {info['step_count']} (sh={info['sh_max_deg']}, mode={info['mode']})")
        return info

    def append_video(self, video_path: str | Path, max_frames: int = -1,
                     name: str | None = None,
                     seed_new_splats: bool = False) -> dict:
        """Run DA3 on a new clip and append every frame to `self.data` +
        extend `self.init.train_mask`. New frames get a new epoch tag
        (`max(frame_epoch) + 1`) so Phase 3 sampling + Phase 5 visual debug
        can distinguish them.

        When `seed_new_splats=True`, also runs Phase 2's
        `_seed_splats_for_new_frames` on the freshly-appended indices, so
        new viewpoints get splat coverage immediately rather than waiting
        for densification (Phase 1 leaves splat count unchanged).

        Returns `{"first_new_idx", "n_added", "epoch", "skipped",
                  "n_candidates", "n_after_dedup", "n_seeded"}`. The
        `n_*` seed fields are 0 when `seed_new_splats=False`.
        """
        if self.data is None or self.init is None:
            raise RuntimeError("call prepare_and_init before append_video")
        if not self._frame_paths:
            raise RuntimeError(
                "append_video needs the original clip's frame paths cached on "
                "the trainer (set by prepare_and_init). State looks inconsistent."
            )
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")
        out_dir = self.output_root / (name or f"{video_path.stem}_append")

        # Phase 2.5: run DA3 ONCE over (original + new) frame paths so we
        # get a single coordinate frame across clips. Per-clip DA3 picks an
        # independent reference view, so without this the new frames'
        # c2w live in a different world frame than the existing splats →
        # "duplicates at wrong locations" rather than dedup-and-merge.
        new_frame_paths = _extract_frames(video_path, out_dir, max_frames)
        n_orig = int(self.data.N)
        n_new = len(new_frame_paths)
        if n_new == 0:
            print(f"append_video: no frames extracted from {video_path}; skipping")
            return {"first_new_idx": n_orig, "n_added": 0,
                    "epoch": max(self.data.frame_epoch) if self.data.frame_epoch else 0,
                    "skipped": 0, "n_candidates": 0,
                    "n_after_dedup": 0, "n_seeded": 0}
        joint_paths = list(self._frame_paths) + list(new_frame_paths)
        print(f"append_video: joint DA3 pass over {len(joint_paths)} frames "
              f"({n_orig} original + {n_new} new)")
        joint_data = _da3_inference_on_paths(joint_paths, self.device)

        # If joint DA3 picked a different (H, W) than the original we'd have
        # to resize every frame + splat space — out of scope. In practice
        # same camera + same DA3 settings → same (H, W).
        if (joint_data.H, joint_data.W) != (self.data.H, self.data.W):
            raise RuntimeError(
                f"joint DA3 produced ({joint_data.H}, {joint_data.W}); existing "
                f"data is ({self.data.H}, {self.data.W}). Pre-resize the appended "
                f"clip to match (same aspect ratio) and re-try."
            )

        # Phase 2.5 v2: align the joint frame to the *original* world frame
        # using per-frame SE(3) average — preserves both positions AND
        # orientations, unlike Kabsch-on-centers which silently leaves a
        # rotational "twist" that propagates to new frames.
        #
        # T_i = c2w_old[i] @ inv(c2w_joint[i]) is the per-frame transform
        # that takes a point in joint-world to old-world. With consistent
        # poses these are all the same; with DA3 noise they differ slightly.
        # `_average_se3` projects their mean onto SE(3) so we get one
        # canonical T_joint_to_old.
        old_c2w_orig = self.data.c2w[:n_orig].detach()
        joint_c2w_orig = joint_data.c2w[:n_orig].detach()
        if n_orig >= 1:
            inv_joint = torch.linalg.inv(joint_c2w_orig)
            T_per_frame = old_c2w_orig @ inv_joint           # (n_orig, 4, 4)
            T = _average_se3(T_per_frame)
        else:
            T = torch.eye(4, device=self.device, dtype=joint_data.c2w.dtype)
            print(f"append_video: no original frames to fit alignment; "
                  f"using identity (new splats may be globally offset).")

        # Residual diagnostics: where did the averaged T leave gaps? Position
        # residual = ||T @ c2w_joint - c2w_old||; angular residual via the
        # SO(3) chordal distance.
        aligned_orig = T.unsqueeze(0) @ joint_c2w_orig       # (n_orig, 4, 4)
        pos_err = (aligned_orig[:, :3, 3] - old_c2w_orig[:, :3, 3]).norm(dim=1)
        # angle(R) = arccos((trace - 1)/2). Compute R_err per frame.
        R_err = aligned_orig[:, :3, :3] @ old_c2w_orig[:, :3, :3].transpose(1, 2)
        tr = R_err.diagonal(dim1=1, dim2=2).sum(dim=1)        # (n_orig,)
        ang_err_rad = torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))
        ang_err_deg = ang_err_rad * 180.0 / math.pi
        print(f"append_video: SE(3) alignment residuals — "
              f"pos mean={pos_err.mean().item():.4f} max={pos_err.max().item():.4f} m "
              f"(scene_scale≈{self.init.scene_scale:.3f}); "
              f"angular mean={ang_err_deg.mean().item():.3f}° max={ang_err_deg.max().item():.3f}°")
        if ang_err_deg.max().item() > 5.0:
            print(f"append_video: WARNING max angular residual > 5° — joint DA3 "
                  f"may have substantially re-estimated the original poses. "
                  f"Existing splats may need ~hundreds of training steps to settle.")

        # Apply T ONLY to new frames' c2w. The original frames' c2w (and
        # depth, K, rgb) are NOT touched — they keep the exact poses the
        # existing splats were trained against. This is the second half of
        # the fix: even with imperfect alignment, the existing splats never
        # see a rotated supervision signal.
        new_c2w_aligned = T.unsqueeze(0) @ joint_data.c2w[n_orig:]

        cur_max_epoch = max(self.data.frame_epoch) if self.data.frame_epoch else 0
        new_epoch = cur_max_epoch + 1

        # Append each new frame to self.data + extend init.train_mask in
        # lock-step (same per-frame mask gates the original init used).
        for i in range(n_new):
            depth_i = joint_data.depth[n_orig + i]
            sky_i = (joint_data.sky[n_orig + i]
                     if joint_data.sky is not None else None)
            conf_i = (joint_data.conf[n_orig + i]
                      if joint_data.conf is not None else None)
            self.data.append_frame(
                rgb=joint_data.rgb[n_orig + i],
                depth=depth_i,
                K=joint_data.K[n_orig + i],
                c2w=new_c2w_aligned[i],
                conf=conf_i,
                sky=sky_i,
                epoch=new_epoch,
            )
            mask_i = _per_frame_train_mask(
                depth_i, sky_i, conf_i,
                self.init.conf_thresh, self.init.remove_sky,
            )
            self.init.append_train_mask(mask_i)

        # Extend cached paths + invalidate the prepare_and_init cache marker.
        self._frame_paths = list(self._frame_paths) + list(new_frame_paths)
        # `self.data` no longer matches the single `_last_video` signature
        # (it now spans both clips). Without this, Reset → Initialize on
        # the original video would skip preprocessing AND silently seed
        # splats from the appended frames still living in self.data.
        self._last_video = None

        # Recompute DA3 normals over the (extended) set in 2DGS mode.
        if self.mode == "2dgs" and self.da3_normal_weight > 0.0:
            with torch.no_grad():
                self._da3_normals = depth_to_normal(
                    self.data.depth[..., None], self.data.c2w, self.data.K,
                )

        first_new = n_orig
        added = n_new
        skipped = 0
        print(f"append_video: appended +{added} frames (epoch {new_epoch}); "
              f"seed_new_splats={seed_new_splats}, first_new_idx={first_new}")

        seed_info = {"n_candidates": 0, "n_after_dedup": 0, "n_seeded": 0}
        if seed_new_splats and added > 0:
            new_idxs = list(range(first_new, first_new + added))
            seed_info_full = self._seed_splats_for_new_frames(new_idxs)
            seed_info = {k: seed_info_full[k]
                         for k in ("n_candidates", "n_after_dedup", "n_seeded")}
        elif not seed_new_splats:
            print("append_video: seed_new_splats=False → splat count unchanged "
                  "(toggle 'seed new splats from added frames' to grow geometry)")

        info = {"first_new_idx": first_new, "n_added": added,
                "epoch": new_epoch, "skipped": skipped, **seed_info}
        print(f"append_video: +{added} frames (epoch {new_epoch}; skipped {skipped}) "
              f"→ data.N={self.data.N}; seeded={seed_info['n_seeded']} new splats")
        return info

    def append_supplied_frames(self, frames: Sequence[dict],
                               seed_new_splats: bool = False) -> dict:
        """Append externally-prepared frames. Each `frames[i]` is a dict:
            {"rgb":   (H, W, 3) float/uint8 numpy or torch,
             "K":     (3, 3) numpy or torch,
             "c2w":   (4, 4) numpy or torch,
             "depth": (H, W) numpy or torch — REQUIRED for Phase 1,
             "conf":  optional (H, W) float in [0,1] — defaults to ones,
             "sky":   optional (H, W) bool — defaults to zeros}.

        Mismatched resolution is auto-resized to (self.data.H, self.data.W);
        K is rescaled accordingly. All new frames share one epoch tag
        (max+1). For Phase 1 the caller must supply depth — running DA3 on
        a per-frame batch is a follow-up enhancement.

        Returns `{"first_new_idx", "n_added", "epoch", "skipped"}`.
        """
        if self.data is None or self.init is None:
            raise RuntimeError("call prepare_and_init before append_supplied_frames")
        cur_max_epoch = max(self.data.frame_epoch) if self.data.frame_epoch else 0
        new_epoch = cur_max_epoch + 1
        first_new = self.data.N
        added = 0
        skipped = 0
        H, W = self.data.H, self.data.W

        for j, fr in enumerate(frames):
            try:
                rgb_t = _to_tensor(fr["rgb"], self.device, torch.float32)
                if rgb_t.dtype != torch.float32:
                    rgb_t = rgb_t.float()
                if rgb_t.max() > 1.5:  # uint8-style payload
                    rgb_t = rgb_t / 255.0
                if rgb_t.ndim != 3 or rgb_t.shape[-1] != 3:
                    raise ValueError(f"rgb must be (H, W, 3); got {tuple(rgb_t.shape)}")
                if "depth" not in fr or fr["depth"] is None:
                    raise ValueError("depth is required (Phase 1)")
                depth_t = _to_tensor(fr["depth"], self.device, torch.float32)
                K_t = _to_tensor(fr["K"], self.device, torch.float32).reshape(3, 3)
                c2w_t = _to_tensor(fr["c2w"], self.device, torch.float32).reshape(4, 4)
                conf_t = (_to_tensor(fr["conf"], self.device, torch.float32)
                          if fr.get("conf") is not None else None)
                sky_t = (_to_tensor(fr["sky"], self.device, torch.bool)
                         if fr.get("sky") is not None else None)
            except Exception as e:
                print(f"append_supplied_frames: frame {j} skipped — {e}")
                skipped += 1
                continue

            if rgb_t.shape[:2] != (H, W):
                resized = _resize_frame_to(rgb_t, depth_t, K_t, conf_t, sky_t, H, W)
                if resized is None:
                    skipped += 1
                    continue
                rgb_t, depth_t, K_t, conf_t, sky_t = resized

            self.data.append_frame(
                rgb_t, depth_t, K_t, c2w_t,
                conf=conf_t, sky=sky_t, epoch=new_epoch,
            )
            mask_t = _per_frame_train_mask(
                depth_t, sky_t, conf_t,
                self.init.conf_thresh, self.init.remove_sky,
            )
            self.init.append_train_mask(mask_t)
            added += 1

        if self.mode == "2dgs" and self.da3_normal_weight > 0.0:
            with torch.no_grad():
                self._da3_normals = depth_to_normal(
                    self.data.depth[..., None], self.data.c2w, self.data.K,
                )

        print(f"append_supplied_frames: appended +{added} frames; "
              f"seed_new_splats={seed_new_splats}, first_new_idx={first_new}")
        seed_info = {"n_candidates": 0, "n_after_dedup": 0, "n_seeded": 0}
        if seed_new_splats and added > 0:
            new_idxs = list(range(first_new, first_new + added))
            seed_info_full = self._seed_splats_for_new_frames(new_idxs)
            seed_info = {k: seed_info_full[k]
                         for k in ("n_candidates", "n_after_dedup", "n_seeded")}
        elif not seed_new_splats:
            print("append_supplied_frames: seed_new_splats=False → splat count unchanged")

        info = {"first_new_idx": first_new, "n_added": added,
                "epoch": new_epoch, "skipped": skipped, **seed_info}
        print(f"append_supplied_frames: +{added} frames (epoch {new_epoch}; "
              f"skipped {skipped}) → data.N={self.data.N}; "
              f"seeded={seed_info['n_seeded']} new splats")
        return info

    # ---- Phase 2: seed new splats from already-appended frames -------- #

    def _seed_splats_for_new_frames(self, new_frame_idxs: Sequence[int]) -> dict:
        """RGBD-unproject the frames at `new_frame_idxs`, voxel-dedup against
        the existing splat distribution at `init.voxel`, voxel-downsample
        the survivors, and concat the resulting splats into the live
        `train.params` ParameterDict. Mirrors the inner logic of
        `_build_initial_gaussians` (lines 236–244 + 302–325) so the new
        seeds are produced by exactly the same recipe as the original ones.

        Per-key Adam optimizers + densify-strategy state are rebuilt fresh
        (same pattern as `prune_splats`). Adam moments for existing splats
        are reset — momentum recovers within a few hundred steps; the
        alternative (per-row state surgery) is brittle.

        Returns `{"n_candidates", "n_after_dedup", "n_seeded", "epoch"}`.
        Caller is responsible for pausing the training thread first
        (the GUI handler does this; the inpainter / CLI must too).
        """
        if self.train is None or self.data is None or self.init is None:
            raise RuntimeError("call prepare_and_init before _seed_splats_for_new_frames")
        idx_list = [int(i) for i in new_frame_idxs]
        if not idx_list:
            print("_seed_splats: called with empty index list — skipping")
            return {"n_candidates": 0, "n_after_dedup": 0, "n_seeded": 0,
                    "epoch": -1}
        print(f"_seed_splats: starting — {len(idx_list)} new frames (indices "
              f"[{idx_list[0]}..{idx_list[-1]}]); voxel={float(self.init.voxel):.5f} "
              f"conf_thresh={self.init.conf_thresh} remove_sky={bool(self.init.remove_sky)}")

        d = self.data
        device = self.device
        voxel = float(self.init.voxel)
        remove_sky = bool(self.init.remove_sky)
        conf_thresh = self.init.conf_thresh
        H, W = int(d.H), int(d.W)

        sel = torch.as_tensor(idx_list, dtype=torch.long, device=device)
        # The epoch label for newly-seeded splats: the max epoch across the
        # selected frames. In the common case (a single append session) all
        # selected frames share one epoch, so this is just that epoch.
        epochs_for_new = [d.frame_epoch[i] for i in idx_list]
        new_epoch = max(epochs_for_new) if epochs_for_new else 0

        # ---- Unproject the selected frames (same as init lines 236-244) -- #
        ii, jj = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device), indexing="ij")
        uv1 = torch.stack([jj, ii, torch.ones_like(ii)], -1).float()
        K_sel = d.K[sel]
        c2w_sel = d.c2w[sel]
        depth_sel = d.depth[sel]
        rgb_sel = d.rgb[sel]
        Kinv = torch.linalg.inv(K_sel)
        cam_pts = torch.einsum("nij,hwj->nhwi", Kinv, uv1) * depth_sel[..., None]
        R = c2w_sel[:, :3, :3]
        tt = c2w_sel[:, :3, 3]
        world = torch.einsum("nij,nhwj->nhwi", R, cam_pts) + tt[:, None, None, :]

        # Validity gate matches `_per_frame_train_mask`.
        valid = depth_sel > 0
        if d.sky is not None and remove_sky:
            valid &= ~d.sky[sel]
        if d.conf is not None and conf_thresh is not None:
            valid &= d.conf[sel] > conf_thresh

        if not bool(valid.any()):
            n_depth = int((depth_sel > 0).sum().item())
            n_sky_ok = int((~d.sky[sel]).sum().item()) if (d.sky is not None and remove_sky) else -1
            n_conf_ok = int((d.conf[sel] > conf_thresh).sum().item()) \
                if (d.conf is not None and conf_thresh is not None) else -1
            print(f"_seed_splats: epoch={new_epoch} ALL CANDIDATES INVALID — "
                  f"depth>0={n_depth}, ~sky={n_sky_ok}, conf>{conf_thresh}={n_conf_ok}; "
                  f"check whether the new clip's depth/sky/conf gates match the originals")
            return {"n_candidates": 0, "n_after_dedup": 0, "n_seeded": 0,
                    "epoch": new_epoch}

        pts_full = world[valid]
        cols_full = rgb_sel[valid]
        z_full = depth_sel[valid]
        fx_nhw = K_sel[:, 0, 0][:, None, None].expand(int(sel.numel()), H, W)
        fx_full = fx_nhw[valid]
        M_cand = int(pts_full.shape[0])

        # Memory-safety cap on the *candidate pool* (raw unprojected pixels)
        # before voxel dedup/downsample. Mirrors the `max_points` cap that
        # `_build_initial_gaussians` applies at lines 269-272 — it's a guard
        # against tens of millions of candidates eating GPU memory, NOT a
        # cap on total splat count. Densify + prune are the right knobs for
        # that. Use `prune_splats` first if you really need to cap total M.
        existing_count = int(self.train.params["means"].shape[0])
        if M_cand > int(self.max_points):
            order = torch.randperm(M_cand, device=device)[:int(self.max_points)]
            pts_full, cols_full = pts_full[order], cols_full[order]
            z_full, fx_full = z_full[order], fx_full[order]
            M_cand = int(self.max_points)
            print(f"_seed_splats: candidate pool subsampled to max_points={self.max_points} "
                  f"(unprojection memory safety; final seeded count is still set by voxel dedup)")

        # ---- Voxel-dedup against existing splats ------------------------- #
        # Use a COARSER voxel for the dedup check than for the downsample
        # below. Joint DA3 pose noise is ~few-voxel scale at init.voxel ≈
        # 0.5% scene scale; exact voxel-key matches miss near-duplicates.
        # `_seed_dedup_multiplier` widens the match radius (default 2.0 ⇒
        # candidate occupied if any existing splat is within ~2 init
        # voxels). Downsample still uses `voxel` so spacing of *kept* new
        # splats matches the init grid.
        existing_means = self.train.params["means"].detach()
        dedup_voxel = float(self.init.voxel) * float(self._seed_dedup_multiplier)
        BIT = 21
        OFF = 1 << (BIT - 1)
        MASK = (1 << BIT) - 1

        def _scalar_keys(pts: torch.Tensor) -> torch.Tensor:
            k = torch.floor(pts / dedup_voxel).long()
            k0 = (k[:, 0] + OFF).clamp(0, MASK)
            k1 = (k[:, 1] + OFF).clamp(0, MASK)
            k2 = (k[:, 2] + OFF).clamp(0, MASK)
            return (k0 << (2 * BIT)) | (k1 << BIT) | k2

        cand_scalar = _scalar_keys(pts_full)
        exist_scalar = _scalar_keys(existing_means)
        occupied = torch.isin(cand_scalar, exist_scalar)
        keep = ~occupied
        M_after_dedup = int(keep.sum().item())
        print(f"_seed_splats: dedup at {dedup_voxel:.4f}m "
              f"({self._seed_dedup_multiplier:.1f}× init voxel {voxel:.4f}m) — "
              f"{int(occupied.sum().item())} candidates dropped as duplicates")
        if M_after_dedup == 0:
            print(f"_seed_splats: epoch={new_epoch} cand={M_cand} after_dedup=0 — "
                  f"every candidate voxel was already occupied by an existing splat. "
                  f"Either the two clips overlap spatially (expected) or the dedup "
                  f"hash misfired (voxel={voxel} too coarse?).")
            return {"n_candidates": M_cand, "n_after_dedup": 0, "n_seeded": 0,
                    "epoch": new_epoch}

        pts_new = pts_full[keep]
        cols_new = cols_full[keep]
        z_new = z_full[keep]
        fx_new = fx_full[keep]

        # ---- Voxel-downsample the survivors (mirrors init lines 302-325) -- #
        keys_new = torch.floor(pts_new / voxel).long()
        uniq_keys, inv = torch.unique(keys_new, dim=0, return_inverse=True)
        G = int(uniq_keys.shape[0])

        means_g = _scatter_mean(pts_new, inv, G)
        cols_g = _scatter_mean(cols_new, inv, G)
        z_g = _scatter_mean(z_new, inv, G)
        fx_g = _scatter_mean(fx_new, inv, G)

        inflate = max(1.0, (M_after_dedup / max(G, 1)) ** 0.5)
        tex_g = (z_g / fx_g * inflate).clamp_min(voxel * 0.5)

        C0 = 0.28209479177387814
        means_new_t = means_g.clone()
        f_dc_new = (cols_g - 0.5) / C0
        log_s_new = torch.log(tex_g[:, None].expand(G, 3).contiguous())
        logit_o_new = torch.full((G,), 2.1972, device=device)
        quats_new_t = torch.zeros((G, 4), device=device)
        quats_new_t[:, 0] = 1.0
        sh0_new = f_dc_new[:, None, :]
        K_rest = int(self.train.params["shN"].shape[1])
        shN_new = torch.zeros(
            (G, K_rest, 3), device=device,
            dtype=self.train.params["shN"].dtype,
        )

        # ---- Concat into the live ParameterDict + rebuild opts ----------- #
        ts = self.train
        p = ts.params
        new_params = torch.nn.ParameterDict({
            "means":     torch.nn.Parameter(torch.cat([p["means"].detach(),     means_new_t], dim=0)),
            "scales":    torch.nn.Parameter(torch.cat([p["scales"].detach(),    log_s_new],   dim=0)),
            "quats":     torch.nn.Parameter(torch.cat([p["quats"].detach(),     quats_new_t], dim=0)),
            "opacities": torch.nn.Parameter(torch.cat([p["opacities"].detach(), logit_o_new], dim=0)),
            "sh0":       torch.nn.Parameter(torch.cat([p["sh0"].detach(),       sh0_new],     dim=0)),
            "shN":       torch.nn.Parameter(torch.cat([p["shN"].detach(),       shN_new],     dim=0)),
        }).to(device)
        means_lr = 1.6e-4 * self.init.scene_scale if self.use_densify else 1.6e-4
        lr_table = {
            "means":     means_lr,
            "scales":    5e-3,
            "quats":     1e-3,
            "opacities": 5e-2,
            "sh0":       2.5e-3,
            "shN":       2.5e-3 / 20.0,
        }
        new_opts = {
            k: torch.optim.Adam([{"params": [new_params[k]], "lr": lr}])
            for k, lr in lr_table.items()
        }
        if ts.strategy is not None:
            ts.strategy.check_sanity(new_params, new_opts)
            ts.strategy_state = ts.strategy.initialize_state(
                scene_scale=self.init.scene_scale,
            )
        ts.params = new_params
        ts.opts = new_opts

        # ---- Extend the per-splat epoch tag (Phase 5 foundation) --------- #
        if self._splat_epoch is None or int(self._splat_epoch.numel()) != existing_count:
            # Backfill with zeros so old splats are tagged epoch 0.
            self._splat_epoch = torch.zeros(existing_count, dtype=torch.int32, device=device)
        new_tags = torch.full((G,), int(new_epoch), dtype=torch.int32, device=device)
        self._splat_epoch = torch.cat([self._splat_epoch, new_tags], dim=0)

        # ---- Extend the per-splat freeze mask (Phase 4) ----------------- #
        # New splats default to trainable (True). If the user wants the new
        # splats also frozen relative to the latest cameras, they can click
        # Recompute after seeding to rebuild the mask from visibility.
        if self._frozen_mask is not None and int(self._frozen_mask.numel()) == existing_count:
            new_trainable = torch.ones((G,), dtype=torch.bool, device=device)
            self._frozen_mask = torch.cat([self._frozen_mask, new_trainable], dim=0)

        self._publish_to_scene()
        info = {"n_candidates": M_cand, "n_after_dedup": M_after_dedup,
                "n_seeded": G, "epoch": new_epoch}
        print(f"_seed_splats: epoch={new_epoch} cand={M_cand} after_dedup={M_after_dedup} "
              f"seeded={G} → M={int(new_params['means'].shape[0])}")
        return info

    # ---- Phase 4: spatial freezing of irrelevant splats --------------- #

    FREEZE_MODES = ("off", "new_frustums")

    def set_freeze_mode(self, mode: str) -> dict:
        """GUI hook to flip the freeze policy at runtime.

        `mode`:
          - "off" (default) — current behaviour, train every splat.
          - "new_frustums" — after `recompute_freeze_mask`, splats invisible
            to every camera in the latest epoch are frozen (gradient zeroed
            before optim.step). Densify is auto-disabled while a freeze
            mask is active so the splat-count can't desync the mask.

        Flipping to "off" clears any existing mask. Flipping the other way
        leaves the mask alone — you still need to click Recompute for it
        to take effect (lets you stage the change before applying).
        Returns a small status dict.
        """
        if mode not in self.FREEZE_MODES:
            raise ValueError(f"mode must be one of {self.FREEZE_MODES}, got {mode!r}")
        self._freeze_mode = mode
        if mode == "off":
            self._frozen_mask = None
        return {
            "mode": self._freeze_mode,
            "mask_active": self._frozen_mask is not None,
            "n_trainable": int(self._frozen_mask.sum().item()) if self._frozen_mask is not None else None,
            "n_total": int(self.train.params["means"].shape[0]) if self.train is not None else 0,
        }

    def recompute_freeze_mask(self) -> dict:
        """Project all `train.params["means"]` into every latest-epoch
        camera and mark splats visible to at least one as trainable. Other
        splats are frozen. Cheap (~O(N_cam × M) batched matmuls); takes a
        fraction of a second for a 3M-splat scene + 32 cameras.

        "Visible" is the rectangular-frustum test: positive z in camera
        space AND projected pixel inside (0, 0) – (W, H). NO occlusion
        test — splats hidden behind geometry are treated as visible
        (we *want* to update those, since they may be the wrong geometry
        that's hiding what the new cameras see). Phase 5 will offer an
        occlusion-aware variant if needed.

        No-op if `_freeze_mode == "off"`. Returns counts.
        """
        if self.train is None or self.data is None:
            return {"mode": self._freeze_mode, "n_total": 0,
                    "n_trainable": 0, "n_frozen": 0, "n_cameras": 0}
        M = int(self.train.params["means"].shape[0])
        if self._freeze_mode == "off":
            self._frozen_mask = None
            print("recompute_freeze_mask: mode=off → mask cleared (train all splats)")
            return {"mode": "off", "n_total": M, "n_trainable": M,
                    "n_frozen": 0, "n_cameras": 0, "latest_epoch": None}

        d = self.data
        epochs = d.frame_epoch or []
        if not epochs:
            self._frozen_mask = None
            return {"mode": self._freeze_mode, "n_total": M, "n_trainable": M,
                    "n_frozen": 0, "n_cameras": 0, "latest_epoch": None}
        latest = max(int(e) for e in epochs)
        cam_idxs = [i for i, e in enumerate(epochs) if int(e) == latest]
        if not cam_idxs:
            self._frozen_mask = None
            return {"mode": self._freeze_mode, "n_total": M, "n_trainable": M,
                    "n_frozen": 0, "n_cameras": 0, "latest_epoch": latest}

        sel = torch.as_tensor(cam_idxs, dtype=torch.long, device=self.device)
        w2c_cams = d.w2c[sel]                                 # (N_cam, 4, 4)
        K_cams = d.K[sel]                                     # (N_cam, 3, 3)
        H, W = int(d.H), int(d.W)

        p = self.train.params["means"].detach()               # (M, 3)
        p_h = torch.cat(
            [p, torch.ones((M, 1), device=p.device, dtype=p.dtype)],
            dim=1,
        )                                                     # (M, 4)
        # Transform: (N_cam, 4, 4) · (M, 4)^T → (N_cam, M, 4)
        p_cam = torch.einsum("nij,mj->nmi", w2c_cams, p_h)[..., :3]
        z = p_cam[..., 2]
        in_front = z > 0
        z_safe = z.clamp_min(1e-6)
        fx = K_cams[:, 0, 0:1]                                 # (N_cam, 1)
        fy = K_cams[:, 1, 1:2]
        cx = K_cams[:, 0, 2:3]
        cy = K_cams[:, 1, 2:3]
        pix_x = fx * (p_cam[..., 0] / z_safe) + cx             # (N_cam, M)
        pix_y = fy * (p_cam[..., 1] / z_safe) + cy
        in_bounds = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H)
        visible = in_front & in_bounds                          # (N_cam, M)
        trainable = visible.any(dim=0)                          # (M,) bool

        self._frozen_mask = trainable
        n_trainable = int(trainable.sum().item())
        n_frozen = M - n_trainable
        print(f"recompute_freeze_mask: mode={self._freeze_mode} latest_epoch={latest} "
              f"cameras={len(cam_idxs)} M={M} trainable={n_trainable} frozen={n_frozen} "
              f"({100.0 * n_trainable / max(M, 1):.1f}% trainable)")
        return {
            "mode": self._freeze_mode,
            "n_total": M,
            "n_trainable": n_trainable,
            "n_frozen": n_frozen,
            "n_cameras": len(cam_idxs),
            "latest_epoch": int(latest),
        }

    # ---- Phase 5: visual debug helpers --------------------------------- #

    def compute_voxel_overlap_layer(self, voxel_mult: float = 1.0) -> dict:
        """Phase 5.2: classify every splat into its (voxel × epoch) bin
        and return a colored point cloud where each point is one
        occupied voxel center. Categories:
          - old-only (red): voxel contains only epoch-0 splats
          - new-only (green): voxel contains only epoch>0 splats
          - shared (yellow): both old and new splats in the same voxel
        Renders quickly into a viser point cloud and lets the user spot
        where two clips overlap vs disjoint regions.

        Uses the same bit-packed scalar voxel hash as `_seed_splats_…`.
        Returns numpy arrays so the viewer can wrap them in a
        `PointCloudLayer` without taking a torch dep on the GUI side.
        """
        empty = {
            "points": np.zeros((0, 3), dtype=np.float32),
            "colors": np.zeros((0, 3), dtype=np.uint8),
            "n_old_only": 0, "n_new_only": 0, "n_shared": 0,
            "n_voxels": 0, "voxel_size": 0.0,
        }
        if self.train is None or self.init is None:
            return empty
        means = self.train.params["means"].detach()
        M = int(means.shape[0])
        if M == 0:
            return empty
        voxel = float(self.init.voxel) * max(0.1, float(voxel_mult))
        if self._splat_epoch is None or int(self._splat_epoch.numel()) != M:
            epochs = torch.zeros(M, dtype=torch.int32, device=means.device)
        else:
            epochs = self._splat_epoch

        # Voxel hash — 21 bits/axis bit-packed into one int64.
        BIT = 21
        OFF = 1 << (BIT - 1)
        MASK = (1 << BIT) - 1
        keys = torch.floor(means / voxel).long()
        k0 = (keys[:, 0] + OFF).clamp(0, MASK)
        k1 = (keys[:, 1] + OFF).clamp(0, MASK)
        k2 = (keys[:, 2] + OFF).clamp(0, MASK)
        scalar = (k0 << (2 * BIT)) | (k1 << BIT) | k2          # (M,)

        is_old = epochs == 0
        is_new = ~is_old
        old_voxels = torch.unique(scalar[is_old]) if int(is_old.sum().item()) > 0 \
            else torch.empty(0, dtype=scalar.dtype, device=scalar.device)
        new_voxels = torch.unique(scalar[is_new]) if int(is_new.sum().item()) > 0 \
            else torch.empty(0, dtype=scalar.dtype, device=scalar.device)
        if old_voxels.numel() == 0 and new_voxels.numel() == 0:
            return empty
        all_voxels = torch.unique(torch.cat([old_voxels, new_voxels]))
        in_old = torch.isin(all_voxels, old_voxels)
        in_new = torch.isin(all_voxels, new_voxels)
        old_only = in_old & ~in_new
        new_only = ~in_old & in_new
        shared   = in_old & in_new

        # Per-voxel centre: mean of contained splat positions. Use
        # searchsorted (all_voxels is unique+sorted) to map each splat to
        # its bin index, then `_scatter_mean` for the per-bin average.
        inv = torch.searchsorted(all_voxels, scalar)
        V = int(all_voxels.numel())
        pos = _scatter_mean(means, inv, V)                     # (V, 3)

        red    = torch.tensor([220,  80,  80], dtype=torch.uint8, device=means.device)
        green  = torch.tensor([ 80, 220,  80], dtype=torch.uint8, device=means.device)
        yellow = torch.tensor([240, 220,  80], dtype=torch.uint8, device=means.device)
        cols = torch.zeros((V, 3), dtype=torch.uint8, device=means.device)
        cols[old_only] = red
        cols[new_only] = green
        cols[shared]   = yellow

        info = {
            "points": pos.cpu().numpy().astype(np.float32),
            "colors": cols.cpu().numpy(),
            "n_old_only": int(old_only.sum().item()),
            "n_new_only": int(new_only.sum().item()),
            "n_shared":   int(shared.sum().item()),
            "n_voxels": V,
            "voxel_size": voxel,
        }
        print(f"compute_voxel_overlap_layer: voxel={voxel:.4f}m → "
              f"{V:,} voxels (old_only={info['n_old_only']:,}, "
              f"new_only={info['n_new_only']:,}, shared={info['n_shared']:,})")
        return info

    def compute_coverage_layer(self, camera_scope: str = "all",
                               max_display_points: int = 200_000) -> dict:
        """Phase 5.3: count how many cameras see each splat (same cheap
        projection test as `recompute_freeze_mask` — z>0 + inside image
        rectangle, no occlusion) and color by count via a 5-stop turbo
        approximation. Hot = many cameras see it; cold = few or none.
        Splats with 0 cameras are the obvious "unsupervised" candidates
        for prune.

        `camera_scope`: "all" = every camera, "latest" = only the latest-
        epoch cameras (matches what Phase 4 freezes).
        `max_display_points`: cap for the returned point cloud to keep
        the websocket happy on large scenes. Uniform random subsample.

        Returns numpy arrays + counts for the viewer to wrap as a layer.
        """
        empty = {
            "points": np.zeros((0, 3), dtype=np.float32),
            "colors": np.zeros((0, 3), dtype=np.uint8),
            "n_total": 0, "max_count": 0, "n_unseen": 0,
            "n_cameras": 0, "scope": camera_scope,
        }
        if self.train is None or self.data is None:
            return empty
        d = self.data
        epochs = d.frame_epoch or []
        if camera_scope == "latest" and epochs:
            latest = max(int(e) for e in epochs)
            cam_idxs = [i for i, e in enumerate(epochs) if int(e) == latest]
        else:
            cam_idxs = list(range(int(d.N)))
        if not cam_idxs:
            return empty

        sel = torch.as_tensor(cam_idxs, dtype=torch.long, device=self.device)
        H, W = int(d.H), int(d.W)
        p = self.train.params["means"].detach()
        M = int(p.shape[0])
        if M == 0:
            return empty

        p_h = torch.cat(
            [p, torch.ones((M, 1), device=p.device, dtype=p.dtype)],
            dim=1,
        )
        # Batch over cameras so peak memory stays sane for big M.
        CHUNK = 8
        counts = torch.zeros(M, dtype=torch.int32, device=p.device)
        for cs in range(0, len(cam_idxs), CHUNK):
            ce = min(cs + CHUNK, len(cam_idxs))
            chunk_sel = sel[cs:ce]
            w2c_chunk = d.w2c[chunk_sel]
            K_chunk = d.K[chunk_sel]
            p_cam = torch.einsum("nij,mj->nmi", w2c_chunk, p_h)[..., :3]
            z = p_cam[..., 2]
            in_front = z > 0
            z_safe = z.clamp_min(1e-6)
            fx = K_chunk[:, 0, 0:1]
            fy = K_chunk[:, 1, 1:2]
            cx = K_chunk[:, 0, 2:3]
            cy = K_chunk[:, 1, 2:3]
            pix_x = fx * (p_cam[..., 0] / z_safe) + cx
            pix_y = fy * (p_cam[..., 1] / z_safe) + cy
            in_bounds = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H)
            counts += (in_front & in_bounds).sum(dim=0).to(torch.int32)

        max_count = int(counts.max().item()) if counts.numel() > 0 else 0
        n_unseen = int((counts == 0).sum().item())

        # Turbo-ish colormap: 5-stop linear gradient cool→hot.
        stops = torch.tensor([
            [ 48,  18,  59],   # deep blue
            [ 33, 156, 203],   # cyan
            [121, 217,  99],   # green
            [253, 192,  19],   # yellow
            [165,  13,   7],   # red
        ], device=p.device, dtype=torch.float32)
        n_stops = stops.shape[0]
        denom = max(max_count, 1)
        t01 = (counts.float() / float(denom)).clamp(0.0, 1.0)
        scaled = t01 * float(n_stops - 1)
        idx0 = scaled.floor().long().clamp(0, n_stops - 1)
        idx1 = (idx0 + 1).clamp(0, n_stops - 1)
        frac = (scaled - scaled.floor()).unsqueeze(-1)
        cols_f = stops[idx0] * (1.0 - frac) + stops[idx1] * frac
        cols = cols_f.clamp(0, 255).to(torch.uint8)

        # Subsample for the websocket — viser bogs down past ~200k points.
        pts_np = p.cpu().numpy().astype(np.float32)
        cols_np = cols.cpu().numpy()
        if max_display_points > 0 and M > max_display_points:
            rng = np.random.default_rng(0)
            keep = rng.choice(M, size=max_display_points, replace=False)
            keep.sort()
            pts_np = pts_np[keep]
            cols_np = cols_np[keep]

        info = {
            "points": pts_np,
            "colors": cols_np,
            "n_total": M,
            "max_count": max_count,
            "n_unseen": n_unseen,
            "n_cameras": len(cam_idxs),
            "scope": camera_scope,
        }
        print(f"compute_coverage_layer: scope={camera_scope} cameras={len(cam_idxs)} "
              f"M={M} max_count={max_count} unseen={n_unseen}")
        return info

    # ---- Phase 3: epoch-aware frame sampling -------------------------- #

    def set_sampling(self, mode: str = "uniform",
                     new_frame_weight: float = 1.0,
                     horizon: int = 1000) -> dict:
        """GUI hook to change the per-step frame sampler at runtime.

        `mode`: "uniform" | "stratified" | "scheduled".
        `new_frame_weight`: the latest epoch's weight relative to every
        other epoch (1.0 = uniform, 2.0 = sample latest epoch 2× as often
        per frame as each other epoch, 0.5 = half as often).
        `horizon`: scheduled mode only — number of steps over which the
        sampler decays from "latest epoch only" to the stratified target.

        Returns a small status dict mirroring the new state. Safe to call
        at any time; the next `step()` picks it up.
        """
        if mode not in ("uniform", "stratified", "scheduled"):
            raise ValueError(f"mode must be uniform/stratified/scheduled, got {mode!r}")
        self._sampling_mode = mode
        self._new_frame_weight = max(0.0, float(new_frame_weight))
        self._sampling_horizon = max(1, int(horizon))
        return {
            "mode": self._sampling_mode,
            "new_frame_weight": self._new_frame_weight,
            "horizon": self._sampling_horizon,
        }

    def epoch_frame_counts(self) -> dict[int, int]:
        """Per-epoch frame counts from `data.frame_epoch`. Used by the GUI
        status panel + by `_sample_frame_idx` to decide which epoch labels
        actually have frames in the current set."""
        if self.data is None or not self.data.frame_epoch:
            return {}
        out: dict[int, int] = {}
        for e in self.data.frame_epoch:
            out[int(e)] = out.get(int(e), 0) + 1
        return out

    def _sample_frame_idx(self) -> int:
        """Per-step frame sampler. `uniform` keeps the original behaviour
        (single `torch.randint`). `stratified` picks an epoch first using
        per-epoch weights (latest gets `new_frame_weight`, others get 1.0),
        then a frame within that epoch. `scheduled` linearly interpolates
        from "sample only the newest epoch" at step 0 to the stratified
        target at step ≥ `_sampling_horizon`.

        Fast-path: any time only one epoch is present, falls back to a
        single uniform draw — saves the python loop for the common case
        (no append yet, or single-clip session)."""
        d = self.data
        if d is None or d.N == 0:
            return 0
        # Fast path #1: uniform mode or no incremental weighting requested.
        if self._sampling_mode == "uniform" or self._new_frame_weight == 1.0:
            return int(torch.randint(0, d.N, (1,)).item())
        epochs = d.frame_epoch or []
        unique = sorted(set(int(e) for e in epochs))
        # Fast path #2: only one epoch — no stratification possible.
        if len(unique) <= 1:
            return int(torch.randint(0, d.N, (1,)).item())
        latest = unique[-1]
        # Stratified target weights: latest gets new_frame_weight, others 1.0.
        target_w = {e: (self._new_frame_weight if e == latest else 1.0)
                    for e in unique}
        if self._sampling_mode == "scheduled":
            h = max(1, self._sampling_horizon)
            progress = min(1.0, float(self._step_count) / h)
            # Initial: all weight on the latest epoch (only-new sampling).
            init_w = {e: (1.0 if e == latest else 0.0) for e in unique}
            cur_w = {e: init_w[e] * (1.0 - progress) + target_w[e] * progress
                     for e in unique}
        else:  # stratified
            cur_w = target_w
        weights = torch.tensor([cur_w[e] for e in unique], dtype=torch.float32)
        total = float(weights.sum().item())
        if total <= 0.0:
            return int(torch.randint(0, d.N, (1,)).item())
        epoch_choice = int(torch.multinomial(weights, 1).item())
        chosen_epoch = unique[epoch_choice]
        # Pick a frame uniformly within the chosen epoch.
        in_epoch = [i for i, e in enumerate(epochs) if int(e) == chosen_epoch]
        return in_epoch[int(torch.randint(0, len(in_epoch), (1,)).item())]

    def refine_gaussians(self, steps: int) -> None:
        """Headless CLI training loop. Requires `prepare_and_init` first."""
        if self.train is None or self.data is None:
            raise RuntimeError("call prepare_and_init() before refine_gaussians()")

        tag, desc = "splats", "training"
        pbar = tqdm(range(steps), desc=desc, unit="step")
        for s in pbar:
            loss = self.step()
            if s % 100 == 0:
                postfix = {
                    "loss": f"{loss:.4f}",
                    "M":    int(self.train.params["means"].shape[0]),
                    "sh":   self._current_sh_degree(),
                }
                if self._last_loss_components:
                    postfix["breakdown"] = format_loss_components(
                        self._last_loss_components
                    )
                pbar.set_postfix(**postfix)

        torch.set_grad_enabled(False)
        self.save_current(f"{tag}.ply")
        print(f"Saved {tag}.ply")
        # Snapshot for render_and_show — mirrors save_current's sh-cat logic.
        p = self.train.params
        sh_all = (p["sh0"] if p["shN"].shape[1] == 0
                  else torch.cat([p["sh0"], p["shN"]], dim=1))
        render_and_show(self.data, p["means"], p["quats"],
                        p["scales"], p["opacities"], sh_all, tag=tag,
                        mode=self.mode)
        self.render_diagnostics(tag)

    def reset(self) -> None:
        """Tear down trainable state and clear the scene. Caller is
        responsible for pausing any running BackgroundTrainingThread before
        calling this so step() can't fire mid-tear-down. Preserves
        `self.data` so re-init on the same video skips preprocessing
        (matches prior behavior)."""
        self._initialized = False
        self._step_count = 0
        self.train = None
        self.init = None
        self._da3_normals = None
        self._splat_epoch = None
        self._frozen_mask = None
        if self.scene is not None:
            with self.scene.write() as s:
                s.means = s.quats = s.scales = None
                s.opacities = s.sh = None
                s.num_splats = 0
                s.step = 0
                s.loss_history = []
                s.splat_version += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prune_splats(self,
                     opa_min: float = 0.05,
                     scale_max_frac: float = 0.10,
                     aniso_max: float = 10.0,
                     use_knn: bool = True,
                     knn_k: int = 20,
                     knn_std: float = 2.0) -> dict:
        """Filter floater / spiky / oversized splats from the trainable
        params in-place, then rebuild the per-key Adam optimizers and
        (if densify is on) re-initialize strategy state. Returns a dict
        of per-filter counts removed.

        Mirrors notebook cell 17 (`splats_2dgs_clean.ply`):
        - opacity < `opa_min`
        - max axis-scale > `scale_max_frac * scene_scale`
        - anisotropy (disk-plane scale ratio) > `aniso_max`
        - KNN floaters: mean-distance > mean + `knn_std`*std (scipy KDTree)

        Caller is responsible for pausing the training thread before calling
        — replacing `self.train.params` mid-step would race with the
        forward pass.
        """
        if self.train is None or self.init is None:
            raise RuntimeError("call prepare_and_init() before prune_splats()")
        t = self.train
        p = t.params
        device = p["means"].device
        M = int(p["means"].shape[0])

        with torch.no_grad():
            scales_exp = torch.exp(p["scales"].detach())     # (M, 3)
            opa        = torch.sigmoid(p["opacities"].detach())
            # 2DGS: disk plane is scales[:, :2]. 3DGS: use the two largest
            # axes as the "disk plane" for an analogous needle test.
            if self.mode == "2dgs":
                s_pair = scales_exp[:, :2]
            else:
                s_pair = scales_exp.topk(2, dim=1).values
            s_pair_sorted, _ = torch.sort(s_pair, dim=1)
            aniso     = s_pair_sorted[:, 1] / s_pair_sorted[:, 0].clamp_min(1e-8)
            max_scale = scales_exp.max(dim=1).values
            scale_cap = scale_max_frac * self.init.scene_scale

            counts: dict = {"started": M}
            keep = torch.ones(M, dtype=torch.bool, device=device)
            for name, m in (
                ("opacity", opa       >= opa_min),
                ("scale",   max_scale <= scale_cap),
                ("aniso",   aniso     <= aniso_max),
            ):
                counts[name] = int((~m).sum().item())
                keep &= m

            if use_knn and M > knn_k + 1:
                try:
                    from scipy.spatial import cKDTree
                    means_np = p["means"].detach().cpu().numpy()
                    tree = cKDTree(means_np)
                    dists, _ = tree.query(means_np, k=knn_k + 1)
                    avg_d = dists[:, 1:].mean(axis=1)
                    thresh = float(avg_d.mean() + knn_std * avg_d.std())
                    knn_mask = torch.from_numpy(avg_d <= thresh).to(device)
                    counts["knn"] = int((~knn_mask).sum().item())
                    keep &= knn_mask
                except ImportError:
                    counts["knn"] = -1   # scipy missing

        kept = int(keep.sum().item())
        counts["kept"] = kept
        counts["removed_total"] = M - kept
        if kept == M:
            return counts

        # Rebuild params + optimizers with filtered tensors. Adam state for
        # kept splats is dropped — momentum recovers within ~hundreds of
        # steps and the alternative (per-param state surgery) is brittle.
        new_params = torch.nn.ParameterDict({
            k: torch.nn.Parameter(p[k].detach()[keep].clone())
            for k in ("means", "scales", "quats", "opacities", "sh0", "shN")
        }).to(device)
        means_lr = 1.6e-4 * self.init.scene_scale if self.use_densify else 1.6e-4
        lr_table = {
            "means":     means_lr,
            "scales":    5e-3,
            "quats":     1e-3,
            "opacities": 5e-2,
            "sh0":       2.5e-3,
            "shN":       2.5e-3 / 20.0,
        }
        new_opts = {
            k: torch.optim.Adam([{"params": [new_params[k]], "lr": lr}])
            for k, lr in lr_table.items()
        }
        if t.strategy is not None:
            t.strategy.check_sanity(new_params, new_opts)
            t.strategy_state = t.strategy.initialize_state(
                scene_scale=self.init.scene_scale,
            )
        t.params = new_params
        t.opts   = new_opts
        # Filter the per-splat epoch tag through the same `keep` mask so it
        # stays in lock-step with the trainable params (Phase 2 bookkeeping).
        if self._splat_epoch is not None and self._splat_epoch.numel() == M:
            self._splat_epoch = self._splat_epoch[keep].contiguous()
        if self._frozen_mask is not None and self._frozen_mask.numel() == M:
            self._frozen_mask = self._frozen_mask[keep].contiguous()
        self._publish_to_scene()
        return counts

    def step(self) -> float:
        if (not self._initialized or self.train is None
                or self.data is None or self.init is None):
            time.sleep(0.05)   # avoid busy-spin if thread is resumed early
            return 0.0
        t, d, i = self.train, self.data, self.init
        p = t.params
        idx = self._sample_frame_idx()

        # SH degree ramps 0→sh_max_deg. `cat` with a (M,0,3) shN is a no-op,
        # so the same path covers sh_max_deg=0.
        cur_sh = self._current_sh_degree()
        sh_all = torch.cat([p["sh0"], p["shN"]], dim=1)

        normals = surf_normals = distort = None
        depth_pred = None
        if self.mode == "2dgs":
            colors, alphas, normals, surf_normals, distort, _, info = rasterization_2dgs(
                p["means"], p["quats"],
                torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                sh_all, d.w2c[idx:idx+1], d.K[idx:idx+1], d.W, d.H,
                sh_degree=cur_sh, packed=False,
                render_mode="RGB+ED", distloss=True)
            pred = colors[0, ..., :3]                       # (H,W,3) float
            depth_pred = colors[0, ..., 3]                  # (H,W) float
        else:
            out, alphas, info = rasterization(
                p["means"], p["quats"],
                torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                sh_all, d.w2c[idx:idx+1], d.K[idx:idx+1], d.W, d.H,
                sh_degree=cur_sh, packed=False)
            pred = out[0]                                   # (H,W,3) float

        # Phase 4: skip densify entirely while a freeze mask is in effect
        # so the splat count can't change underneath the mask. `freeze_active`
        # is checked at *both* pre-backward and post-backward to keep the
        # strategy's internal state consistent (skip both halves of the pair
        # — the strategy expects matched calls).
        freeze_active = (self._frozen_mask is not None
                         and self._frozen_mask.numel() == p["means"].shape[0])
        if t.strategy is not None and not freeze_active:
            t.strategy.step_pre_backward(p, t.opts, t.strategy_state,
                                         self._step_count, info)

        # Masked L1. The mask excludes sky / low-confidence-depth pixels.
        diff = (pred - d.rgb[idx]).abs()
        mask = i.train_mask[idx].unsqueeze(-1)              # (H,W,1) bool
        n_kept = int(mask.sum().item())
        if n_kept == 0:
            # nothing to supervise — count the iteration but skip optimizer.
            self._step_count += 1
            self._last_loss_components = {}
            if self.scene is not None:
                self.scene.record_step(self._step_count, 0.0, {})
            return 0.0
        l1 = (diff * mask).sum() / (n_kept * 3)
        loss = self.l1_weight * l1
        # Per-term breakdown — each entry is the *weighted* contribution to
        # `loss` (so they sum to ~loss). Read by tqdm + GUI status panel.
        components: dict[str, float] = {"l1": float((self.l1_weight * l1).detach().item())}

        # Void loss: penalise rendered alpha inside `~train_mask`. Without
        # this the masked region has zero gradient signal, so boundary
        # splats grow into it for free; here we explicitly drive their
        # rendered alpha to zero so they shrink / rotate / lose opacity.
        if self.void_weight > 0.0:
            void_mask = (~i.train_mask[idx]).float()        # (H,W) float
            n_void = int(void_mask.sum().item())
            if n_void > 0:
                alpha_img = alphas[0, ..., 0]               # (H,W) in [0,1]
                void = (alpha_img.pow(2) * void_mask).sum() / n_void
                loss = loss + self.void_weight * void
                components["void"] = float((self.void_weight * void).detach().item())

        # LPIPS over the full frame (the mask only weights L1; lpips wants a
        # spatially-contiguous image and per-pixel masking it makes no sense).
        if self.lpips_weight > 0.0:
            net = self._get_lpips_net()
            pred_bchw = pred.permute(2, 0, 1)[None]
            gt_bchw   = d.rgb[idx].permute(2, 0, 1)[None]
            lp = net(pred_bchw * 2 - 1, gt_bchw * 2 - 1).mean()
            loss = loss + self.lpips_weight * lp
            components["lpips"] = float((self.lpips_weight * lp).detach().item())

        # 2DGS-only regularizers. Distortion + internal normal-consistency
        # are gated by step-count warmups so the splats don't collapse onto
        # the rendered surface before they've learned RGB. DA3 depth + normal
        # supervision are active from step 0 (DA3 is the init source — these
        # just keep the disks anchored to those surfaces).
        if self.mode == "2dgs":
            if distort is not None and self._step_count >= self.dist_warmup_steps:
                dist_term = self.distortion_weight * distort.mean()
                loss = loss + dist_term
                components["distortion"] = float(dist_term.detach().item())
            if (normals is not None and surf_normals is not None
                    and self._step_count >= self.normal_warmup_steps):
                nc_term = self.normal_consistency_weight * (
                    1 - (normals * surf_normals).sum(-1)
                ).mean()
                loss = loss + nc_term
                components["normal_consistency"] = float(nc_term.detach().item())
            if self.depth_sup_weight > 0.0 and depth_pred is not None:
                z_gt = d.depth[idx]
                zmask = z_gt > 0.01
                if zmask.any():
                    dep_term = self.depth_sup_weight * (
                        (depth_pred - z_gt).abs()[zmask].mean()
                    )
                    loss = loss + dep_term
                    components["depth_sup"] = float(dep_term.detach().item())
            if (self.da3_normal_weight > 0.0 and self._da3_normals is not None
                    and normals is not None):
                nmask = d.depth[idx] > 0.01
                if nmask.any():
                    cos = (normals[0] * self._da3_normals[idx]).sum(-1)
                    da3_term = self.da3_normal_weight * (1 - cos[nmask]).mean()
                    loss = loss + da3_term
                    components["da3_normal"] = float(da3_term.detach().item())

        loss.backward()
        # Phase 4: zero gradients for frozen splats so their params stay
        # exactly put (Adam moments also stay 0 because of zero_grad below).
        # Cheaper than torch.no_grad-detaching the rows because we only need
        # to act on `.grad`, not the params themselves.
        if freeze_active:
            keep = self._frozen_mask
            for param in t.params.values():
                if param.grad is not None and param.shape[0] == keep.shape[0]:
                    param.grad[~keep] = 0
        for o in t.opts.values():
            o.step()
            o.zero_grad(set_to_none=True)

        if t.strategy is not None and not freeze_active:
            t.strategy.step_post_backward(p, t.opts, t.strategy_state,
                                          self._step_count, info, packed=False)

        # Cap gaussian scales at `scale_clamp_voxel_mult * voxel`. In-place
        # clamp under no_grad. After densify the Parameter object may be a
        # fresh tensor with new M; the dict lookup picks up the latest.
        if self._log_scale_max is not None:
            with torch.no_grad():
                p["scales"].clamp_(max=self._log_scale_max)

        loss_f = float(loss.item())
        self._last_loss_components = components
        self._step_count += 1
        if self.scene is not None:
            self.scene.record_step(self._step_count, loss_f, components)
            if self._step_count % self.publish_every == 0:
                self._publish_to_scene()
        return loss_f

    def _publish_to_scene(self) -> None:
        """Push current splat params into the shared SceneState. Activates
        pre-activation working tensors to match the post-activation
        convention the renderer expects (see PlyLoader at viewer.py:284-291).

        Publishes the full SH tensor. `scene.sh_degree` is set to the trained
        capacity `sh_max_deg`; unfilled bands are zero so the renderer using
        them has no visual effect until the ramp catches up. `num_splats`
        reflects the current count even after densify changes M."""
        if self.scene is None or self.train is None:
            return
        p = self.train.params
        with torch.no_grad():
            means = p["means"].detach()
            quats = F.normalize(p["quats"].detach(), dim=-1)
            scales = torch.exp(p["scales"].detach())
            opacities = torch.sigmoid(p["opacities"].detach())
            sh = (p["sh0"].detach() if p["shN"].shape[1] == 0
                  else torch.cat([p["sh0"].detach(), p["shN"].detach()], dim=1))
        with self.scene.write() as s:
            s.means, s.quats, s.scales = means, quats, scales
            s.opacities, s.sh = opacities, sh
            s.sh_degree = int(self.sh_max_deg)
            s.num_splats = int(means.shape[0])
            s.splat_version += 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(
        description="Barebones splat trainer scaffold: mp4 -> VIPE cameras."
    )
    p.add_argument("--video", type=Path, default=Path("outputs/zoomgs/videos/14.mp4"),
                   help="Input mp4 file.")
    p.add_argument("--name", type=str, default=None,
                   help="Subdirectory under --out-dir. Defaults to the video stem.")
    p.add_argument("--out-dir", type=Path, default=Path("vipe_outputs"),
                   help="Root output directory (default: ./outputs).")
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--remove-sky", type=int, default=1)
    p.add_argument("--confidence_quantile", type=float, default=0.6,
                   help="Percentile to keep in depth confidence")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--sh-max-deg", type=int, default=0,
                   help="0 = L1-only (v1). >0 enables SH-band progression 0→N.")
    p.add_argument("--lpips-weight", type=float, default=0.0,
                   help="LPIPS perceptual weight added to L1 (0 disables).")
    p.add_argument("--void-weight", type=float, default=0.5,
                   help="Penalty on rendered alpha inside `~train_mask` "
                        "(removes free-growth into masked regions). 0 disables.")
    p.add_argument("--densify", type=int, default=0,
                   help="1 enables gsplat DefaultStrategy clone/split/prune (v3).")
    p.add_argument("--densify-total-steps", type=int, default=7000,
                   help="Total steps the densify schedule plans for (sets refine_stop_iter).")
    p.add_argument("--mode", choices=("3dgs", "2dgs"), default="3dgs",
                   help="3dgs = gsplat.rasterization; 2dgs = rasterization_2dgs + "
                        "distortion/normal/depth/da3-normal regularizers (notebook recipe).")
    args = p.parse_args()

    #DEBUGGING
    args.max_frames = 32
    args.confidence_quantile = 0.6
    args.remove_sky = 1
    #args.video = Path("/home/kristofe/Documents/Projects/lyra/Lyra-2/outputs/zoomgs/videos/14.mp4")

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    trainer = SplatTrainer(output_root=args.out_dir)
    trainer.prepare_and_init(
        video=args.video,
        max_frames=args.max_frames,
        confidence=args.confidence_quantile,
        remove_sky=(args.remove_sky != 0),
        name=args.name,
        sh_max_deg=args.sh_max_deg,
        lpips_weight=args.lpips_weight,
        void_weight=args.void_weight,
        use_densify=(args.densify != 0),
        densify_total_steps=args.densify_total_steps,
        mode=args.mode,
    )
    trainer.refine_gaussians(args.steps)

    print("[splat_trainer] done")


if __name__ == "__main__":
    main()
