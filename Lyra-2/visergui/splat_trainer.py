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
from gsplat import rasterization, DefaultStrategy

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


def _preprocess_video(video_path: Path, out_dir: Path, device: torch.device,
                      max_frames: int) -> VideoData:
    frame_paths = _extract_frames(video_path, out_dir, max_frames)

    model = DepthAnything3.from_pretrained(
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    ).to(device).eval()
    pred = model.inference(
        image=frame_paths,
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
    print(f"N={N} H={H} W={W}; depth range [{depth.min():.3f}, {depth.max():.3f}]")

    return VideoData(rgb=rgb, depth=depth, K=K, w2c=w2c, c2w=c2w,
                     conf=conf, sky=sky, N=N, H=H, W=W)


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
    train_mask = data.depth > 0
    if data.sky is not None and remove_sky:
        train_mask &= ~data.sky
    if data.conf is not None and conf_thresh is not None:
        train_mask &= data.conf > conf_thresh
    print(f"loss mask: {train_mask.float().mean().item()*100:.1f}% pixels kept across {data.N} frames")

    return GaussianInit(
        means=means_vx, quats=quats_vx, log_s=log_s_vx, logit_o=logit_o_vx,
        sh=sh_vx, train_mask=train_mask,
        scene_scale=scene_scale, conf_thresh=conf_thresh,
        voxel=voxel,
    )


# --------------------------------------------------------------------------- #
# Free helpers — train state, rendering, PLY save
# --------------------------------------------------------------------------- #


def _make_train_state(init: GaussianInit, sh_max_deg: int = 0,
                      use_densify: bool = False,
                      densify_total_steps: int = 7000) -> TrainState:
    """Build the ParameterDict + per-key Adam optimizers + (optional)
    DefaultStrategy.

    `sh_max_deg > 0` allocates shN of shape (M, (sh_max_deg+1)²−1, 3) with
    its own Adam at 1/20 of the sh0 LR. `use_densify=True` wires up the
    gsplat clone/split/prune strategy; `means` LR is scaled by scene_scale
    to match the v3 notebook recipe."""
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
        # Aggressive schedule: start ~2.5x sooner, densify 2x more often,
        # reset opacity 2x more often, and keep refining longer at the tail.
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


def render_view(data: VideoData, means, quats, log_s, logit_o, sh_all, cam_idx):
    sh_deg = int(round(sh_all.shape[1] ** 0.5)) - 1
    out, _, _ = rasterization(
        means, quats, torch.exp(log_s), torch.sigmoid(logit_o), sh_all,
        data.w2c[cam_idx:cam_idx+1], data.K[cam_idx:cam_idx+1], data.W, data.H,
        sh_degree=sh_deg, packed=False)
    return out[0].clamp(0, 1)


def render_and_show(data: VideoData, means, quats, log_s, logit_o, sh_all,
                    tag: str, n_views: int = 4):
    idxs = torch.linspace(0, data.N - 1, n_views).round().long().tolist()
    fig, ax = plt.subplots(2, n_views, figsize=(4 * n_views, 8))
    with torch.no_grad():
        for c, i in enumerate(idxs):
            r = render_view(data, means, quats, log_s, logit_o, sh_all, i).cpu().numpy()
            ax[0, c].imshow(data.rgb[i].cpu().numpy()); ax[0, c].set_title(f"gt frame {i}"); ax[0, c].axis("off")
            ax[1, c].imshow(r);                         ax[1, c].set_title(f"{tag} cam {i}"); ax[1, c].axis("off")
    plt.suptitle(tag)
    plt.tight_layout()
    plt.savefig(f"{tag}.png", dpi=150, bbox_inches='tight')
    #plt.show()


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
        self.use_densify: bool = False
        self.densify_total_steps: int = 7000
        # Max gaussian axis-scale as a multiple of the init voxel edge. The
        # log-space cap is recomputed in prepare_and_init from init.voxel.
        self.scale_clamp_voxel_mult: float = 2.0
        self._log_scale_max: float | None = None

        self.data:  VideoData    | None = None
        self.init:  GaussianInit | None = None
        self.train: TrainState   | None = None

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
                         sh_ramp_steps_per_band: int = 1000,
                         use_densify: bool = False,
                         densify_total_steps: int = 7000) -> None:
        """One-call init: preprocess video (skipped if same video already
        processed), build initial gaussians, build train state, and publish
        to the scene.

        `sh_max_deg` > 0 enables SH-band progression (0→sh_max_deg) ramped
        every `sh_ramp_steps_per_band` steps. `lpips_weight` > 0 adds an
        LPIPS perceptual term to the L1 loss. `use_densify` wires up the
        gsplat clone/split/prune strategy; `densify_total_steps` sets when
        the strategy stops refining (refine_stop_iter = total - 500)."""
        self.sh_max_deg = int(sh_max_deg)
        self.lpips_weight = float(lpips_weight)
        self.l1_weight = float(l1_weight)
        self.sh_ramp_steps_per_band = max(1, int(sh_ramp_steps_per_band))
        self.use_densify = bool(use_densify)
        self.densify_total_steps = int(densify_total_steps)

        video = Path(video)
        signature = (video, int(max_frames))
        out_dir = self.output_root / (name or video.stem)
        if self._last_video != signature or self.data is None:
            self.data = _preprocess_video(video, out_dir, self.device, max_frames)
            self._last_video = signature
        self.init = _build_initial_gaussians(
            self.data, self.max_points,
            confidence=confidence, remove_sky=remove_sky,
        )
        self.train = _make_train_state(
            self.init, sh_max_deg=self.sh_max_deg,
            use_densify=self.use_densify,
            densify_total_steps=self.densify_total_steps,
        )
        # Cap each gaussian axis-scale at `scale_clamp_voxel_mult * voxel`.
        # Clamping is applied after each optimizer step in log-space.
        self._log_scale_max = math.log(self.scale_clamp_voxel_mult * self.init.voxel)
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

    def set_scale_clamp(self, voxel_mult: float) -> None:
        """Update the per-axis scale cap as a multiple of the init voxel edge.
        Takes effect on the next `step()`. Safe to call before init (the cap
        will be re-derived from `init.voxel` once prepare_and_init runs)."""
        self.scale_clamp_voxel_mult = float(voxel_mult)
        if self.init is not None:
            self._log_scale_max = math.log(self.scale_clamp_voxel_mult * self.init.voxel)

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

    def refine_gaussians(self, steps: int) -> None:
        """Headless CLI training loop. Requires `prepare_and_init` first."""
        if self.train is None or self.data is None:
            raise RuntimeError("call prepare_and_init() before refine_gaussians()")

        tag, desc = "splats", "training"
        pbar = tqdm(range(steps), desc=desc, unit="step")
        for s in pbar:
            loss = self.step()
            if s % 100 == 0:
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    M=int(self.train.params["means"].shape[0]),
                    sh=self._current_sh_degree(),
                )

        torch.set_grad_enabled(False)
        self.save_current(f"{tag}.ply")
        print(f"Saved {tag}.ply")
        # Snapshot for render_and_show — mirrors save_current's sh-cat logic.
        p = self.train.params
        sh_all = (p["sh0"] if p["shN"].shape[1] == 0
                  else torch.cat([p["sh0"], p["shN"]], dim=1))
        render_and_show(self.data, p["means"], p["quats"],
                        p["scales"], p["opacities"], sh_all, tag=tag)

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

    def step(self) -> float:
        if (not self._initialized or self.train is None
                or self.data is None or self.init is None):
            time.sleep(0.05)   # avoid busy-spin if thread is resumed early
            return 0.0
        t, d, i = self.train, self.data, self.init
        p = t.params
        idx = int(torch.randint(0, d.N, (1,)).item())

        # SH degree ramps 0→sh_max_deg. `cat` with a (M,0,3) shN is a no-op,
        # so the same path covers sh_max_deg=0.
        cur_sh = self._current_sh_degree()
        sh_all = torch.cat([p["sh0"], p["shN"]], dim=1)

        out, _, info = rasterization(
            p["means"], p["quats"],
            torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
            sh_all, d.w2c[idx:idx+1], d.K[idx:idx+1], d.W, d.H,
            sh_degree=cur_sh, packed=False)
        pred = out[0]                                       # (H,W,3) float

        if t.strategy is not None:
            t.strategy.step_pre_backward(p, t.opts, t.strategy_state,
                                         self._step_count, info)

        # Masked L1. The mask excludes sky / low-confidence-depth pixels.
        diff = (pred - d.rgb[idx]).abs()
        mask = i.train_mask[idx].unsqueeze(-1)              # (H,W,1) bool
        n_kept = int(mask.sum().item())
        if n_kept == 0:
            # nothing to supervise — count the iteration but skip optimizer.
            self._step_count += 1
            if self.scene is not None:
                self.scene.record_step(self._step_count, 0.0)
            return 0.0
        l1 = (diff * mask).sum() / (n_kept * 3)
        loss = self.l1_weight * l1

        # LPIPS over the full frame (the mask only weights L1; lpips wants a
        # spatially-contiguous image and per-pixel masking it makes no sense).
        if self.lpips_weight > 0.0:
            net = self._get_lpips_net()
            pred_bchw = pred.permute(2, 0, 1)[None]
            gt_bchw   = d.rgb[idx].permute(2, 0, 1)[None]
            lp = net(pred_bchw * 2 - 1, gt_bchw * 2 - 1).mean()
            loss = loss + self.lpips_weight * lp

        loss.backward()
        for o in t.opts.values():
            o.step()
            o.zero_grad(set_to_none=True)

        if t.strategy is not None:
            t.strategy.step_post_backward(p, t.opts, t.strategy_state,
                                          self._step_count, info, packed=False)

        # Cap gaussian scales at `scale_clamp_voxel_mult * voxel`. In-place
        # clamp under no_grad. After densify the Parameter object may be a
        # fresh tensor with new M; the dict lookup picks up the latest.
        if self._log_scale_max is not None:
            with torch.no_grad():
                p["scales"].clamp_(max=self._log_scale_max)

        loss_f = float(loss.item())
        self._step_count += 1
        if self.scene is not None:
            self.scene.record_step(self._step_count, loss_f)
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
    p.add_argument("--densify", type=int, default=0,
                   help="1 enables gsplat DefaultStrategy clone/split/prune (v3).")
    p.add_argument("--densify-total-steps", type=int, default=7000,
                   help="Total steps the densify schedule plans for (sets refine_stop_iter).")
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
        use_densify=(args.densify != 0),
        densify_total_steps=args.densify_total_steps,
    )
    trainer.refine_gaussians(args.steps)

    print("[splat_trainer] done")


if __name__ == "__main__":
    main()
