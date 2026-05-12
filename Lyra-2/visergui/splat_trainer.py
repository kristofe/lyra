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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

import torch.nn.functional as F
import matplotlib.pyplot as plt
#import lpips
from tqdm.auto import tqdm
from plyfile import PlyData, PlyElement
from gsplat import rasterization, DefaultStrategy

# --------------------------------------------------------------------------- #
# Video loading
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# SplatTrainer scaffold (Phase 5 seam-compatible)
# --------------------------------------------------------------------------- #


class SplatTrainer:
    """Barebones trainer. Today: only `prepare_video()`. `step()` raises
    until real training is wired up. The class shape matches the
    `Trainer`-protocol seam in `visergui/training.py` so it can be wrapped
    in `BackgroundTrainingThread` later without restructuring."""

    def __init__(
        self,
        output_root: Path = Path("outputs"),
        device: str = "cuda",
        max_points : int = 1_000_000
    ) -> None:
        self.output_root = Path(output_root)
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.max_points = max_points
        self.output_dir: Path | None = None
        self.imgs: torch.Tensor
        self.depth: torch.Tensor
        self.K: torch.Tensor
        self.w2c34: torch.Tensor
        self.conf: torch.Tensor | None
        self.sky: torch.Tensor | None
        self.c2w: torch.Tensor
        self.N: int
        self.H: int
        self.W: int

        self._lpips_net = None

    def read_video_frames(self, output_dir: Path, video_path: Path, max_frames: int = -1) -> list[Path]:
        Path(output_dir).mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, total // max_frames)
        frame_paths = []
        if max_frames <= 0:
            max_frames = total
        
        for i in tqdm(range(0, max_frames), desc="Extracting frames", unit="frame"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * stride)
            ok, bgr = cap.read()
            if not ok:
                break
            p = f"{output_dir}/f{i:04d}.png"
            cv2.imwrite(p, bgr)
            frame_paths.append(p)
        cap.release()
        print(f"{len(frame_paths)} frames @ stride {stride} from {total} total")
        self.frame_paths = frame_paths

    def depth_anything_inference(self, device: torch.device):
        model = DepthAnything3.from_pretrained( "depth-anything/DA3NESTED-GIANT-LARGE-1.1").to(device).eval()
        pred = model.inference(
            image=self.frame_paths,
            process_res=504, #DA3's default value - here for clarity
            process_res_method="upper_bound_resize",#DA3's default value - here for clarity
        )

        self.imgs  = torch.from_numpy(pred.processed_images).to(device)   # (N,H,W,3) uint8
        self.depth = torch.from_numpy(pred.depth).to(device)              # (N,H,W) f32
        self.K     = torch.from_numpy(pred.intrinsics).to(device)         # (N,3,3)
        self.w2c34 = torch.from_numpy(pred.extrinsics).to(device)         # (N,3,4) OpenCV w2c
        conf_np = getattr(pred, "conf", None)
        if conf_np is None:
            conf_np = getattr(pred, "confidence", None)
        self.conf = torch.from_numpy(conf_np).to(device) if conf_np is not None else None
        sky_np = getattr(pred, "sky", None)
        self.sky = torch.from_numpy(sky_np).to(device).bool() if sky_np is not None else None

        self.N, self.H, self.W, _ = self.imgs.shape
        self.w2c = torch.eye(4, device=device).expand(self.N, 4, 4).clone()
        self.w2c[:, :3, :4] = self.w2c34
        self.c2w = torch.linalg.inv(self.w2c)

        del model
        torch.cuda.empty_cache()
        print(f"N={self.N} H={self.H} W={self.W}; depth range [{self.depth.min():.3f}, {self.depth.max():.3f}]")


    
    def process_video(
        self,
        video_path: Path,
        name: str | None = None,
        max_frames: int = -1,
    ):
        self.read_video_frames(self.output_root / (name or video_path.stem), video_path, max_frames=max_frames)
        self.depth_anything_inference(self.device)

    def initialize_gaussians(self, remove_sky : bool = True):
        #Put pixels as a point cloud in world space
        #Then create a gaussian splat per point.
        # position — unprojected world point.
        # color (SH DC) — (rgb - 0.5) / C0, the inverse of the standard 3DGS DC activation.
        # scale — one texel at the point's depth (z / fx), inflated by sqrt of subsample_ratio so gaussians cover the gaps left by random subsampling.
        # opacity — sigmoid^-1(0.9) ~ 2.20 (pre-activation).
        # rotation — identity quaternion.

        ii, jj = torch.meshgrid(
            torch.arange(self.H, device=self.device), torch.arange(self.W, device=self.device), indexing="ij")
        uv1 = torch.stack([jj, ii, torch.ones_like(ii)], -1).float()        # (H,W,3)
        self.rgb = self.imgs.float() / 255.0                                          # (N,H,W,3)

        Kinv = torch.linalg.inv(self.K)                                          # (N,3,3)
        cam_pts = torch.einsum("nij,hwj->nhwi", Kinv, uv1) * self.depth[..., None]  # (N,H,W,3)
        R, t = self.c2w[:, :3, :3], self.c2w[:, :3, 3]
        self.world = torch.einsum("nij,nhwj->nhwi", R, cam_pts) + t[:, None, None, :]

        valid = self.depth > 0
        if self.sky is not None and remove_sky:
            valid &= ~self.sky
        if self.conf is not None:
            conf_flat = self.conf.flatten()
            if conf_flat.numel() > 16_000_000:
                idx = torch.randint(0, conf_flat.numel(), (16_000_000,), device=conf_flat.device)
                thresh = conf_flat[idx].quantile(0.6)
                print(f'WARNING total unfiltered splats more than 16 Million, Capping')
            else:
                thresh = conf_flat.quantile(0.6)
            valid &= self.conf > thresh   # drop only the bottom 20% by confidence
        

        fx_nhw = self.K[:, 0, 0][:, None, None].expand(self.N, self.H, self.W)
        # Keep the full valid-sample tensors as cols_full / z_full / fx_full for the voxel cell.
        pts_full   = self.world[valid]
        cols_full  = self.rgb[valid]
        z_full     = self.depth[valid]
        fx_full    = fx_nhw[valid]

        
        #Subsample if there are too many points
        total_valid = pts_full.shape[0]
        if total_valid > self.max_points:
            sel = torch.randperm(total_valid, device=self.device)[:self.max_points]
            pts, cols, z_sel, fx_sel = pts_full[sel], cols_full[sel], z_full[sel], fx_full[sel]
            print(f'WARNING: Too many initial gaussians {total_valid} subsampling down to {self.max_points}')
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
        logit_o_init = torch.full((M,), 2.1972, device=self.device)            # sigmoid⁻¹(0.9)
        quats_init   = torch.zeros((M, 4), device=self.device); quats_init[:, 0] = 1.0
        scene_scale  = (pts.std(dim=0).mean()).item()
        print(f"M={M} gaussians (from {total_valid} valid samples), scene_scale={scene_scale:.3f}")


        sh_init = f_dc_init[:, None, :]
        self.save_inria_ply("splats_init.ply", means_init, sh_init, log_s_init, logit_o_init, quats_init)
        print(f"Saved splats_init.ply: {M} gaussians")
        self.render_and_show(means_init, quats_init, log_s_init, logit_o_init, sh_init, tag="init gaussians")


        # voxel init: voxel-downsample the RGBD points so multi-frame overlap
        # stops producing redundant co-located gaussians. typical reduction 3–10× vs v0, with
        # comparable coverage. uses the *full* valid-sample tensors (pts_full, cols_full, ...),
        # not the random-subsampled ones, so we benefit from every view.
        voxel_frac = 0.005                       # voxel edge as fraction of scene_scale; tweak for density
        voxel = max(scene_scale * voxel_frac, 1e-4)
        print(f"voxel size: {voxel:.4f}  (scene_scale={scene_scale:.3f})")

        keys = torch.floor(pts_full / voxel).long()                         # (t, 3)
        uniq_keys, inv = torch.unique(keys, dim=0, return_inverse=True)
        G = uniq_keys.shape[0]

        def _scatter_mean(vals, inv, g):
            """vals: (t,) or (t, d). returns mean over the `inv` partition, shape (g,) or (g, d)."""
            ones = torch.ones(vals.shape[0], device=vals.device)
            counts = torch.zeros(g, device=vals.device).scatter_add_(0, inv, ones).clamp_min(1)
            if vals.ndim == 1:
                s = torch.zeros(g, device=vals.device).scatter_add_(0, inv, vals)
                return s / counts
            D = vals.shape[1]
            s = torch.zeros(g, D, device=vals.device).scatter_add_(0, inv[:, None].expand(-1, D), vals)
            return s / counts[:, None]

        means_vox = _scatter_mean(pts_full,  inv, G)
        cols_vox  = _scatter_mean(cols_full, inv, G)
        z_vox     = _scatter_mean(z_full,    inv, G)
        fx_vox    = _scatter_mean(fx_full,   inv, G)

        # scale: roughly the larger of (one texel at this depth, inflated for remaining sparsity)
        # and (half the voxel edge, so neighbouring voxels overlap).
        inflate = max(1.0, (total_valid / G) ** 0.5)
        tex_vox = (z_vox / fx_vox * inflate).clamp_min(voxel * 0.5)

        means_vx_init   = means_vox.clone()
        f_dc_vx_init    = (cols_vox - 0.5) / C0
        log_s_vx_init   = torch.log(tex_vox[:, None].expand(G, 3).contiguous())
        logit_o_vx_init = torch.full((G,), 2.1972, device=self.device)
        quats_vx_init   = torch.zeros((G, 4), device=self.device); quats_vx_init[:, 0] = 1.0
        print(f"m_voxel={G} gaussians  ({100.0 * G / M:.1f}% of v0's {M}, inflate={inflate:.2f})")

        sh_vx = f_dc_vx_init[:, None, :]
        self.save_inria_ply("splats_voxel.ply", means_vx_init, sh_vx, log_s_vx_init, logit_o_vx_init, quats_vx_init)
        print(f"saved splats_voxel.ply")
        self.render_and_show(means_vx_init, quats_vx_init, log_s_vx_init, logit_o_vx_init, sh_vx, tag="v0 voxel init")

        
    def step(self) -> float:
        raise NotImplementedError(
            "splattrainer.step() is a placeholder; no training implemented yet."
        )
    
    def save_inria_ply(self, path, means, sh_all, log_s, logit_o, quats):
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


    def render_view(self, means, quats, log_s, logit_o, sh_all, cam_idx):
        sh_deg = int(round(sh_all.shape[1] ** 0.5)) - 1
        out, _, _ = rasterization(
            means, quats, torch.exp(log_s), torch.sigmoid(logit_o), sh_all,
            self.w2c[cam_idx:cam_idx+1], self.K[cam_idx:cam_idx+1], self.W, self.H,
            sh_degree=sh_deg, packed=False)
        return out[0].clamp(0, 1)


    def render_and_show(self, means, quats, log_s, logit_o, sh_all, tag, n_views=4):
        idxs = torch.linspace(0, self.N - 1, n_views).round().long().tolist()
        fig, ax = plt.subplots(2, n_views, figsize=(4 * n_views, 8))
        with torch.no_grad():
            for c, i in enumerate(idxs):
                r = self.render_view(means, quats, log_s, logit_o, sh_all, i).cpu().numpy()
                ax[0, c].imshow(self.rgb[i].cpu().numpy()); ax[0, c].set_title(f"gt frame {i}"); ax[0, c].axis("off")
                ax[1, c].imshow(r);                    ax[1, c].set_title(f"{tag} cam {i}"); ax[1, c].axis("off")
        plt.suptitle(tag); plt.tight_layout(); plt.show()

    def lpips_loss(pred_bchw, gt_bchw):
        """Inputs in [0,1], shape (B,3,H,W). Returns scalar LPIPS distance."""
        global _lpips_net
        if _lpips_net is None:
            #_lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
            for p in _lpips_net.parameters():
                p.requires_grad_(False)
        return _lpips_net(pred_bchw * 2 - 1, gt_bchw * 2 - 1).mean()



# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(
        description="Barebones splat trainer scaffold: mp4 -> VIPE cameras."
    )
    p.add_argument("--video", type=Path, default=Path("outputs/zoomgs/videos/14.mp4"), help="Input mp4 file.")
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Subdirectory under --out-dir. Defaults to the video stem.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("vipe_outputs"),
        help="Root output directory (default: ./outputs).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=-1,
    )
    p.add_argument(
        '--remove-sky',
        type=int,
        default=1
    )
    args = p.parse_args()

    #DEBUGGING
    args.max_frames = 32
    #args.video = Path("/home/kristofe/Documents/Projects/lyra/Lyra-2/outputs/zoomgs/videos/14.mp4")

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    trainer = SplatTrainer(output_root=args.out_dir)
    trainer.process_video(
        args.video, name=args.name, max_frames=args.max_frames
    )
    remove_sky = False if args.remove_sky == 0 else True
    trainer.initialize_gaussians(remove_sky=remove_sky)

    out = None
    print(f"[splat_trainer] done -> {out}")


if __name__ == "__main__":
    main()
