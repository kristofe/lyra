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
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Video loading
# --------------------------------------------------------------------------- #
class DepthPrediction:
    def __init__(self):
        self.imgs: torch.Tensor
        self.depth: torch.Tensor
        self.K: torch.Tensor
        self.w2c34: torch.Tensor
        self.conf: torch.Tensor | None
        self.c2w: torch.Tensor
        self.N: int
        self.H: int
        self.W: int

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
        return frame_paths

    def depth_anything_inference(self, device: torch.device):
        model = DepthAnything3.from_pretrained( "depth-anything/DA3NESTED-GIANT-LARGE-1.1").to(device).eval()
        pred = model.inference(
            image=self.frame_paths,
            process_res=504, #DepthAnything3's default is 512, but we use 504 to get an exact 16x downsample for the 16x16 grid of splats in Phase 5.
            process_res_method="upper_bound_resize",
        )

        self.imgs  = torch.from_numpy(pred.processed_images).to(device)   # (N,H,W,3) uint8
        self.depth = torch.from_numpy(pred.depth).to(device)              # (N,H,W) f32
        self.K     = torch.from_numpy(pred.intrinsics).to(device)         # (N,3,3)
        self.w2c34 = torch.from_numpy(pred.extrinsics).to(device)         # (N,3,4) OpenCV w2c
        conf_np = getattr(pred, "conf", None)
        if conf_np is None:
            conf_np = getattr(pred, "confidence", None)
        self.conf = torch.from_numpy(conf_np).to(device) if conf_np is not None else None

        self.N, self.H, self.W, _ = self.imgs.shape
        w2c = torch.eye(4, device=device).expand(self.N, 4, 4).clone()
        w2c[:, :3, :4] = self.w2c34
        self.c2w = torch.linalg.inv(w2c)

        del model
        torch.cuda.empty_cache()
        print(f"N={self.N} H={self.H} W={self.W}; depth range [{self.depth.min():.3f}, {self.depth.max():.3f}]")



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
    ) -> None:
        self.output_root = Path(output_root)
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.output_dir: Path | None = None
    
    def prepare_video(
        self,
        video_path: Path,
        name: str | None = None,
        max_frames: int = -1,
    ):
        depth_pred = DepthPrediction()
        frame_paths = depth_pred.read_video_frames(self.output_root / (name or video_path.stem), video_path, max_frames=max_frames)
        depth_pred.depth_anything_inference(self.device)


    def step(self) -> float:
        raise NotImplementedError(
            "SplatTrainer.step() is a placeholder; no training implemented yet."
        )


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
    args = p.parse_args()

    #DEBUGGING
    args.stride = 16
    #args.video = Path("/home/kristofe/Documents/Projects/lyra/Lyra-2/outputs/zoomgs/videos/14.mp4")

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    trainer = SplatTrainer(output_root=args.out_dir)
    out = trainer.prepare_video(
        args.video, name=args.name, max_frames=args.max_frames
    )
    print(f"[splat_trainer] done -> {out}")


if __name__ == "__main__":
    main()
