"""Machine-checkable success criteria for splat_segmentation.ipynb.

Usage: python verify_seg.py [seg_out_dir] [cameras_npz] [source_ply]
Exit 0 = segmentation artifacts present and sane.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


out = Path(sys.argv[1] if len(sys.argv) > 1 else "_work/segment")
npz_path = Path(sys.argv[2] if len(sys.argv) > 2 else "_work/cameras.npz")
src_ply = Path(sys.argv[3] if len(sys.argv) > 3 else "_work/splats_voxel.ply")

N = int(np.load(npz_path)["rgb"].shape[0])

# per-frame masks: all present, a majority non-empty
mask_files = sorted(out.glob("objmask_*.png"))
if len(mask_files) != N:
    fail(f"expected {N} objmask PNGs, found {len(mask_files)}")
nonempty = sum(1 for f in mask_files if np.asarray(Image.open(f)).any())
if nonempty < (N + 1) // 2:
    fail(f"object mask non-empty in only {nonempty}/{N} frames — propagation failed?")

# object + background PLYs partition the source scene
def count(p):
    if not p.exists():
        fail(f"{p} missing")
    return len(PlyData.read(str(p))["vertex"].data)

n_obj = count(out / "object.ply")
n_bg = count(out / "background.ply")
n_src = count(src_ply)
if n_obj + n_bg != n_src:
    fail(f"object({n_obj}) + background({n_bg}) != source({n_src})")
if not (100 < n_obj < 0.9 * n_src):
    fail(f"object selection degenerate: {n_obj} of {n_src} splats")

for img in ("click_seed.png", "seed_mask.png", "masks_montage.png", "segment_result.png"):
    p = out / img
    if not p.exists() or p.stat().st_size < 10_000:
        fail(f"{img} missing or suspiciously small")

print(f"PASS: masks in {nonempty}/{N} frames; object {n_obj} + background {n_bg} "
      f"= {n_src} splats; all figures rendered")
