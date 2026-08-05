"""Machine-checkable success criteria for the notebook's init stage.

Usage: python verify_init.py [work_dir]   (default: _work)
Exit 0 = initialization artifacts are present and sane.
"""
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


work = Path(sys.argv[1] if len(sys.argv) > 1 else "_work")

ply_path = work / "splats_voxel.ply"
if not ply_path.exists():
    fail(f"{ply_path} missing")
v = PlyData.read(str(ply_path))["vertex"].data

required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
missing = [f for f in required if f not in v.dtype.names]
if missing:
    fail(f"missing PLY fields: {missing}")

n = len(v)
if n <= 5000:
    fail(f"only {n} gaussians (expected > 5000)")

xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
rest = np.stack([np.asarray(v[f], dtype=np.float64) for f in required[3:]], 1)
if not (np.isfinite(xyz).all() and np.isfinite(rest).all()):
    fail("non-finite values in PLY")

if not np.allclose(v["opacity"], 2.1972, atol=1e-3):
    fail("opacity logits != 2.1972 — not an untouched init PLY")

std = xyz.std(0)
if not (std > 1e-6).all():
    fail(f"degenerate spatial spread: std={std}")

for img in ("init_diagnostics.png", "init_scatter.png"):
    p = work / img
    if not p.exists() or p.stat().st_size < 10_000:
        fail(f"{img} missing or suspiciously small")

print(f"PASS: {n} gaussians, spatial std={np.round(std, 3).tolist()}, all checks green")
