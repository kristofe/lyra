"""Machine-checkable success criteria for the notebook's TRAINING stage.

Usage: python verify_train.py [work_dir] [expected_sh_deg]
Exit 0 = trained artifacts present and sane.
"""
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


work = Path(sys.argv[1] if len(sys.argv) > 1 else "_work")
sh_deg = int(sys.argv[2]) if len(sys.argv) > 2 else None

ply_path = work / "splats_trained.ply"
if not ply_path.exists():
    fail(f"{ply_path} missing")
v = PlyData.read(str(ply_path))["vertex"].data
names = v.dtype.names

n = len(v)
if n <= 1000:
    fail(f"only {n} gaussians in trained ply")

if sh_deg is not None:
    k_rest = 3 * ((sh_deg + 1) ** 2 - 1)
    have = len([f for f in names if f.startswith("f_rest_")])
    if have != k_rest:
        fail(f"expected {k_rest} f_rest fields for SH degree {sh_deg}, found {have}")

xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
if not np.isfinite(xyz).all():
    fail("non-finite positions")
for f in names:
    if not np.isfinite(np.asarray(v[f], dtype=np.float64)).all():
        fail(f"non-finite values in {f}")

# training must have MOVED the parameters off the init constants
if np.allclose(v["opacity"], 2.1972, atol=1e-4):
    fail("opacities untouched (still exactly the init logit) — did training run?")
if np.allclose(v["rot_1"], 0.0, atol=1e-6) and np.allclose(v["rot_2"], 0.0, atol=1e-6):
    fail("rotations untouched (still exactly identity) — did training run?")

loss_png = work / "train_loss.png"
if not loss_png.exists() or loss_png.stat().st_size < 5_000:
    fail("train_loss.png missing or too small")

# init ply must also still exist (goal: PLY before AND after training)
if not (work / "splats_voxel.ply").exists():
    fail("splats_voxel.ply (init PLY) missing")

print(f"PASS: trained ply has {n} gaussians, "
      f"{len([f for f in names if f.startswith('f_rest_')])} f_rest fields, "
      f"params moved off init values")
