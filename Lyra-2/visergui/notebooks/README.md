# Splat notebooks

Pip deps for both notebooks: [requirements.txt](requirements.txt) (install torch
from the cu128 index first — instructions at the top of the file).

## `splat_init_and_train.ipynb`

Standalone, fully self-contained distillation of `visergui/splat_trainer.py`:
video (or folder of frames) → Depth-Anything-3 → confidence-quantile filter →
voxelized gaussian init → `_work/splats_voxel.ply` → optional gsplat training.
Every cell cites the `splat_trainer.py` lines it was distilled from; nothing is
imported from `visergui`.

Run it in the `splat` conda env (`Lyra-2/SPLAT_INSTALL.md`). Initialization does
**not** need `gsplat`/`fvdb`/`sam2`/`viser` — only torch, opencv, plyfile,
matplotlib and the editable `depth_anything_3` package.

Key parameters (first code cell, papermill-injectable):

| name | default | meaning |
|---|---|---|
| `VIDEO_OR_DIR` | `../arch_orbit1.mp4` | video file **or** folder of png/jpg frames |
| `MAX_FRAMES` | 32 | uniform-stride cap for video input (`<=0` = all) |
| `CONFIDENCE_QUANTILE` | 0.6 | keep pixels above this quantile of DA3 confidence |
| `REMOVE_SKY` | True | drop DA3-detected sky pixels |
| `MODEL_ID` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | HF checkpoint |
| `PROCESS_RES` | 504 | DA3 inference resolution — lower if you OOM |
| `TRAIN_STEPS` | 0 | `0` = init only; `>0` runs the gsplat training loop |
| `MODE` | `3dgs` | `3dgs` or `2dgs` (flat disks + distortion/normal regularizers) |
| `SH_MAX_DEG` | 2 | max spherical-harmonics degree (0 = flat color) |
| `SH_RAMP` | 1000 | steps per SH band unlock |
| `LPIPS_WEIGHT` | 0.05 | perceptual loss weight (0 disables) |
| `USE_DENSIFY` | True | gsplat clone/split/prune during training |

Besides the PLYs, the init run saves `_work/cameras.npz` (frames, K, poses,
depth) so downstream notebooks can reuse the scene without re-running DA3.

## `splat_segmentation.ipynb`

Distills the Segment tab (`segmenter.py` + `multiview_mask.py` +
`splat_trainer.select_splats_by_masks`): a **click** — here `(CLICK_FRAME,
CLICK_U, CLICK_V)` instead of a live viser pointer event — is backprojected via
rendered splat depth to a world point, the closest camera that sees it becomes
the seed frame, SAM 2 segments the seed and propagates the mask across all
frames as a video, and an occlusion-aware multi-view vote lifts the masks to a
splat selection. Outputs in `_work/segment/`: per-frame `objmask_*.png`,
`object.ply` + `background.ply` (partition of the input scene), and proof
figures (`click_seed.png`, `seed_mask.png`, `masks_montage.png`,
`segment_result.png`).

Needs `_work/` from an init run, `sam2`, and the SAM 2.1 hiera-large checkpoint
at `Lyra-2/vendor/InstaInpaint/checkpoints/sam2.1_hiera_large.pt` (URL in
requirements.txt).

## Smoke harness

```powershell
.\run_smoke.ps1                          # 8 frames, init only, default video
.\run_smoke.ps1 -MaxFrames 32            # full-size confirmation run
.\run_smoke.ps1 -Video ..\..\path\to\other.mp4
```

Training smoke runs on Windows go through `run_train_smoke.cmd`, which sets up the
MSVC + CUDA env that gsplat's JIT kernel build needs and forwards its args:

```bat
run_train_smoke.cmd -TrainSteps 600 -Mode 3dgs -ShDeg 2 -ShRamp 200 -LpipsWeight 0.05 -Densify 1
run_train_smoke.cmd -TrainSteps 800 -Mode 2dgs -ShDeg 1
```

Output lands in `_work\train_smoke.log` (last line `TRAIN_SMOKE_EXIT=<code>`);
`verify_train.py` checks the trained PLY (count, SH field count, params moved off
init values, loss curve rendered) in addition to `verify_init.py`.

`run_smoke.ps1` executes the notebook headlessly with papermill into
`_work/smoke_out.ipynb` (per-cell stdout + the failing cell's traceback survive
there), then runs `verify_init.py`, which checks: `_work/splats_voxel.ply` exists,
parses, has all Inria 3DGS fields, > 5000 gaussians, finite values, untouched init
opacity, non-degenerate spatial spread, and that both diagnostic PNGs rendered.
Exit code 0 = init milestone passed.

Iteration triage, by error class:

- `ModuleNotFoundError` → a DA3 runtime dep is missing from the env.
- CUDA OOM → rerun with `-p MODEL_ID <smaller DA3 variant>`, or `PROCESS_RES=336`,
  or fewer frames.
- `cv2 could not decode` → codec problem; extract frames externally and pass the
  folder instead.
- HF 401/timeouts → checkpoint download problem; pre-fetch with
  `huggingface_hub.snapshot_download(MODEL_ID)`. Never delete the HF cache
  (`%USERPROFILE%\.cache\huggingface`) between runs.
