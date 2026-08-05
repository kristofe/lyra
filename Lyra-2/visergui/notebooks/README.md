# Splat notebooks

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

## Smoke harness

```powershell
.\run_smoke.ps1                          # 8 frames, init only, default video
.\run_smoke.ps1 -MaxFrames 32            # full-size confirmation run
.\run_smoke.ps1 -Video ..\..\path\to\other.mp4
```

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
