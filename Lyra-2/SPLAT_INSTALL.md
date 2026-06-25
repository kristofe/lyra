# Installation — `splat` env (Gaussian splatting / visergui)

The `splat` conda environment powers the visergui splat trainer / viewer
(`visergui/splat_trainer.py`, `viewer.py`, `mesher.py`). It is **separate** from
the `lyra2` inference env in [INSTALL.md](INSTALL.md) / [INSTALL_BLACKWELL.md](INSTALL_BLACKWELL.md):
Python 3.10, **torch 2.10.0 + cu128**, gsplat, fvdb, sam2.

All pins live in [requirements-splat.txt](requirements-splat.txt).

## Prerequisites

- An NVIDIA GPU + driver that supports CUDA 12.8 (Blackwell `sm_120` tested).
- A **CUDA 12.8 dev toolkit** (step 2 below installs one into the conda env).
  `fvdb-core` ships only as an sdist on PyPI, so pip **compiles it from source**
  against the installed torch — this needs `nvcc` **plus the CUDA headers and
  cudart dev library** (`cuda-cudart-dev`), and the toolkit version must match
  torch's `cu128`. A bare `nvcc` is not enough — and a partial *system* CUDA 13
  install (common on cloud studios) makes the build fail with
  `Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY)`.
  (`gsplat` JIT-compiles its CUDA kernels at first run, so it also needs `nvcc`
  on PATH at runtime.)
- The repo cloned **with submodules** — `depth_anything_3` is a submodule and is
  installed editable from inside this repo.

## Install

```bash
# 0. Clone with submodules (or init them in an existing clone):
git clone --recursive <repo-url>
cd Lyra-2
git submodule update --init --recursive      # ensures depth_anything_3 is populated

# 1. Create the env (Python 3.10 — the pins in requirements-splat.txt are a
#    3.10 freeze; a different Python may not have matching wheels for every pin).
conda create -n splat -y python=3.10 pip
conda activate splat

# 2. CUDA 12.8 dev toolkit INTO the env, so fvdb-core's source build can find
#    nvcc + cudart + headers, and so a stray system CUDA (e.g. 13.x preinstalled
#    on the box) doesn't get picked up. Match torch's cu128.
conda install -y -c nvidia/label/cuda-12.8.1 -c conda-forge \
  cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev cuda-cccl \
  cuda-libraries-dev cuda-driver-dev libnvjitlink-dev cuda-nvtx
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version                                # MUST say "release 12.8", not 13.x

# 3. Bulk deps (run from the repo root). Pulls the cu128 torch wheels from the
#    PyTorch index and builds fvdb-core from source against the toolkit above.
#    --no-build-isolation makes the source build link the installed cu128 torch.
pip install --no-build-isolation -r requirements-splat.txt

# 4. Editable, in-repo Depth-Anything-3 (imported by visergui/splat_trainer.py).
#    Kept OUT of requirements-splat.txt, and installed with --no-deps, because:
#      - pip resolves -e paths against your CWD (not the file location), so
#        listing it there breaks `pip install -r` when run from outside the repo;
#      - DA3's pyproject pins numpy<2, which conflicts with the numpy==2.2.6 in
#        requirements-splat.txt. --no-deps keeps numpy 2.x — DA3 imports fine
#        under it and all its real runtime deps are already installed by step 3.
pip install --no-deps -e ./lyra_2/_src/inference/depth_anything_3
```

## Verify

```bash
python -c "
import torch, gsplat, fvdb, sam2
from fvdb import Grid
from depth_anything_3.api import DepthAnything3
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('device cc:', torch.cuda.get_device_capability(0))
print('all imports OK')
"
```

## Native OpenGL viewer

`visergui/native_viewer.py` is a standalone, browser-free viewer: it reuses the
exact same `gsplat` rasterizer as `viewer.py` but blits the rendered frame to a
local OpenGL window (glfw + moderngl) instead of shipping it to a viser browser
tab. Output is pixel-identical to training; it just drops the websocket hop, so
it's much snappier for local inspection. Deps (`glfw`, `moderngl`, `glcontext`)
are pinned in `requirements-splat.txt`.

```bash
# from the repo root, in the splat env:
python visergui/native_viewer.py --ply splats.ply
# controls: left-drag orbit · right-drag pan · scroll/W/S zoom ·
#           R reset · C cycle RGB/Depth/Normals · [ ] FOV · ESC/Q quit
```

Needs a real desktop GL context (run on the workstation, not headless SSH; for
headless use `xvfb-run`). First run may pause ~1 min while gsplat JIT-compiles
its CUDA kernels; the script auto-prepends a complete CUDA toolkit include to
`CPATH` so that compile can't fail on a missing `cuda_runtime.h` / `crt/`.

**Env gotcha (this box):** the login profile hard-prepends `…/envs/lyra2/bin` to
`PATH`, so a bare `python` — even after `conda activate splat` or under
`conda run -n splat` — resolves to the **lyra2** interpreter. Run the splat
interpreter by absolute path while keeping `conda activate splat` for the CUDA
`activate.d` env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate splat
/home/kristofe/miniconda3/envs/splat/bin/python visergui/native_viewer.py --ply splats.ply
```

## Gotchas

- **`fvdb-core==0.4.2+pt210.cu128` is not downloadable.** The `+pt210.cu128`
  local tag is stamped by the *source build* on this machine; it exists on no
  index. requirements-splat.txt therefore pins plain `fvdb-core==0.4.2` and lets
  pip build it.
- **fvdb-core build fails with `Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS
  CUDA_CUDART_LIBRARY)` / `Caffe2 ... cannot find the CUDA libraries`.** The
  source build needs a full CUDA 12.8 dev toolkit, not just `nvcc`. Do step 2
  (install `cuda-cudart-dev` et al. into the env) and `export CUDA_HOME=$CONDA_PREFIX`.
  If `nvcc --version` reports 13.x, a preinstalled system CUDA is shadowing the
  env — re-`export PATH="$CUDA_HOME/bin:$PATH"` in the same shell before retrying.
- **`depth_anything_3` editable install fails with "not a valid editable
  requirement"** → the submodule wasn't initialized (the directory is an empty
  gitlink with no `pyproject.toml`). Run `git submodule update --init --recursive`.
- **`ResolutionImpossible ... depth-anything-3 ... depends on numpy<2` vs
  `numpy==2.2.6`** → you're installing DA3 in the same pass as the rest (an old
  requirements-splat.txt that still listed `-e ...depth_anything_3` on the last
  line). Use the current file (DA3 is not in it) and install DA3 separately with
  `--no-deps` (step 4).
- **`viser` shows blank point clouds** on 1.0.25–1.0.27 (broken point-cloud
  shader). The env pins `viser==1.0.27`; if you hit this, pin `viser==1.0.24`
  (see note 6 in [INSTALL_BLACKWELL.md](INSTALL_BLACKWELL.md)).
