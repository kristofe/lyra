# Installation (Blackwell / RTX PRO 6000)

This is the Blackwell-tailored variant of [INSTALL.md](INSTALL.md). It targets NVIDIA Blackwell consumer/workstation GPUs with compute capability `sm_120` (e.g. RTX PRO 6000 Blackwell, RTX 5090). Tested with **driver 580+** and the **conda CUDA 12.8 toolkit**. Other Linux distributions and CUDA 12.8+ should also work but are not officially verified.

Six changes versus the H100 instructions:

1. **gcc 14.3 instead of 13.3.** `nvcc` in the current `nvidia/label/cuda-12.8.x` channels lists only `gxx_linux-64` 11.2 or 14.3 as compatible host compilers — gcc 13.3 is rejected by the conda solver. CUDA 12.8 officially supports gcc 14, and downstream packages (PyTorch wheels, flash-attn, transformer_engine) build fine with it.
2. **CUDA label `12.8.1`, not `12.8.0`.** Same toolkit version; the 12.8.1 patch ships nvcc 12.8.93 which has the gcc 14 dep wired up correctly. The 12.8.0 label's repodata was missing valid Linux solutions for the current conda-forge gcc lineup.
3. **cuDNN bumped to 9.21.1.3.** PyTorch 2.7.1 cu128 ships `nvidia-cudnn-cu12==9.7.1.26`, which is below the `sm_120` minimum of 9.18.1. Without this upgrade, transformer_engine logs `cudnn_runtime_version < 91801 is not supported` and falls back to unfused attention, OOMing at ~109 GB allocations even on 96 GB Blackwell GPUs.
4. **flash-attn arch list restricted.** `flash-attn==2.6.3` source has no `sm_120` kernels and will fail to build if PyTorch's default `TORCH_CUDA_ARCH_LIST` includes `12.0`. We restrict the build to pre-Blackwell archs. `flash_attn` is only invoked on SM90 by [`lyra_2/_src/modules/attention.py`](lyra_2/_src/modules/attention.py) (lines 119–138 dispatch to PyTorch SDPA on Blackwell), so the package just needs to import cleanly — its kernels are never launched on `sm_120`.
5. **Activate hook strips system `/usr/local/cuda-*/lib*` from `LD_LIBRARY_PATH` AND `LD_PRELOAD`s a tiny `libnocudart13.so` shim.** Many Blackwell workstations ship with system CUDA 13 registered in `/etc/ld.so.cache` via `/etc/ld.so.conf.d/{000_cuda,987_cuda-13,gds-13-*}.conf`. Stripping LD_LIBRARY_PATH alone is *insufficient* because the system cache still resolves `libcudart.so.13`. transformer_engine 2.14 dlopens both `libcudart.so.12` and `libcudart.so.13` and aborts with `RuntimeError: Multiple libcudart libraries found` if both succeed. The shim wraps `dlopen()` and returns NULL for `libcudart.so.13` — surgical, no sudo, no impact on other conda envs that legitimately need system CUDA 13 (e.g. a sibling Jupyter env).
6. An explicit `git submodule update` step is added in case the repo was not cloned with `--recursive`.

```bash
# 0. Clone repository
git clone --recursive git@github.com:nv-tlabs/lyra.git
cd Lyra-2

# 0a. Ensure submodules are present (vipe, depth_anything_3)
#     Skip if you cloned with --recursive and the submodule dirs are populated.
git submodule update --init --recursive

# 1. Create conda environment
#    Use --override-channels everywhere. Without it, your global ~/.condarc
#    almost certainly has `defaults` (pkgs/main) at top priority, which silently
#    wins over conda-forge/nvidia and yanks in CUDA 13.x or a broken
#    `defaults::gdk-pixbuf` whose post-link script exits 1 on Linux.
conda create -n lyra2 --override-channels -c conda-forge -y \
  python=3.10 pip cmake ninja libgl ffmpeg packaging
conda activate lyra2
CONDA_BACKUP_CXX="" conda install --override-channels -c conda-forge -y \
  gcc=14.3.0 gxx=14.3.0 eigen zlib

# 2. Install CUDA dev components inside the conda environment
#    Notes:
#    - We install only the dev subpackages we need — NOT the `cuda` or
#      `cuda-toolkit` metapackages. Both transitively pull in `cuda-tools` →
#      `cuda-visual-tools` → `nsight-compute` / `cuda-nvvp`, which depend on
#      gdk-pixbuf (whose `defaults` build has a broken post-link script).
#      PyTorch 2.7.1 cu128 ships its own runtime libs; we only need conda's
#      CUDA for source builds (flash-attn, transformer_engine, vipe, da3).
#    - --override-channels forces conda to ignore ~/.condarc and use ONLY
#      nvidia/label/cuda-12.8.1 + conda-forge. Without this, defaults can pull
#      in CUDA 13.x packages (verified: defaults' cuda-nvcc was 13.1.115).
#    - We pair the cuda-12.8.1 label with gcc 14.3 from step 1. Other
#      combinations fail to solve: nvcc on these labels accepts only
#      gxx_linux-64 11.2 or 14.3 as host compilers (no 12.x or 13.x), and the
#      12.8.0 label's Linux repodata is missing valid solutions for current
#      conda-forge gcc.
conda install --override-channels -c nvidia/label/cuda-12.8.1 -c conda-forge -y \
  cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev cuda-cccl \
  cuda-libraries-dev cuda-cupti cuda-cuobjdump cuda-cuxxfilt \
  cuda-driver-dev cuda-nvtx cuda-profiler-api libnvjitlink-dev

# 2a. Pin CUDA to the conda env on every activation.
#     If your system has its own CUDA at /usr/local/cuda-* (very common on
#     Blackwell workstations shipping with CUDA 13), it will likely sit ahead
#     of the conda env's bin on PATH and shadow `nvcc`. This activate hook
#     re-prepends the env bin and sets CUDA_HOME after every `conda activate`.
#     It also LD_PRELOADs a 5-line shim that hides libcudart.so.13 from
#     transformer_engine, which would otherwise abort because the system
#     /etc/ld.so.cache still resolves libcudart.so.13 even with LD_LIBRARY_PATH
#     scrubbed. See header note 5 for full reasoning.

# Build the libcudart.so.13 dlopen-blocker shim:
cat > /tmp/nocudart13.c << 'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stddef.h>
void *dlopen(const char *filename, int flags) {
    static void *(*real_dlopen)(const char *, int) = NULL;
    if (!real_dlopen) real_dlopen = dlsym(RTLD_NEXT, "dlopen");
    if (filename && strstr(filename, "libcudart.so.13")) return NULL;
    return real_dlopen(filename, flags);
}
CEOF
gcc -shared -fPIC -O2 -o "$CONDA_PREFIX/lib/libnocudart13.so" /tmp/nocudart13.c -ldl

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_path.sh" << 'EOF'
export CUDA_HOME_BACKUP="${CUDA_HOME:-}"
export LD_LIBRARY_PATH_BACKUP="${LD_LIBRARY_PATH:-}"
export LD_PRELOAD_BACKUP="${LD_PRELOAD:-}"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"

SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
# Strip any /usr/local/cuda-*/lib* that the user's shell adds (e.g. via ~/.bashrc).
_clean_ld="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/usr/local/cuda-' | paste -sd: -)"
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$SITE/nvidia/nvtx/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64${_clean_ld:+:$_clean_ld}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# Block dlopen of libcudart.so.13 so transformer_engine 2.14 doesn't trip its
# "Multiple libcudart libraries found" check (the system /etc/ld.so.cache still
# resolves .13 even after stripping LD_LIBRARY_PATH).
export LD_PRELOAD="$CONDA_PREFIX/lib/libnocudart13.so${LD_PRELOAD:+:$LD_PRELOAD}"

unset SITE _clean_ld
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cuda_path.sh" << 'EOF'
export CUDA_HOME="${CUDA_HOME_BACKUP:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_BACKUP:-}"
export LD_PRELOAD="${LD_PRELOAD_BACKUP:-}"
unset CUDA_HOME_BACKUP LD_LIBRARY_PATH_BACKUP LD_PRELOAD_BACKUP
EOF
conda deactivate && conda activate lyra2

# Verify you got CUDA 12.8 (not 13.x) AND the conda env's nvcc wins:
which nvcc                                     # /home/.../envs/lyra2/bin/nvcc
nvcc --version | grep -E "release [0-9]+\.[0-9]+"
# Expected: "Cuda compilation tools, release 12.8, V12.8.93"

# 3. Install PyTorch (cu128 wheels include sm_120 kernels)
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

# 3a. Upgrade cuDNN above the sm_120 minimum (≥ 9.18.1).
#     PyTorch 2.7.1 cu128 wheels pin nvidia-cudnn-cu12==9.7.1.26, which is too old
#     for Blackwell — at runtime cuDNN logs:
#       "Given combination of sm_arch_ == 120 and cudnn_runtime_version < 91801 is not supported"
#     and transformer_engine then falls back from fused cuDNN attention to UNFUSED
#     attention, which materializes the full N×N attention matrix and OOMs at ~109 GB
#     allocations even on 96 GB Blackwell GPUs. The pip resolver will warn that 9.21
#     conflicts with torch's pin; the warning is cosmetic — cuDNN 9.x is forward-ABI-
#     compatible and torch 2.7.1 loads it fine.
pip install --upgrade nvidia-cudnn-cu12==9.21.1.3

# 4. Set build environment variables
#    Includes nvidia/nvtx/include — the conda `cuda-nvtx` toolkit package does
#    NOT ship nvtx3/nvToolsExt.h under $CUDA_HOME/include. The header lives
#    inside the nvidia-nvtx-cu12 pip wheel that PyTorch pulls in. Without this,
#    transformer_engine's source build fails with
#    "fatal error: nvtx3/nvToolsExt.h: No such file or directory".
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$SITE/nvidia/nvtx/include:$CPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# 5. Install Python dependencies
pip install --no-deps -r requirements.txt
# requirements.txt pins `hatchling` and `editables` but not their runtime deps;
# step 7 builds depth_anything_3 via hatchling with --no-build-isolation, which
# needs these. Also yacs (fvcore) and asciitree (zarr 2.18.3) are missed by the
# upstream `--no-deps` install. gdown is pinned because VIPE calls
# gdown.download(..., fuzzy=True) to fetch DROID-SLAM weights; gdown 6.0.0
# removed the `fuzzy` kwarg, breaking VIPE's first-run checkpoint download.
pip install pathspec pluggy trove-classifiers yacs asciitree 'gdown==5.2.0'
pip install "git+https://github.com/microsoft/MoGe.git"
pip install --no-build-isolation "transformer_engine[pytorch]"
# Symlink cuda_runtime as cudart for transformer_engine compatibility
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

# 6. Install Flash Attention
#    flash-attn 2.6.3 has no sm_120 kernels. Restrict the build to pre-Blackwell
#    archs so compilation succeeds. Lyra-2 only calls flash_attn on Hopper (SM90);
#    on Blackwell it dispatches to PyTorch SDPA, so the missing sm_120 kernels are
#    never invoked at runtime — the package only needs to import successfully.
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" MAX_JOBS=16 \
  pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3
#    Fallback if the 2.6.3 build still fails: bump to a Blackwell-aware release.
#    pip install --no-build-isolation flash-attn==2.7.4.post1

# 7. Build vendored CUDA extensions
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'

# 7a. depth_anything_3[gs] downgrades numpy 2.x -> 1.26 because its pyproject
#     pins numpy<2 (overly conservative — it imports cleanly with 2.x). Push
#     numpy back up to satisfy requirements.txt and rerun-sdk.
pip install 'numpy>=2.0'
```

Add the following to your shell profile (e.g. `~/.bashrc`) to persist `LD_LIBRARY_PATH` across sessions:

```bash
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

```bash
# 8. Verify installation
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('device cc:', torch.cuda.get_device_capability(0))
print('all imports OK')
"
# Expected on Blackwell: torch 2.7.1+cu128, cuda True, device cc (12, 0), all imports OK.

PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference --help
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon --help
```
