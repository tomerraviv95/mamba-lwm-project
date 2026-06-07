#!/bin/bash
# One-time environment setup for the BGU-HPC spectro pipeline, using uv.
#
# RUN THIS ON THE LOGIN NODE (it has internet; uv downloads packages). A GPU is NOT required to
# *install* mamba-ssm — only to *run* it — so do NOT use a compute job for this.
#
#     ssh <bgu_user>@slurm.bgu.ac.il
#     cd ~/lwm-competition-2025
#     bash cluster/setup_env.sh
#
# Creates a uv-managed venv at $REPO_ROOT/.venv (shared home), so every later sbatch job on any
# compute node uses it via `uv run --no-sync`. Pins the validated stack via uv.lock
# (torch 2.10 cu128, sionna 2.0.1, ...). mamba-ssm 2.3.0 is built separately with
# --no-build-isolation (its setup.py imports torch + has dynamic metadata, which breaks uv's
# isolated build); causal-conv1d is intentionally omitted (mamba-ssm runs without it).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"

# Bootstrap uv into ~/.local/bin if it isn't already available (no admin needed).
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

cd "$REPO_ROOT"
echo "uv sync -> $REPO_ROOT/.venv (from uv.lock; uv fetches a managed Python if needed) ..."
uv sync

# Build mamba-ssm against the just-synced torch. nvcc comes from the cuda module; with no GPU
# on the login node we must declare the target arch (rtx_3090 = 8.6).
# Use cuda/12.4 (same MAJOR as torch's bundled CUDA 12.8 -> minor mismatch is fine, just warns).
# Do NOT use cuda/13: a major-version mismatch (13 vs 12) makes PyTorch's extension build fail.
# (The module is only for building; torch's pip wheel bundles its own 12.8 runtime for execution.)
module load cuda/12.4 || true
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

# --- make the conda-forge host compiler available (no system g++ on this cluster) ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba-build
export CC=$(command -v x86_64-conda-linux-gnu-gcc)
export CXX=$(command -v x86_64-conda-linux-gnu-g++)
# nvcc needs to be told which host compiler to use
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
echo "Using CC=$CC"
echo "Using CXX=$CXX"

echo "Building mamba-ssm 2.3.0 (--no-build-isolation) ..."
uv pip install --no-build-isolation "mamba-ssm==2.3.0"

# --- Dr.Jit/Mitsuba LLVM backend so `import sionna` works on ANY node ---------------------
# Sionna's top-level import eagerly loads sionna.rt -> Mitsuba/Dr.Jit, which needs libLLVM.so
# even though this pipeline only uses sionna.phy. The GPU-less login node (and any CUDA-init
# hiccup in a job) needs the LLVM fallback. Install libLLVM into the conda build env and pin
# DRJIT_LIBLLVM_PATH to it (Dr.Jit accepts LLVM 14-19), persisting it for every sbatch job.
echo "Installing libLLVM (conda-forge) for Dr.Jit's LLVM backend ..."
# mamba-build is the currently-activated env (above), so $CONDA_PREFIX points at it.
conda install -n mamba-build -c conda-forge -y llvmdev >/dev/null
LLVM_LIB="$CONDA_PREFIX/lib/libLLVM.so"
if [ ! -e "$LLVM_LIB" ]; then
  LLVM_LIB="$(ls "$CONDA_PREFIX"/lib/libLLVM*.so* 2>/dev/null | head -n1 || true)"
fi
if [ -z "$LLVM_LIB" ] || [ ! -e "$LLVM_LIB" ]; then
  echo "WARNING: could not locate libLLVM.so after install; sionna import may fail on CPU nodes." >&2
else
  export DRJIT_LIBLLVM_PATH="$LLVM_LIB"
  echo "DRJIT_LIBLLVM_PATH=$DRJIT_LIBLLVM_PATH"
  # Persist for every sbatch job (config.env sources secrets.env). Idempotent: drop any prior
  # line first, then append the resolved absolute path.
  SECRETS="$HERE/secrets.env"
  touch "$SECRETS"
  grep -v '^export DRJIT_LIBLLVM_PATH=' "$SECRETS" > "$SECRETS.tmp" 2>/dev/null || true
  mv "$SECRETS.tmp" "$SECRETS"
  echo "export DRJIT_LIBLLVM_PATH=\"$DRJIT_LIBLLVM_PATH\"" >> "$SECRETS"
fi

echo
echo "Verifying CPU-safe imports (GPU is exercised later, in jobs) ..."
uv run --no-sync python - <<'PY'
import os, importlib.util, torch
print("DRJIT_LIBLLVM_PATH:", os.environ.get("DRJIT_LIBLLVM_PATH", "<unset>"))
import sionna  # triggers sionna.rt -> Mitsuba/Dr.Jit; needs the LLVM backend on a CPU node
print("torch", torch.__version__, "| sionna", sionna.__version__)
from sionna.phy.channel.tr38901 import TDL  # noqa  (the PHY bits the pipeline actually uses)
print("mamba-ssm installed:", importlib.util.find_spec("mamba_ssm") is not None)
print("NOTE: mamba_ssm import + CUDA run is validated by the first GPU job (02_pretrain).")
print("-> setup OK")
PY

echo
echo "Done. Set your HF token (cluster/secrets.env) or run 'uv run --no-sync huggingface-cli login'."
echo "Then submit GPU jobs with sbatch (they use 'uv run --no-sync', no network needed)."
