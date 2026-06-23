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
# isolated build). causal-conv1d is built the same way (non-fatal): without it mamba_ssm falls
# back to a ~8x-slower unfused conv+scan path on our bidirectional 12-layer experts.
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

# Build the CUDA extensions against the .venv's torch (2.10.0+cu128). nvcc's MAJOR must match
# torch's CUDA major (12) — a CUDA-13 toolkit HARD-FAILS PyTorch's extension build. The cluster
# has a cuda/12.8 module (exact match for cu128); 12.4 also works (same major, just warns).
# We pin CUDA_HOME from the LOADED nvcc so a stale `export CUDA_HOME=...cuda-13.0` in your shell
# can't poison the build. (The toolkit is only for building; torch's wheel bundles its 12.8 runtime.)
module load cuda/12.8 || module load cuda/12.4 || true
if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
echo "Using CUDA_HOME=${CUDA_HOME:-<unset>}"; nvcc --version 2>/dev/null | tail -2 || true
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

# CRITICAL: target the project .venv explicitly (--python). conda 'mamba-build' is activated above
# only for its host compiler; without --python, `uv pip install` would install into the ACTIVE
# CONDA env (which may carry a different torch, e.g. cu130) instead of the .venv the jobs use,
# producing a CUDA-version mismatch at build time.
VENV_PY="$REPO_ROOT/.venv/bin/python"
echo "Building mamba-ssm 2.3.0 into the .venv (--no-build-isolation) ..."
uv pip install --python "$VENV_PY" --no-build-isolation "mamba-ssm==2.3.0"

# Build causal-conv1d (same toolchain). WITHOUT it, mamba_ssm.Mamba can't use its fully-fused
# mamba_inner_fn and falls back to an unfused conv+scan path — measured ~8x slower per epoch than
# the Transformer expert on our bidirectional 12-layer experts (seq=1025). With it, mamba is viable.
# Non-fatal: if the build fails the env still runs (just slow), so don't abort setup.
# IMPORTANT end-state contract: causal-conv1d must be EITHER fully working (the compiled
# `causal_conv1d_cuda` extension imports) OR completely absent. A half-install (python wrapper
# present, CUDA ext missing) makes mamba_ssm take its fused path and CRASH with
# "causal_conv1d_cuda is not available" — even the eager path then fails, since it also calls the
# wrapper. So: clear any prior install, FORCE a from-source build (no cached wheel), verify the
# CUDA ext actually imports, and if not, UNINSTALL it so mamba falls back to the safe eager conv.
echo "Building causal-conv1d 1.4.0 into the .venv (from source; enables fused fast mamba) ..."
uv pip uninstall --python "$VENV_PY" causal-conv1d >/dev/null 2>&1 || true
if CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --python "$VENV_PY" --no-build-isolation --no-cache "causal-conv1d==1.4.0" \
   && "$VENV_PY" -c "import causal_conv1d_cuda" >/dev/null 2>&1; then
    echo "causal-conv1d OK: causal_conv1d_cuda imports (FUSED fast mamba)."
else
    echo "WARNING: causal_conv1d_cuda unavailable; UNINSTALLING causal-conv1d so mamba uses the" >&2
    echo "         (slower but working) eager conv path instead of crashing on the fused path." >&2
    uv pip uninstall --python "$VENV_PY" causal-conv1d >/dev/null 2>&1 || true
fi

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
print("torch", torch.__version__, "| torch.cuda", torch.version.cuda,
      "| sionna", getattr(sionna, "__version__", "?"))
from sionna.phy.channel.tr38901 import TDL  # noqa  (the PHY bits the pipeline actually uses)
def _imp(m):
    try:
        importlib.import_module(m); return True
    except Exception:
        return False
print("mamba_ssm:", _imp("mamba_ssm"), "| selective_scan_cuda:", _imp("selective_scan_cuda"))
# What matters for mamba is the COMPILED ext, not just the python wrapper. A python-only
# causal_conv1d (no causal_conv1d_cuda) CRASHES mamba — setup removes it in that case.
_cce = _imp("causal_conv1d_cuda")
print("causal_conv1d_cuda:", _cce, "(FUSED fast mamba)" if _cce
      else "(absent -> mamba uses eager conv path; OK, just slower)")
if _imp("causal_conv1d") and not _cce:
    print("  !! BAD STATE: causal_conv1d python wrapper present but CUDA ext missing -> mamba will"
          " crash. Re-run setup_env.sh (it should have removed it).")
print("NOTE: mamba_ssm import + CUDA run is validated by the first GPU job (02_pretrain).")
print("-> setup OK")
PY

echo
echo "Done. Set your HF token (cluster/secrets.env) or run 'uv run --no-sync huggingface-cli login'."
echo "Then submit GPU jobs with sbatch (they use 'uv run --no-sync', no network needed)."
