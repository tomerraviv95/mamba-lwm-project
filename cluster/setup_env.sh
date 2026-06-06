#!/bin/bash
# One-time environment setup for the BGU-HPC spectro pipeline.
#
# RUN THIS ON A GPU NODE (mamba-ssm compiles/links CUDA kernels):
#     sinteractive --gpus=1 --constraint=rtx_3090
#     module load anaconda
#     module load cuda/12.4
#     bash cluster/setup_env.sh
#
# Pins the exact stack validated locally: torch 2.10 (cu128), sionna 2.0.1, mamba-ssm 2.3.0.
# causal-conv1d is OPTIONAL (mamba-ssm runs without it; we omit it to match the validated env).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"

echo "Creating conda env '$ENV_NAME' (python $PYTHON_VERSION) ..."
conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
# shellcheck disable=SC1091
source activate "$ENV_NAME"

echo "Installing PyTorch 2.10 (cu128) ..."
pip install --upgrade pip
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

echo "Installing scientific stack + Hugging Face Hub ..."
pip install "numpy>=2.0" "scipy>=1.13" "matplotlib>=3.7" "huggingface_hub>=1.0" tqdm scikit-learn

echo "Installing Sionna PHY 2.x ..."
pip install "sionna>=2.0"

echo "Installing mamba-ssm 2.3.0 (needs a GPU node + 'module load cuda/12.4' if it builds) ..."
pip install "mamba-ssm==2.3.0"
# If the above tries to build and fails, retry after: export MAMBA_FORCE_BUILD=TRUE
# (and ensure nvcc is on PATH via `module load cuda/12.4`).

echo
echo "Verifying imports on the allocated GPU ..."
python - <<'PY'
import torch, sionna, mamba_ssm
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
from sionna.phy.channel.tr38901 import TDL  # noqa
from mamba_ssm import Mamba  # noqa
print("sionna", sionna.__version__, "mamba_ssm", mamba_ssm.__version__, "-> OK")
PY

echo
echo "Setup complete. Next: set your HF token (cluster/secrets.env) or run 'huggingface-cli login'."
echo "Then submit jobs from the LOGIN node with the env DEACTIVATED:  conda deactivate"
