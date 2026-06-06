#!/bin/bash
# One-time environment setup for the BGU-HPC spectro pipeline.
#
# RUN THIS ON THE LOGIN NODE (it has internet; pip needs it). A GPU is NOT required to
# *install* mamba-ssm — only to *run* it — so do NOT use a compute job for this.
#
#     ssh <bgu_user>@slurm.bgu.ac.il
#     cd ~/lwm-competition-2025
#     module load anaconda
#     module load cuda/12.4            # provides nvcc IF mamba-ssm source-builds
#     bash cluster/setup_env.sh
#     conda deactivate                 # submit jobs with the env DEACTIVATED
#
# The env is created in shared home (~/.conda/envs/$ENV_NAME), so every later sbatch
# job on any compute node sees it via `source activate $ENV_NAME`.
#
# Pins the stack validated locally: torch 2.10 (cu128), sionna 2.0.1, mamba-ssm 2.3.0.
# causal-conv1d is OPTIONAL (mamba-ssm runs without it; omitted to match the validated env).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"

# If mamba-ssm has no matching prebuilt wheel and source-builds, target the rtx_3090 arch
# (Ampere = 8.6). Harmless when a wheel is used. Add 8.9 if you also run on rtx_4090.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

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

echo "Installing mamba-ssm 2.3.0 (uses a wheel if available; else source-builds via nvcc) ..."
pip install "mamba-ssm==2.3.0"

echo
echo "Verifying CPU-safe imports on the login node (GPU is exercised later, in jobs) ..."
python - <<'PY'
import torch, sionna
print("torch", torch.__version__, "| sionna", sionna.__version__)
from sionna.phy.channel.tr38901 import TDL  # noqa  (CPU-safe import)
import importlib.util
print("mamba-ssm installed:", importlib.util.find_spec("mamba_ssm") is not None)
print("NOTE: mamba_ssm import + CUDA run is validated by the first GPU job (02_pretrain).")
print("-> setup OK")
PY

echo
echo "Done. Set your HF token (cluster/secrets.env) or run 'huggingface-cli login',"
echo "then 'conda deactivate' and submit jobs from the login node."
