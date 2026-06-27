#!/bin/bash
################################################################################
# Downstream sample-variation sweep on the REAL (magnitude) demo tasks, comparing BOTH our
# DeepMIMO-contrastive-pretrained backbones against the published baseline + a floor. Arms:
#   transformer        - LWM-Spectro baseline (precomputed real-corpus embeddings)
#   transformer_synth  - OUR Transformer MoE   (loads spectro_transformer_weights from step 01)
#   mamba              - OUR Mamba MoE         (loads spectro_mamba_weights from step 01)
#   raw                - mean-pooled patches (floor)
# Produces spectro/outputs/plots/spectro_performance_vs_samples.png.
# Run AFTER both pretrain jobs (cluster/01_pretrain_spectro.sbatch ARCH=transformer & ARCH=mamba).
#
# Needs the real demo_data.pt + HF source (spectro/hf_cache/) — same download as step 1:
#     uv run --no-sync python spectro/scripts/download_spectro_hf.py --skip-weights   # login node, once
# Then:
#     sbatch cluster/02_downstream_spectro.sbatch
################################################################################
#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name spectro-sweep
#SBATCH --output cluster/logs/sweep-%J.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --constraint=rtx_3090
#SBATCH --mail-type=END,FAIL
# typed --gpus=rtx_3090:1 pins the GPU MODEL (the bare --constraint was ignored -> 1080 Ti).
# mamba-ssm/causal-conv1d are compiled for sm_86 only, so it MUST be rtx_3090.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$HOME/lwm-competition-2025}/cluster/config.env"

export PATH="$HOME/.local/bin:$PATH"     # uv
# No cuda module needed at RUNTIME (torch bundles its 12.8 runtime; avoids a module load conflict).
export PYTHONUNBUFFERED=TRUE
cd "$REPO_ROOT"

echo "GPU:"; nvidia-smi -L

if [ ! -f "$REPO_ROOT/spectro/hf_cache/demo_data.pt" ]; then
    echo "ERROR: demo_data.pt missing. On the LOGIN node run:" >&2
    echo "  uv run --no-sync python spectro/scripts/download_spectro_hf.py --skip-weights" >&2
    exit 1
fi

# Arms to run; override with: sbatch --export=ALL,ARMS="transformer mamba raw" ...
ARMS="${ARMS:-transformer transformer_synth mamba random_init raw}"
for arm in $ARMS; do
    echo "===== arm: $arm ====="
    uv run --no-sync python spectro/scripts/spectro_train_heads.py --arm "$arm" --seed "$SEED"
done

echo "Plotting accuracy vs #samples ..."
uv run --no-sync python spectro/scripts/spectro_plot_sample_variation.py
echo "Done -> spectro/outputs/plots/spectro_performance_vs_samples.png"
