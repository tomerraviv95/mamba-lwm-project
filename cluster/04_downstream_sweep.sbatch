#!/bin/bash
################################################################################
# Phase 4: downstream sample-variation sweep on the REAL demo tasks. Arms:
#   transformer        - LWM-Spectro baseline (precomputed real-corpus embeddings)
#   transformer_synth  - our Transformer MoE pretrained on synthetic   (needs ARCH=transformer pretrain)
#   mamba              - our Mamba MoE pretrained on synthetic          (needs ARCH=mamba pretrain)
#   raw                - mean-pooled patches (floor)
# Produces spectro/outputs/plots/spectro_performance_vs_samples.png.
#
# Needs the real demo_data.pt (with precomputed baseline embeddings). Download it
# ONCE on the login node (internet):
#     uv run --no-sync python spectro/scripts/download_spectro_hf.py --skip-weights
# Then:
#     sbatch cluster/04_downstream_sweep.sbatch
################################################################################
#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name spectro-sweep
#SBATCH --output cluster/logs/sweep-%J.out
#SBATCH --gpus=1
#SBATCH --constraint=rtx_3090
#SBATCH --mail-type=END,FAIL

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$HOME/lwm-competition-2025}/cluster/config.env"

export PATH="$HOME/.local/bin:$PATH"     # uv
module load cuda/12.4 || true
export PYTHONUNBUFFERED=TRUE
cd "$REPO_ROOT"

echo "GPU:"; nvidia-smi -L

if [ ! -f "$REPO_ROOT/spectro/hf_cache/demo_data.pt" ]; then
    echo "ERROR: demo_data.pt missing. On the LOGIN node run:" >&2
    echo "  uv run --no-sync python spectro/scripts/download_spectro_hf.py --skip-weights" >&2
    exit 1
fi

# Arms to run; override with: sbatch --export=ALL,ARMS="transformer mamba raw" ...
ARMS="${ARMS:-transformer transformer_synth mamba raw}"
for arm in $ARMS; do
    echo "===== arm: $arm ====="
    uv run --no-sync python spectro/scripts/spectro_train_heads.py --arm "$arm" --seed "$SEED"
done

echo "Plotting accuracy vs #samples ..."
uv run --no-sync python spectro/scripts/spectro_plot_sample_variation.py
echo "Done -> spectro/outputs/plots/spectro_performance_vs_samples.png"
