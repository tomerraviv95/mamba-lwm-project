#!/bin/bash
################################################################################
# Optional: before/after relevance experiment (Transformer + Mamba), proving the
# synthetic data teaches features useful for the real demo tasks. Writes the
# comparison plot + per-arm sweep results under spectro/outputs.
#
#     sbatch cluster/03_relevance.sbatch
################################################################################
#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name spectro-relevance
#SBATCH --output cluster/logs/relevance-%J.out
#SBATCH --gpus=1
#SBATCH --constraint=rtx_3090
#SBATCH --mail-type=END,FAIL

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HOME/lwm-competition-2025/cluster/config.env"

export PATH="$HOME/.local/bin:$PATH"     # uv
module load cuda/12.4 || true
export PYTHONUNBUFFERED=TRUE
cd "$REPO_ROOT"

echo "GPU:"; nvidia-smi -L
if [ ! -f "$DATA_DIR/manifest.json" ]; then
    echo "ERROR: no corpus at $DATA_DIR (run 01_gen_data first)." >&2; exit 1
fi

# Transformer attention is O(seq^2) at seq=1025 -> keep batches small (<=16).
uv run --no-sync python spectro/scripts/spectro_relevance_experiment.py \
    --n-layers "$N_LAYERS" \
    --pretrain-epochs "$PRETRAIN_EPOCHS" \
    --pretrain-batch "$PRETRAIN_BATCH" \
    --extract-batch 16 \
    --head-epochs 120 \
    --synthetic-dir "$DATA_DIR" \
    --seed "$SEED"

echo "Done. Plot: spectro/outputs/plots/spectro_relevance_before_after.png"
