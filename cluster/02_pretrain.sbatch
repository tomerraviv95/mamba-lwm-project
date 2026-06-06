#!/bin/bash
################################################################################
# Phase 2: pretrain the Mamba MoE on the synthetic corpus (masked-spectrogram).
# Reads $DATA_DIR, writes checkpoints to $CKPT_DIR/spectro_mamba_weights.
#
# Chain after generation:
#     gid=$(sbatch --parsable cluster/01_gen_data.sbatch)
#     sbatch --dependency=afterok:$gid cluster/02_pretrain.sbatch
################################################################################
#SBATCH --partition main
#SBATCH --time 1-00:00:00
#SBATCH --job-name spectro-pretrain
#SBATCH --output cluster/logs/pretrain-%J.out
#SBATCH --gpus=1
#SBATCH --constraint=rtx_3090
#SBATCH --mail-type=END,FAIL

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"

module load anaconda
source activate "$ENV_NAME"
module load cuda/12.4 || true
export PYTHONUNBUFFERED=TRUE

echo "GPU:"; nvidia-smi -L

# If the corpus isn't on disk (e.g. fresh node), pull it from HF (needs internet).
if [ ! -f "$DATA_DIR/manifest.json" ]; then
    echo "No local corpus; pulling $HF_DATASET_REPO -> $DATA_DIR"
    python "$REPO_ROOT/spectro/scripts/hf_sync.py" pull-dataset --repo "$HF_DATASET_REPO" --dir "$DATA_DIR"
fi

echo "Pretraining Mamba MoE (n_layers=$N_LAYERS, epochs=$PRETRAIN_EPOCHS) on synthetic ..."
python "$REPO_ROOT/spectro/scripts/spectro_pretrain.py" \
    --data synthetic \
    --synthetic-dir "$DATA_DIR" \
    --n-layers "$N_LAYERS" \
    --epochs "$PRETRAIN_EPOCHS" \
    --router-epochs "$ROUTER_EPOCHS" \
    --mask-percent "$MASK_PERCENT" \
    --batch-size "$PRETRAIN_BATCH" \
    --seed "$SEED"

echo "Done. Checkpoints in $CKPT_DIR/spectro_mamba_weights."
echo "Publish them with:  bash cluster/push_to_hf.sh ckpts   # run on the LOGIN node"
