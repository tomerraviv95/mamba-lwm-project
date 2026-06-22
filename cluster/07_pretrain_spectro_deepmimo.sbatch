#!/bin/bash
################################################################################
# Spectro MoE pretraining on the FIXED DeepMIMO-channel spectrogram dataset,
# multi-epoch (the validated recipe — fixed data avoids the streaming collapse).
#
# NOTE: this is a SIMPLIFIED variant of the LWM-Spectro authors' training:
#   - MLM-only (authors also use supervised contrastive on mod+mobility)
#   - magnitude spectrograms, element-16 (authors use complex, element-32)
# It gives a clean transformer-vs-mamba comparison + train/val loss curves.
#
# One-time on the LOGIN node (internet):
#   uv run --no-sync python spectro/scripts/hf_sync.py pull-dataset \
#       --repo "$SPECTRO_DM_REPO" --dir "$SPECTRO_DM_DIR"
#
# Run once per backbone (batch is auto-set: transformer=8 (O(seq^2) attention), mamba=32):
#   sbatch --export=ALL,ARCH=transformer cluster/07_pretrain_spectro_deepmimo.sbatch
#   sbatch --export=ALL,ARCH=mamba       cluster/07_pretrain_spectro_deepmimo.sbatch
# Per-epoch curves land in spectro/outputs/pretrained_models/spectro_${ARCH}_weights/{LTE,WiFi,5G}_losses.csv
################################################################################
#SBATCH --partition main
#SBATCH --time 1-12:00:00
#SBATCH --job-name spectro-dm
#SBATCH --output cluster/logs/spectro-dm-%J.out
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
if [ ! -f "$SPECTRO_DM_DIR/manifest.json" ]; then
    echo "ERROR: no dataset at $SPECTRO_DM_DIR. Pull on the LOGIN node:" >&2
    echo "  uv run --no-sync python spectro/scripts/hf_sync.py pull-dataset --repo $SPECTRO_DM_REPO --dir $SPECTRO_DM_DIR" >&2
    exit 1
fi

ARCH="${ARCH:-mamba}"                       # transformer | mamba
BATCH=$([ "$ARCH" = "transformer" ] && echo 8 || echo 32)   # transformer attention is memory-bound
# A 12-layer transformer from random init DIVERGES at lr=1e-3 (train loss climbs) even with warmup;
# 5e-4 + warmup descends cleanly. Mamba (SSM) is stable at 1e-3. Warmup itself is in the trainer.
LR=$([ "$ARCH" = "transformer" ] && echo 5e-4 || echo 1e-3)
echo "Spectro $ARCH MoE on FIXED DeepMIMO data ($SPECTRO_DM_EPOCHS epochs, batch $BATCH, lr $LR) ..."
uv run --no-sync python spectro/scripts/spectro_pretrain.py \
    --arch "$ARCH" \
    --data synthetic \
    --synthetic-dir "$SPECTRO_DM_DIR" \
    --epochs "$SPECTRO_DM_EPOCHS" \
    --router-epochs "$ROUTER_EPOCHS" \
    --n-layers "$N_LAYERS" \
    --mask-percent "$MASK_PERCENT" \
    --batch-size "$BATCH" \
    --lr "$LR" \
    --seed "$SEED"

echo "Done. Weights + {LTE,WiFi,5G}_losses.csv in spectro/outputs/pretrained_models/spectro_${ARCH}_weights."
echo "Publish: bash cluster/push_to_hf.sh ckpts   # on the LOGIN node"
