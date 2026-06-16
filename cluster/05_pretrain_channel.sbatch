#!/bin/bash
################################################################################
# Channel-domain pretraining: LWM (Transformer or Mamba) on the token-balanced
# DeepMIMO channel dataset (~10M patch-tokens across the 20 LWM cities).
#
# The dataset is generated OFF-CLUSTER and downloaded — the cluster runs NO DeepMIMO.
# One-time, on the LOGIN node (internet):
#     uv run --no-sync python spectro/scripts/hf_sync.py pull-dataset \
#         --repo "$HF_CHANNEL_DATASET_REPO" --dir "$CHANNEL_DATA_DIR"
#
# Architecture via ARCH env (default transformer). Run once per backbone:
#     sbatch --export=ALL,ARCH=transformer cluster/05_pretrain_channel.sbatch
#     sbatch --export=ALL,ARCH=mamba       cluster/05_pretrain_channel.sbatch
# Then publish:  bash cluster/push_to_hf.sh channel   # on the LOGIN node
################################################################################
#SBATCH --partition main
#SBATCH --time 1-12:00:00
#SBATCH --job-name channel-pretrain
#SBATCH --output cluster/logs/channel-%J.out
#SBATCH --gpus=1
#SBATCH --constraint=rtx_3090
#SBATCH --mem=64G
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

if [ ! -f "$CHANNEL_DATA_DIR/manifest.json" ]; then
    echo "ERROR: no channel dataset at $CHANNEL_DATA_DIR." >&2
    echo "Pull it on the LOGIN node:" >&2
    echo "  uv run --no-sync python spectro/scripts/hf_sync.py pull-dataset --repo $HF_CHANNEL_DATASET_REPO --dir $CHANNEL_DATA_DIR" >&2
    exit 1
fi

ARCH="${ARCH:-transformer}"   # transformer | mamba
echo "Channel pretraining ($ARCH) from $CHANNEL_DATA_DIR ..."
uv run --no-sync python scripts/pretrain_channel_sampled.py \
    --arch "$ARCH" \
    --dataset-dir "$CHANNEL_DATA_DIR" \
    --n-layers "$N_LAYERS" \
    --seed "$SEED"

echo "Done. Checkpoints in $CHANNEL_CKPT_DIR/${ARCH}_sampled."
echo "Publish them with:  bash cluster/push_to_hf.sh channel   # run on the LOGIN node"
