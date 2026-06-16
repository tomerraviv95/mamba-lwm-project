#!/bin/bash
################################################################################
# Channel-domain pretraining: LWM (Transformer or Mamba) on a token-balanced
# DeepMIMO sample of the 20 LWM cities (~10M patch-tokens, ~116k realizations).
# Writes checkpoints to $CHANNEL_CKPT_DIR/{arch}_sampled/.
#
# PREREQUISITE: the DeepMIMO city scenarios must already be present under
# deepmimo_scenarios/ (DeepMIMO_data_gen does NOT auto-download). If missing,
# download them once on the LOGIN node (internet) before submitting.
#
# Architecture via ARCH env (default transformer). Run once per backbone:
#     sbatch --export=ALL,ARCH=transformer cluster/05_pretrain_channel.sbatch
#     sbatch --export=ALL,ARCH=mamba       cluster/05_pretrain_channel.sbatch
# Then publish:  bash cluster/push_to_hf.sh channel   # on the LOGIN node
################################################################################
#SBATCH --partition main
#SBATCH --time 2-00:00:00
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

ARCH="${ARCH:-transformer}"   # transformer | mamba
echo "Channel pretraining ($ARCH) on ~10M tokens (${TOKENS_PER_CITY}/city across 20 cities) ..."
uv run --no-sync python scripts/pretrain_channel_sampled.py \
    --arch "$ARCH" \
    --tokens-per-city "$TOKENS_PER_CITY" \
    --n-layers "$N_LAYERS" \
    --seed "$SEED"

echo "Done. Checkpoints in $CHANNEL_CKPT_DIR/${ARCH}_sampled."
echo "Publish them with:  bash cluster/push_to_hf.sh channel   # run on the LOGIN node"
