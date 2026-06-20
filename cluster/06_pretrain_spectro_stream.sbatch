#!/bin/bash
################################################################################
# Streaming spectro MoE pretraining on ~10M DeepMIMO-channel spectrograms generated
# ON THE FLY from the compact PDP pool (no 327 GB dataset). Needs Sionna (in the env)
# + the PDP pool; runs NO DeepMIMO.
#
# One-time, on the LOGIN node (internet):
#     uv run --no-sync python spectro/scripts/hf_sync.py pull-dataset \
#         --repo "$SPECTRO_PDP_REPO" --dir "$SPECTRO_PDP_DIR"
#
# Architecture via ARCH env (default mamba):
#     sbatch --export=ALL,ARCH=mamba       cluster/06_pretrain_spectro_stream.sbatch
#     sbatch --export=ALL,ARCH=transformer cluster/06_pretrain_spectro_stream.sbatch
################################################################################
#SBATCH --partition main
#SBATCH --time 2-00:00:00
#SBATCH --job-name spectro-stream
#SBATCH --output cluster/logs/spectro-stream-%J.out
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

if [ ! -f "$SPECTRO_PDP_DIR/pdp_pool.pt" ]; then
    echo "ERROR: no PDP pool at $SPECTRO_PDP_DIR. Pull it on the LOGIN node:" >&2
    echo "  uv run --no-sync python spectro/scripts/hf_sync.py pull-dataset --repo $SPECTRO_PDP_REPO --dir $SPECTRO_PDP_DIR" >&2
    exit 1
fi

ARCH="${ARCH:-mamba}"   # mamba | transformer
echo "Streaming ~10M DeepMIMO-channel spectrograms ($ARCH) from PDP pool ..."
uv run --no-sync python spectro/scripts/spectro_pretrain_stream.py \
    --arch "$ARCH" \
    --pdp-pool "$SPECTRO_PDP_DIR" \
    --target-samples 10000000 \
    --n-layers "$N_LAYERS" \
    --mask-percent "$MASK_PERCENT" \
    --batch 32 \
    --seed "$SEED"

echo "Done. Streaming MoE weights in spectro/outputs/pretrained_models/spectro_${ARCH}_stream_weights."
