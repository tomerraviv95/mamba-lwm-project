#!/bin/bash
################################################################################
# Phase 1: generate the synthetic spectrogram corpus on a GPU (Sionna).
# Writes to $DATA_DIR (shared home). Publishing to HF is a separate login-node
# step (cluster/push_to_hf.sh) since compute nodes may have no internet.
#
# Submit from the LOGIN node (conda DEACTIVATED):
#     sbatch cluster/01_gen_data.sbatch
################################################################################
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name spectro-gen
#SBATCH --output cluster/logs/gen-%J.out
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
echo "Generating ~$((315 * PER_COMBO)) samples (per-combo=$PER_COMBO) -> $DATA_DIR"

# Generation uses the single allocated GPU as cuda:0 (Slurm masks the rest).
uv run --no-sync python spectro/datagen/generate.py \
    --per-combo "$PER_COMBO" \
    --batch "$GEN_BATCH" \
    --seed "$SEED" \
    --out "$DATA_DIR"

echo "Done. Corpus in $DATA_DIR (manifest.json + shards)."
echo "Publish it with:  bash cluster/push_to_hf.sh dataset   # run on the LOGIN node"
