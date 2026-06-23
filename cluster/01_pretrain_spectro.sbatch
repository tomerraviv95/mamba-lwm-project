#!/bin/bash
################################################################################
# Spectro MoE pretraining on the FIXED DeepMIMO-channel spectrogram dataset,
# multi-epoch (the validated recipe — fixed data avoids the streaming collapse),
# with the LWM-Spectro authors' FLAGSHIP objective: MLM + supervised contrastive
# on modulation & mobility (loss = 1*MLM + 50*SupCon(mod) + 30*SupCon(mob)),
# AdamW(wd=0.05), 5/20-epoch warmup -> cosine to 1e-5, 20 epochs.
#
# Deviation from the authors (deliberate, agreed): magnitude spectrograms
# (element_length=16, 4x4x1) instead of complex (element_length=32). Everything
# else — the two-stage per-protocol-expert + router structure, the contrastive
# loss/weights, optimizer & schedule — mirrors train_lwm_spectro_contrastive.py.
#
# One-time on the LOGIN node (internet):
#   uv run --no-sync python spectro/scripts/hf_sync.py pull-dataset \
#       --repo "$SPECTRO_DM_REPO" --dir "$SPECTRO_DM_DIR"
#
# Run once per backbone (batch/accum auto-set per arch; effective batch ~256 either way):
#   sbatch --export=ALL,ARCH=transformer cluster/01_pretrain_spectro.sbatch
#   sbatch --export=ALL,ARCH=mamba       cluster/01_pretrain_spectro.sbatch
# Per-epoch curves -> spectro/outputs/pretrained_models/spectro_${ARCH}_weights/{LTE,WiFi,5G}_losses.csv
# and to W&B (offline; sync command printed at the end). Checkpoints pushed to HF at the end.
################################################################################
#SBATCH --partition main
#SBATCH --time 1-12:00:00
#SBATCH --job-name spectro-pre
#SBATCH --output cluster/logs/spectro-pre-%J.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --constraint=rtx_3090
#SBATCH --mail-type=END,FAIL
# NOTE: typed --gpus=rtx_3090:1 (not bare --gpus=1) is what actually pins the GPU MODEL on this
# cluster — the node-feature --constraint alone was ignored and landed jobs on a GTX 1080 Ti
# (sm_61), which torch cu128 cannot run. Our mamba-ssm/causal-conv1d are compiled for sm_86 ONLY
# (TORCH_CUDA_ARCH_LIST=8.6), so it MUST be rtx_3090. If rtx_3090 is busy the job will PEND.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$HOME/lwm-competition-2025}/cluster/config.env"

export PATH="$HOME/.local/bin:$PATH"     # uv
# No cuda module needed at RUNTIME: torch's wheel bundles its 12.8 runtime and the compiled
# mamba-ssm/causal-conv1d link against it. (A `module load cuda` here only risks a load conflict.)
export PYTHONUNBUFFERED=TRUE
# W&B: compute nodes may have no internet -> log offline, sync from the login node afterwards.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$REPO_ROOT/cluster/logs"
cd "$REPO_ROOT"

echo "GPU:"; nvidia-smi -L
if [ ! -f "$SPECTRO_DM_DIR/manifest.json" ]; then
    echo "ERROR: no dataset at $SPECTRO_DM_DIR. Pull on the LOGIN node:" >&2
    echo "  uv run --no-sync python spectro/scripts/hf_sync.py pull-dataset --repo $SPECTRO_DM_REPO --dir $SPECTRO_DM_DIR" >&2
    exit 1
fi
# The transformer arm wraps the published LWM class from the HF mirror; fetch it once (login node).
if [ "${ARCH:-mamba}" = "transformer" ] && [ ! -f "$REPO_ROOT/spectro/hf_cache/pretraining/pretrained_model.py" ]; then
    echo "ERROR: transformer arm needs the HF LWM source (spectro/hf_cache/). On the LOGIN node run:" >&2
    echo "  uv run --no-sync python spectro/scripts/download_spectro_hf.py --skip-weights" >&2
    exit 1
fi

ARCH="${ARCH:-mamba}"                       # transformer | mamba
# Transformer attention is O(seq^2) at seq=1025 -> small physical batch; mamba is cheap.
# accum brings both to ~256 effective (authors used 64x4). Lower BATCH if the transformer OOMs.
if [ "$ARCH" = "transformer" ]; then BATCH=16; ACCUM=16; else BATCH=64; ACCUM=4; fi

echo "Spectro $ARCH MoE (contrastive) on FIXED DeepMIMO data:"
echo "  epochs=$SPECTRO_DM_EPOCHS  batch=${BATCH}x${ACCUM}  lr=$SPECTRO_LR  wd=$SPECTRO_WD  warmup=$SPECTRO_WARMUP"
uv run --no-sync python spectro/scripts/spectro_pretrain.py \
    --arch "$ARCH" \
    --data synthetic \
    --synthetic-dir "$SPECTRO_DM_DIR" \
    --contrastive \
    --w-mlm "$SPECTRO_W_MLM" --w-mod "$SPECTRO_W_MOD" --w-mob "$SPECTRO_W_MOB" \
    --epochs "$SPECTRO_DM_EPOCHS" \
    --router-epochs "$ROUTER_EPOCHS" \
    --n-layers "$N_LAYERS" \
    --mask-percent "$MASK_PERCENT" \
    --batch-size "$BATCH" \
    --accum-steps "$ACCUM" \
    --lr "$SPECTRO_LR" --min-lr "$SPECTRO_MIN_LR" \
    --warmup-frac "$SPECTRO_WARMUP" --weight-decay "$SPECTRO_WD" \
    --seed "$SEED" \
    --wandb --wandb-project "$WANDB_PROJECT" --run-name "spectro-${ARCH}"

echo "Done. Weights + {LTE,WiFi,5G}_losses.csv in spectro/outputs/pretrained_models/spectro_${ARCH}_weights."

# --- Save to Hugging Face (best effort here; the canonical step is on the LOGIN node) ----------
echo "Pushing checkpoints to HF ($HF_MODEL_REPO) ..."
if bash cluster/push_to_hf.sh ckpts; then
    echo "HF push OK."
else
    echo "HF push failed (compute node has no internet?). Run on the LOGIN node: bash cluster/push_to_hf.sh ckpts" >&2
fi
echo "W&B logged offline. Sync from the LOGIN node:  wandb sync $WANDB_DIR/wandb/offline-run-*"
