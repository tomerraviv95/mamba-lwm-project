#!/bin/bash
################################################################################
# Spectro MoE STEP-BASED real pretraining (spectro_pretrain_real.py) on the DeepMIMO-channel
# spectrogram corpus, with the LWM-Spectro PAPER recipe (Table I): MLM + supervised contrastive
# on mod & mobility, lambda_recon=1.0 / lambda_cont=0.3, tau=0.2, mask 70%, AdamW(wd=0.05),
# lr 5e-4 w/ warmup -> cosine to 1e-8. Trains each per-protocol expert for SPECTRO_STEPS steps and
# probes the REAL demo set every SPECTRO_EVAL_EVERY steps (logged to W&B + {proto}_steps.csv).
# Magnitude spectrograms (element_length=16); MLM-driven features transfer to the real demo tasks.
#
# One-time on the LOGIN node (internet):
#   uv run --no-sync python spectro/scripts/hf_sync.py pull-dataset --repo "$SPECTRO_DM_REPO" --dir "$SPECTRO_DM_DIR"
#   uv run --no-sync python spectro/scripts/download_spectro_hf.py --skip-weights   # demo + LWM source
#
# Run once per backbone (batch auto-set: transformer 8, mamba 32):
#   sbatch --export=ALL,ARCH=transformer cluster/01_pretrain_spectro.sbatch
#   sbatch --export=ALL,ARCH=mamba       cluster/01_pretrain_spectro.sbatch
# demo-acc-vs-steps -> spectro/outputs/pretrained_models/spectro_${ARCH}_weights/{LTE,WiFi,5G}_steps.csv
# and W&B (offline; sync from login node). Checkpoints pushed to HF at the end.
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
# Step-based real pretrainer probes the REAL demo set every --eval-every steps -> needs hf_cache
# (demo_data.pt + the HF LWM source for the transformer arm). Fetch once on the LOGIN node.
if [ ! -f "$REPO_ROOT/spectro/hf_cache/demo_data.pt" ] || \
   [ ! -f "$REPO_ROOT/spectro/hf_cache/pretraining/pretrained_model.py" ]; then
    echo "ERROR: need the HF mirror (spectro/hf_cache/). On the LOGIN node run:" >&2
    echo "  uv run --no-sync python spectro/scripts/download_spectro_hf.py --skip-weights" >&2
    exit 1
fi

ARCH="${ARCH:-mamba}"                       # transformer | mamba
PATCH="${PATCH:-${SPECTRO_PATCH:-4}}"       # 4 | 6 | 8 (baked into the weights dir name)
BATCH=$([ "$ARCH" = "transformer" ] && echo 8 || echo 32)   # transformer attention is memory-bound

echo "Spectro $ARCH MoE STEP-BASED real pretrain (paper recipe) patch=$PATCH on $SPECTRO_DM_DIR:"
echo "  steps=$SPECTRO_STEPS/expert  eval_every=$SPECTRO_EVAL_EVERY ($SPECTRO_EVAL_TASK)  batch=$BATCH"
echo "  w_mlm=$SPECTRO_W_MLM w_cont=$SPECTRO_W_MOD tau=$SPECTRO_TEMP mask=$MASK_PERCENT lr=$SPECTRO_LR wd=$SPECTRO_WD"
uv run --no-sync python spectro/scripts/spectro_pretrain_real.py \
    --arch "$ARCH" \
    --patch "$PATCH" \
    --pretrain-dir "$SPECTRO_DM_DIR" \
    --steps "$SPECTRO_STEPS" --eval-every "$SPECTRO_EVAL_EVERY" --eval-task "$SPECTRO_EVAL_TASK" \
    --router-epochs "$ROUTER_EPOCHS" \
    --n-layers "$N_LAYERS" \
    --batch-size "$BATCH" \
    --mask-percent "$MASK_PERCENT" \
    --w-mlm "$SPECTRO_W_MLM" --w-cont "$SPECTRO_W_MOD" --temperature "$SPECTRO_TEMP" \
    --lr "$SPECTRO_LR" --min-lr "$SPECTRO_MIN_LR" \
    --warmup-frac "$SPECTRO_WARMUP" --weight-decay "$SPECTRO_WD" \
    --seed "$SEED" \
    --wandb --wandb-project "$WANDB_PROJECT" --run-name "spectro-${ARCH}-p${PATCH}"

echo "Done. Weights + {LTE,WiFi,5G}_steps.csv (demo-acc-vs-steps) in spectro/outputs/pretrained_models/spectro_${ARCH}_p${PATCH}_weights."

# --- Save to Hugging Face (best effort here; the canonical step is on the LOGIN node) ----------
echo "Pushing checkpoints to HF ($HF_MODEL_REPO) ..."
if bash cluster/push_to_hf.sh ckpts; then
    echo "HF push OK."
else
    echo "HF push failed (compute node has no internet?). Run on the LOGIN node: bash cluster/push_to_hf.sh ckpts" >&2
fi
echo "W&B logged offline. Sync from the LOGIN node:  wandb sync $WANDB_DIR/wandb/offline-run-*"
