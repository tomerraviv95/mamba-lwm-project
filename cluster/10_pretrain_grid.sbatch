#!/usr/bin/env bash
#SBATCH --job-name=lwm-study-pretrain
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --output=cluster/logs/10_pretrain_%A_%a.out
# COMPUTE-NODE step (STUDY): pretrain ONE (arch, patch, seed) per array task, reconstruction-only.
# The grid is (arch in {mamba,transformer}) x (patch in PATCHES) x (seed in STUDY_SEEDS) = 2*2*5 = 20 tasks
# by default -> --array=0-19. If you change PATCHES / STUDY_SEEDS, update the --array range to match
# (the script prints the required size and exits cleanly for out-of-range indices).
#
#   sbatch --export=ALL cluster/10_pretrain_grid.sbatch          # submits the whole grid
#
# Checkpoints -> spectro_{arch}_p{patch}_{STUDY_SUFFIX}_s{seed}_weights (seed stamped in, so seeds never
# collide). Publish with cluster/12_publish_study.sh on the LOGIN node. Requires setup_env.sh + download_data.sh.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$ROOT/cluster/config.env" ] || { echo "ERROR: submit from the repo root — cluster/config.env not found under $ROOT"; exit 1; }
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="${PY:-uv run --no-sync python}"

# ---- enumerate the (arch, patch, seed) grid and pick this array task ----
COMBOS=()
for ARCH in mamba transformer; do
  for PATCH in $PATCHES; do
    for SEED in $STUDY_SEEDS; do
      COMBOS+=("$ARCH $PATCH $SEED")
    done
  done
done
N=${#COMBOS[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "grid size = $N (set --array=0-$((N-1)))"
[ "$IDX" -ge "$N" ] && { echo "array index $IDX >= grid size $N — nothing to do, exiting"; exit 0; }
read -r ARCH PATCH SEED <<< "${COMBOS[$IDX]}"

SUF="${STUDY_SUFFIX}_s${SEED}"
[ "$ARCH" = transformer ] && { ARM=transformer_synth; BATCH="--batch-size $TF_BATCH --accum-steps $TF_ACCUM"; } \
                          || { ARM=mamba;             BATCH="--batch-size $MAMBA_BATCH --accum-steps $MAMBA_ACCUM"; }
WDIR="$CKPT_DIR/spectro_${ARCH}_p${PATCH}_${SUF}_weights"
LOG="cluster/logs/10_${ARCH}_p${PATCH}_${SUF}.log"
echo "=== 10 STUDY PRETRAIN idx=$IDX arch=$ARCH patch=$PATCH seed=$SEED suffix=$SUF START $(date) ===" | tee -a "$LOG"

[ -f "$CORPUS_DIR/manifest.json" ] || { echo "MISSING $CORPUS_DIR/manifest.json — run cluster/download_data.sh on the login node." | tee -a "$LOG"; exit 1; }
if [ "$ARCH" = mamba ]; then
  $PY -c "import mamba_ssm" 2>/dev/null || {
    echo "ERROR: mamba_ssm not importable. Rebuild: bash cluster/setup_env.sh (bare 'uv sync' REMOVES it; use --inexact)." | tee -a "$LOG"; exit 1; }
fi

done3() { [ "$(ls "$1" 2>/dev/null | grep -c expert.pth)" = 3 ] && [ -f "$1/router.pth" ]; }
if done3 "$WDIR"; then
  echo "## $ARM p$PATCH seed$SEED already pretrained ($WDIR) — skip" | tee -a "$LOG"; exit 0
fi

$PY spectro/scripts/spectro_pretrain_real.py --arch "$ARCH" --patch "$PATCH" \
  --pretrain-dir "$CORPUS_DIR" $BATCH \
  --steps "$SPECTRO_STEPS" --eval-every "$SPECTRO_EVAL_EVERY" --eval-task "$SPECTRO_EVAL_TASK" \
  --early-stop-patience "$SPECTRO_EARLY_STOP_PATIENCE" --early-stop-metric "$SPECTRO_EARLY_STOP_METRIC" \
  --n-layers "$N_LAYERS" --router-epochs "$ROUTER_EPOCHS" --mask-percent "$MASK_PERCENT" \
  --w-mlm "$SPECTRO_W_MLM" --w-cont "$SPECTRO_W_CONT" --temperature "$SPECTRO_TEMP" \
  --lr "$SPECTRO_LR" --min-lr "$SPECTRO_MIN_LR" --warmup-frac "$SPECTRO_WARMUP" \
  --weight-decay "$SPECTRO_WD" --seed "$SEED" --weights-suffix "$SUF" --probe-heldout-frac 0.1 \
  --wandb --wandb-project "$WANDB_PROJECT" --run-name "study-pretrain-${ARM}-p${PATCH}-s${SEED}" 2>&1 | tee -a "$LOG"

done3 "$WDIR" || { echo "PRETRAIN FAILED (no 3 experts + router in $WDIR)" | tee -a "$LOG"; exit 1; }
echo "=== 10 DONE idx=$IDX ($ARM p$PATCH s$SEED) $(date) ===" | tee -a "$LOG"
