#!/usr/bin/env bash
#SBATCH --job-name=lwm-pretrain-ft
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=cluster/logs/01_pretrain_ft_%j.out
# COMPUTE-NODE step: pretrain ONE arch (reconstruction-only) then LoRA-finetune it on the downstream
# task (70/10/20 split, TOTAL-samples few-shot axis, 3 seeds), for every patch in PATCHES. Results are
# saved LOCALLY only; run `bash cluster/03_publish.sh` on the LOGIN node afterwards to upload to HF + W&B.
#
#   sbatch --export=ALL,ARCH=mamba       cluster/01_pretrain_finetune.sbatch
#   sbatch --export=ALL,ARCH=transformer cluster/01_pretrain_finetune.sbatch
#
# Requires: cluster/setup_env.sh run once, and cluster/download_data.sh run on the login node.
set -uo pipefail
# sbatch copies this script to a spool dir, so ${BASH_SOURCE[0]} isn't the repo. Slurm sets
# SLURM_SUBMIT_DIR to where sbatch was launched (submit from the repo root); fall back to PWD for `bash`.
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$ROOT/cluster/config.env" ] || { echo "ERROR: submit from the repo root — cluster/config.env not found under $ROOT"; exit 1; }
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="${PY:-uv run --no-sync python}"

ARCH="${ARCH:-mamba}"; SUF="${WEIGHTS_SUFFIX}"
[ "$ARCH" = transformer ] && { ARM=transformer_synth; BATCH="--batch-size $TF_BATCH --accum-steps $TF_ACCUM"; } \
                          || { ARM=mamba;             BATCH="--batch-size $MAMBA_BATCH --accum-steps $MAMBA_ACCUM"; }
LOG="cluster/logs/01_${ARCH}_${SUF}.log"
echo "=== 01 PRETRAIN + LoRA-FT ($ARM, patches=[$PATCHES], suffix=$SUF) START $(date) ===" | tee -a "$LOG"

# ---- data present? (pretrain corpus + the downstream dataset) -------------------------------
for d in "$CORPUS_DIR" "$DOWNSTREAM_DIR"; do
  [ -f "$d/manifest.json" ] || { echo "MISSING $d/manifest.json — run cluster/download_data.sh on the login node." | tee -a "$LOG"; exit 1; }
done

# ---- env preflight: fail in seconds (not after the ~8GB corpus load) if a dep is missing ----
if [ "$ARCH" = mamba ]; then
  $PY -c "import mamba_ssm" 2>/dev/null || {
    echo "ERROR: mamba_ssm not importable. Rebuild it: bash cluster/setup_env.sh (or uv pip install " \
         "--python .venv/bin/python --no-build-isolation mamba-ssm==2.3.0). A bare 'uv sync' REMOVES it " \
         "(not in the lock) — use 'uv sync --inexact'." | tee -a "$LOG"; exit 1; }
fi

done3() { [ "$(ls "$1" 2>/dev/null | grep -c expert.pth)" = 3 ] && [ -f "$1/router.pth" ]; }

for PATCH in $PATCHES; do
  WDIR="$CKPT_DIR/spectro_${ARCH}_p${PATCH}_${SUF}_weights"

  # 1) pretrain (reconstruction-only; contrastive is a fine-tuning-stage loss so W_CONT=0) unless done
  if done3 "$WDIR"; then
    echo "## $ARM p$PATCH already pretrained ($WDIR) — skip" | tee -a "$LOG"
  else
    echo "===== PRETRAIN $ARM p$PATCH $(date) =====" | tee -a "$LOG"
    $PY spectro/scripts/spectro_pretrain_real.py --arch "$ARCH" --patch "$PATCH" \
      --pretrain-dir "$CORPUS_DIR" $BATCH \
      --steps "$SPECTRO_STEPS" --eval-every "$SPECTRO_EVAL_EVERY" --eval-task "$SPECTRO_EVAL_TASK" \
      --n-layers "$N_LAYERS" --router-epochs "$ROUTER_EPOCHS" --mask-percent "$MASK_PERCENT" \
      --w-mlm "$SPECTRO_W_MLM" --w-cont "$SPECTRO_W_CONT" --temperature "$SPECTRO_TEMP" \
      --lr "$SPECTRO_LR" --min-lr "$SPECTRO_MIN_LR" --warmup-frac "$SPECTRO_WARMUP" \
      --weight-decay "$SPECTRO_WD" --seed "$SEED" --weights-suffix "$SUF" --probe-heldout-frac 0.1 \
      --wandb --wandb-project "$WANDB_PROJECT" --run-name "pretrain-${ARM}-${SUF}-p${PATCH}" 2>&1 | tee -a "$LOG"
    done3 "$WDIR" || { echo "PRETRAIN FAILED (no 3 experts + router in $WDIR)" | tee -a "$LOG"; exit 1; }
  fi

  # 2) LoRA-finetune on the downstream task (adapters <= LORA_BUDGET of backbone; head trainable)
  echo "===== LoRA-FT $ARM p$PATCH @ $(basename "$DOWNSTREAM_DIR") ($DOWNSTREAM_TAG) $(date) =====" | tee -a "$LOG"
  $PY spectro/scripts/spectro_finetune.py --arch "$ARCH" --patch "$PATCH" \
    --weights-suffix "$SUF" --synth-dir "$DOWNSTREAM_DIR" --run-tag "$DOWNSTREAM_TAG" \
    --sample-counts $SAMPLE_COUNTS --seeds $DOWNSTREAM_SEEDS \
    --val-frac "$VAL_FRAC" --test-frac "$TEST_FRAC" \
    --lora-budget "$LORA_BUDGET" --lora-max-rank "$LORA_MAX_RANK" \
    --epochs "$FT_EPOCHS" --patience "$FT_PATIENCE" --batch-size "$FT_BATCH" \
    --w-recon "$FT_W_RECON" --w-cont "$FT_W_CONT" --temperature "$SPECTRO_TEMP" --mask-percent "$MASK_PERCENT" 2>&1 | tee -a "$LOG"
done

echo "=== 01 DONE ($ARM) $(date). Publish on the LOGIN node: bash cluster/03_publish.sh ===" | tee -a "$LOG"
