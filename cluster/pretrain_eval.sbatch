#!/usr/bin/env bash
#SBATCH --job-name=lwm-pretrain-eval
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=cluster/logs/pretrain_eval_%j.out
# STEP 2 — pretrain ONE arch (paper reconstruction-only), publish the checkpoints to HF, then run the
# paper downstream protocol (frozen "LWM" + fine-tuned "LWM FT", 1-D CNN head, macro-F1, samples/class)
# on BOTH eval sets and publish the results to HF + W&B.
#
#   sbatch --export=ALL,ARCH=mamba       cluster/pretrain_eval.sbatch
#   sbatch --export=ALL,ARCH=transformer cluster/pretrain_eval.sbatch
#
# Requires: cluster/setup_env.sh already run once, and cluster/download_data.sh run on the login node.
# Env knobs (see config.env): ARCH, WEIGHTS_SUFFIX, PER_CLASS_COUNTS, DOWNSTREAM_SEEDS, RUN_FROZEN,
# RUN_FT, FT_*, SPECTRO_STEPS, HF_MODEL_REPO, HF_RESULTS_REPO, WANDB_PROJECT.
set -uo pipefail
# sbatch COPIES this script to a spool dir, so ${BASH_SOURCE[0]} does NOT point at the repo. Slurm
# sets SLURM_SUBMIT_DIR to where `sbatch` was launched (submit from the repo root); fall back to PWD
# for a direct `bash cluster/pretrain_eval.sbatch` run.
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$ROOT/cluster/config.env" ] || { echo "ERROR: submit from the repo root — cluster/config.env not found under $ROOT"; exit 1; }
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="${PY:-uv run --no-sync python}"

ARCH="${ARCH:-mamba}"; PATCH="${SPECTRO_PATCH}"; SUF="${WEIGHTS_SUFFIX}"
[ "$ARCH" = transformer ] && { ARM=transformer_synth; BATCH="--batch-size $TF_BATCH --accum-steps $TF_ACCUM"; } \
                          || { ARM=mamba;             BATCH="--batch-size $MAMBA_BATCH --accum-steps $MAMBA_ACCUM"; }
WDIR="$CKPT_DIR/spectro_${ARCH}_p${PATCH}_${SUF}_weights"
LOG="cluster/logs/pretrain_eval_${ARCH}_${SUF}.log"
# eval sets: "dir|run-tag" (run-tag disambiguates the submission dirs so the two evals don't collide)
EVALS=("$EVAL_INDIST_DIR|alluser15" "$EVAL_XENV_DIR|heldoutcities")
SUBS=()

echo "=== PRETRAIN+EVAL ($ARM p$PATCH $SUF) START $(date) ===" | tee -a "$LOG"

# ---- 0. data present? (pretrain corpus + both evals) ---------------------------------------
for d in "$CORPUS_DIR" "$EVAL_INDIST_DIR" "$EVAL_XENV_DIR"; do
  [ -f "$d/manifest.json" ] || { echo "MISSING $d/manifest.json — run cluster/download_data.sh on the login node." | tee -a "$LOG"; exit 1; }
done

# ---- 1. pretrain (reconstruction-only; contrastive is a fine-tuning loss, W_CONT=0) --------
done3(){ [ "$(ls "$WDIR" 2>/dev/null | grep -c expert.pth)" = 3 ] && [ -f "$WDIR/router.pth" ]; }
if done3; then
  echo "## $ARM p$PATCH $SUF already pretrained ($WDIR) — skip" | tee -a "$LOG"
else
  echo "===== PRETRAIN $ARM $(date) =====" | tee -a "$LOG"
  $PY spectro/scripts/spectro_pretrain_real.py --arch "$ARCH" --patch "$PATCH" \
    --pretrain-dir "$CORPUS_DIR" $BATCH \
    --steps "$SPECTRO_STEPS" --eval-every "$SPECTRO_EVAL_EVERY" --eval-task "$SPECTRO_EVAL_TASK" \
    --n-layers "$N_LAYERS" --router-epochs "$ROUTER_EPOCHS" --mask-percent "$MASK_PERCENT" \
    --w-mlm "$SPECTRO_W_MLM" --w-cont "$SPECTRO_W_CONT" --temperature "$SPECTRO_TEMP" \
    --lr "$SPECTRO_LR" --min-lr "$SPECTRO_MIN_LR" --warmup-frac "$SPECTRO_WARMUP" \
    --weight-decay "$SPECTRO_WD" --seed "$SEED" --weights-suffix "$SUF" --probe-heldout-frac 0.1 \
    --wandb --wandb-project "$WANDB_PROJECT" --run-name "pretrain-${ARM}-${SUF}" 2>&1 | tee -a "$LOG"
  done3 || { echo "PRETRAIN FAILED (no 3 experts + router in $WDIR)" | tee -a "$LOG"; exit 1; }
fi

# ---- 2. publish checkpoints to HF ----------------------------------------------------------
PRIV=$([ "${HF_PRIVATE:-1}" = 1 ] && echo --private || echo "")
if [ "${DO_UPLOAD:-1}" = 1 ]; then
  echo "===== PUSH CKPTS -> $HF_MODEL_REPO $(date) =====" | tee -a "$LOG"
  $PY spectro/scripts/hf_sync.py push-ckpts --repo "$HF_MODEL_REPO" --dir "$WDIR" $PRIV 2>&1 | tee -a "$LOG" \
    || echo "WARN: ckpt push failed (run on login node: uv run python spectro/scripts/hf_sync.py push-ckpts --repo $HF_MODEL_REPO --dir $WDIR $PRIV)" | tee -a "$LOG"
fi

# ---- 3. downstream on both evals: frozen LWM (+ LWM FT) ------------------------------------
for spec in "${EVALS[@]}"; do
  EDIR="${spec%%|*}"; ETAG="${spec##*|}"
  if [ "${RUN_FROZEN}" = 1 ]; then
    echo "===== FROZEN $ARM @ $ETAG $(date) =====" | tee -a "$LOG"
    $PY spectro/scripts/spectro_train_heads.py --arm "$ARM" --patch "$PATCH" --head "$DOWNSTREAM_HEAD" \
      --weights-suffix "$SUF" --synth-dir "$EDIR" --run-tag "$ETAG" \
      --per-class-counts $PER_CLASS_COUNTS --seeds $DOWNSTREAM_SEEDS --head-restarts "$HEAD_RESTARTS" 2>&1 | tee -a "$LOG" \
      && SUBS+=("spectro/outputs/submissions/submission_spectro_${ARM}_p${PATCH}_heldout_${SUF}_${ETAG}")
  fi
  if [ "${RUN_FT}" = 1 ]; then
    echo "===== LWM-FT $ARM @ $ETAG $(date) =====" | tee -a "$LOG"
    $PY spectro/scripts/spectro_finetune.py --arch "$ARCH" --patch "$PATCH" \
      --weights-suffix "$SUF" --synth-dir "$EDIR" --run-tag "$ETAG" \
      --per-class-counts $FT_PER_CLASS_COUNTS --seeds $FT_SEEDS --epochs "$FT_EPOCHS" \
      --w-recon "$SPECTRO_W_MLM" --w-cont 0.3 --temperature "$SPECTRO_TEMP" --mask-percent "$MASK_PERCENT" 2>&1 | tee -a "$LOG" \
      && SUBS+=("spectro/outputs/submissions/submission_spectro_${ARM}_ft_p${PATCH}_heldout_${SUF}_${ETAG}")
  fi
done

# ---- 4. publish downstream results to HF + W&B --------------------------------------------
if [ "${#SUBS[@]}" -gt 0 ] && [ "${DO_UPLOAD:-1}" = 1 ]; then
  echo "===== UPLOAD RESULTS (${#SUBS[@]}) -> $HF_RESULTS_REPO + W&B $(date) =====" | tee -a "$LOG"
  $PY spectro/scripts/upload_results.py --submissions "${SUBS[@]}" \
    --hf-repo "$HF_RESULTS_REPO" $PRIV --wandb-project "$WANDB_PROJECT" 2>&1 | tee -a "$LOG" || true
elif [ "${#SUBS[@]}" -eq 0 ]; then
  echo "WARN: no downstream results produced (RUN_FROZEN=$RUN_FROZEN RUN_FT=$RUN_FT)" | tee -a "$LOG"
fi
[ "${WANDB_MODE:-offline}" = offline ] && echo "NOTE: W&B logged OFFLINE -> on the login node run: (cd $REPO_ROOT && uv run wandb sync cluster/logs/wandb/offline-*)" | tee -a "$LOG"
echo "=== PRETRAIN+EVAL ($ARM p$PATCH $SUF) DONE $(date) ===" | tee -a "$LOG"
