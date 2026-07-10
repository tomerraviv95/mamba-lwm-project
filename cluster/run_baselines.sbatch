#!/usr/bin/env bash
#SBATCH --job-name=lwm-baselines
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/baselines_%j.out
# STEP 3 — run the non-LWM baselines under the SAME paper downstream protocol (macro-F1, samples/class)
# on BOTH eval sets, and publish the results to HF + W&B. Independent of any pretrained checkpoint.
#
#   sbatch --export=ALL cluster/run_baselines.sbatch
#
# Baselines (BASELINES env): deepcnn (end-to-end supervised), resnet18 (frozen ImageNet),
# random_init (untrained MoE — the pretraining-lift floor), raw (mean-pooled patches).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="${PY:-uv run --no-sync python}"

PATCH="${SPECTRO_PATCH}"
BASELINES="${BASELINES:-deepcnn resnet18 random_init raw}"
LOG="cluster/logs/baselines.log"
EVALS=("$EVAL_INDIST_DIR|alluser15" "$EVAL_XENV_DIR|heldoutcities")
PRIV=$([ "${HF_PRIVATE:-1}" = 1 ] && echo --private || echo "")
SUBS=()

echo "=== BASELINES [$BASELINES] p$PATCH START $(date) ===" | tee -a "$LOG"
for d in "$EVAL_INDIST_DIR" "$EVAL_XENV_DIR"; do
  [ -f "$d/manifest.json" ] || { echo "MISSING $d/manifest.json — run cluster/download_data.sh first." | tee -a "$LOG"; exit 1; }
done

for spec in "${EVALS[@]}"; do
  EDIR="${spec%%|*}"; ETAG="${spec##*|}"
  for arm in $BASELINES; do
    echo "===== $arm @ $ETAG $(date) =====" | tee -a "$LOG"
    $PY spectro/scripts/spectro_train_heads.py --arm "$arm" --patch "$PATCH" --head "$DOWNSTREAM_HEAD" \
      --synth-dir "$EDIR" --run-tag "$ETAG" \
      --per-class-counts $PER_CLASS_COUNTS --seeds $DOWNSTREAM_SEEDS --head-restarts "$HEAD_RESTARTS" 2>&1 | tee -a "$LOG" \
      && SUBS+=("spectro/outputs/submissions/submission_spectro_${arm}_p${PATCH}_heldout_${ETAG}")
  done
done

if [ "${#SUBS[@]}" -gt 0 ] && [ "${DO_UPLOAD:-1}" = 1 ]; then
  echo "===== UPLOAD RESULTS (${#SUBS[@]}) -> $HF_RESULTS_REPO + W&B $(date) =====" | tee -a "$LOG"
  $PY spectro/scripts/upload_results.py --submissions "${SUBS[@]}" \
    --hf-repo "$HF_RESULTS_REPO" $PRIV --wandb-project "$WANDB_PROJECT" 2>&1 | tee -a "$LOG" || true
elif [ "${#SUBS[@]}" -eq 0 ]; then
  echo "WARN: no baseline results produced" | tee -a "$LOG"
fi
[ "${WANDB_MODE:-offline}" = offline ] && echo "NOTE: W&B logged OFFLINE -> on the login node run: (cd $REPO_ROOT && uv run wandb sync cluster/logs/wandb/offline-*)" | tee -a "$LOG"
echo "=== BASELINES DONE $(date) ===" | tee -a "$LOG"
