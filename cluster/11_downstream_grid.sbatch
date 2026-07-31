#!/usr/bin/env bash
#SBATCH --job-name=lwm-study-downstream
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=cluster/logs/11_downstream_%A_%a.out
# COMPUTE-NODE step (STUDY): for ONE (patch, seed) per array task, evaluate every arm on BOTH eval sets
# (seen=in-dist, unseen=held-out cities), then collate a single tidy CSV. Grid = (patch in PATCHES) x
# (seed in STUDY_SEEDS) = 2*5 = 10 tasks -> --array=0-9 (update if you change PATCHES / STUDY_SEEDS).
#
#   sbatch --export=ALL --dependency=afterok:<10_jobid> cluster/11_downstream_grid.sbatch
#
# Protocol: FROZEN backbone + trainable MLP head (meanstd_t pool) for mamba, transformer_synth, resnet18,
# raw, and random_init(mamba/transformer); DeepCNN trained END-TO-END. Requires the matching (patch, seed)
# checkpoints from 10_pretrain_grid.sbatch. Publish CSVs with cluster/12_publish_study.sh on the LOGIN node.
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

# ---- enumerate (patch, seed) and pick this array task ----
COMBOS=()
for PATCH in $PATCHES; do
  for SEED in $STUDY_SEEDS; do
    COMBOS+=("$PATCH $SEED")
  done
done
N=${#COMBOS[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "grid size = $N (set --array=0-$((N-1)))"
[ "$IDX" -ge "$N" ] && { echo "array index $IDX >= grid size $N — nothing to do, exiting"; exit 0; }
read -r PATCH SEED <<< "${COMBOS[$IDX]}"

SUF="${STUDY_SUFFIX}_s${SEED}"
LOG="cluster/logs/11_downstream_p${PATCH}_s${SEED}.log"
echo "=== 11 STUDY DOWNSTREAM idx=$IDX patch=$PATCH seed=$SEED START $(date) ===" | tee -a "$LOG"

for d in "$EVAL_INDIST_DIR" "$EVAL_XENV_DIR"; do
  [ -f "$d/manifest.json" ] || { echo "MISSING $d/manifest.json — run cluster/download_data.sh on the login node." | tee -a "$LOG"; exit 1; }
done

# spectro_train_heads.py wrapper: frozen backbone + trainable head (or e2e for deepcnn).
run_arm() {  # arm run_tag [extra flags...]
  local arm="$1" tag="$2"; shift 2
  $PY spectro/scripts/spectro_train_heads.py --arm "$arm" --patch "$PATCH" \
    --head "$STUDY_HEAD" --pool "$STUDY_POOL" --synth-dir "$EVAL_DIR" --run-tag "$tag" \
    --per-class-counts $STUDY_PER_CLASS --seed "$SEED" --head-restarts "$STUDY_HEAD_RESTARTS" \
    --val-frac "$VAL_FRAC" --test-frac "$TEST_FRAC" "$@" 2>&1 | tee -a "$LOG"
}

for EV in seen unseen; do
  [ "$EV" = seen ] && EVAL_DIR="$EVAL_INDIST_DIR" || EVAL_DIR="$EVAL_XENV_DIR"
  echo "----- eval=$EV ($(basename "$EVAL_DIR")) p$PATCH s$SEED $(date) -----" | tee -a "$LOG"
  # frozen backbone + MLP head arms
  for arm in $STUDY_FROZEN_ARMS; do
    if [ "$arm" = mamba ] || [ "$arm" = transformer_synth ]; then
      run_arm "$arm" "${EV}_s${SEED}" --weights-suffix "$SUF"
    else
      run_arm "$arm" "${EV}_s${SEED}"
    fi
  done
  # random-init controls, one per backbone arch
  run_arm random_init "${EV}_s${SEED}_mambarand" --moe-arch mamba
  run_arm random_init "${EV}_s${SEED}_tfrand"    --moe-arch transformer
  # end-to-end arm(s)
  for arm in $STUDY_E2E_ARMS; do
    run_arm "$arm" "${EV}_s${SEED}" --epochs "$STUDY_DEEPCNN_EPOCHS"
  done
done

echo "----- collate CSV p$PATCH s$SEED $(date) -----" | tee -a "$LOG"
$PY spectro/scripts/collate_csv.py --patch "$PATCH" --seed "$SEED" \
  --submissions "$REPO_ROOT/spectro/outputs/submissions" --out "$STUDY_CSV_DIR/study_csv" 2>&1 | tee -a "$LOG"

echo "=== 11 DONE idx=$IDX (p$PATCH s$SEED) $(date). Publish on the LOGIN node: bash cluster/12_publish_study.sh ===" | tee -a "$LOG"
