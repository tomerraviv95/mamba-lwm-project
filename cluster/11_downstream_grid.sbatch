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
FAILED_ARMS=""
run_arm() {  # arm run_tag restarts [extra flags...]
  local arm="$1" tag="$2" restarts="$3"; shift 3
  $PY spectro/scripts/spectro_train_heads.py --arm "$arm" --patch "$PATCH" \
    --head "$STUDY_HEAD" --pool "$STUDY_POOL" --synth-dir "$EVAL_DIR" --run-tag "$tag" \
    --per-class-counts $STUDY_PER_CLASS --seed "$SEED" --head-restarts "$restarts" \
    --val-frac "$VAL_FRAC" --test-frac "$TEST_FRAC" "$@" 2>&1 | tee -a "$LOG"
  # PIPESTATUS[0] = python's status; $? is tee's. Without this a traceback is invisible: the loop
  # continues, collate emits a SHORT csv, and the job still exits 0 (a missing arm then looks
  # identical to an arm that never existed).
  local rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && FAILED_ARMS="$FAILED_ARMS $arm/$tag(rc=$rc)"
  return 0
}

HR="$STUDY_HEAD_RESTARTS"; HR_E2E="${STUDY_E2E_HEAD_RESTARTS:-1}"; H="${STUDY_VARIANT:-$STUDY_HEAD}"   # namespace stamped into run-tags/CSV (head itself = $STUDY_HEAD inside run_arm)
for EV in seen unseen; do
  [ "$EV" = seen ] && EVAL_DIR="$EVAL_INDIST_DIR" || EVAL_DIR="$EVAL_XENV_DIR"
  echo "----- head=$H eval=$EV ($(basename "$EVAL_DIR")) p$PATCH s$SEED $(date) -----" | tee -a "$LOG"
  # frozen backbone + shared head (best-of-HR head inits)
  # --project-dim equalizes the head across arms. Conv1dHead's proj is Conv1d(d_model,128,k=1), so
  # its parameter count scales with whatever width the backbone hands over: MEASURED 347k params for
  # MobileNet (576-d) and 331k for ResNet-18 (512-d) vs 232k for the 128-d LWM arms -- a ~1.5x
  # capacity advantage unrelated to representation quality. Projecting every arm to 128-d makes all
  # heads exactly 232,325 params. (Widths already <=128, e.g. raw at 64-d, pass through unchanged.)
  for arm in $STUDY_FROZEN_ARMS; do
    if [ "$arm" = mamba ] || [ "$arm" = transformer_synth ]; then
      run_arm "$arm" "${H}_${EV}_s${SEED}" "$HR" --weights-suffix "$SUF" --routing "$STUDY_ROUTING"
    else
      run_arm "$arm" "${H}_${EV}_s${SEED}" "$HR" --project-dim "$STUDY_PROJECT_DIM"
    fi
  done
  # Random-init controls, one per backbone arch. These MUST use the same --routing as the pretrained
  # arms: random_init was previously hardcoded to oracle while the pretrained arms used the learned
  # router, so the control was handed ground-truth protocol labels the treatment never got.
  run_arm random_init "${H}_${EV}_s${SEED}_mambarand" "$HR" --moe-arch mamba       --routing "$STUDY_ROUTING"
  run_arm random_init "${H}_${EV}_s${SEED}_tfrand"    "$HR" --moe-arch transformer --routing "$STUDY_ROUTING"
  # end-to-end arm(s): full-backbone train per restart -> keep restarts low
  for arm in $STUDY_E2E_ARMS; do
    run_arm "$arm" "${H}_${EV}_s${SEED}" "$HR_E2E" --epochs "$STUDY_DEEPCNN_EPOCHS"
  done
done

echo "----- collate CSV (variant=$H) p$PATCH s$SEED $(date) -----" | tee -a "$LOG"
$PY spectro/scripts/collate_csv.py --patch "$PATCH" --seed "$SEED" --variant "$H" \
  --submissions "$REPO_ROOT/spectro/outputs/submissions" --out "$STUDY_CSV_DIR/study_csv_${H}" 2>&1 | tee -a "$LOG"

if [ -n "$FAILED_ARMS" ]; then
  echo "=== 11 FAILED idx=$IDX (p$PATCH s$SEED): arms ->$FAILED_ARMS ===" | tee -a "$LOG"
  exit 1
fi
echo "=== 11 DONE idx=$IDX (p$PATCH s$SEED) $(date). Publish on the LOGIN node: bash cluster/12_publish_study.sh ===" | tee -a "$LOG"
