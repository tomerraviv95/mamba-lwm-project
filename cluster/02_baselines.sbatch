#!/usr/bin/env bash
#SBATCH --job-name=lwm-baselines
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/02_baselines_%j.out
# COMPUTE-NODE step: the NO-PRETRAIN baselines. Each trains on the 70% pool (val 10%, test 20%) with the
# same TOTAL-samples axis + 3 seeds as the LWM runs, for every patch in PATCHES. Results saved LOCALLY;
# run `bash cluster/03_publish.sh` on the LOGIN node afterwards to upload.
#
#   sbatch --export=ALL cluster/02_baselines.sbatch
#
# Arms (BASELINES in config.env): deepcnn (end-to-end CNN), random_init (untrained MoE, per moe-arch),
# raw (mean-pooled patches), resnet18 (frozen ImageNet features).
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

LOG="cluster/logs/02_baselines.log"
echo "=== 02 BASELINES (patches=[$PATCHES], arms=[$BASELINES], eval=$DOWNSTREAM_TAG) START $(date) ===" | tee -a "$LOG"
[ -f "$DOWNSTREAM_DIR/manifest.json" ] || { echo "MISSING $DOWNSTREAM_DIR/manifest.json — run cluster/download_data.sh on the login node." | tee -a "$LOG"; exit 1; }

run_head() {   # arm patch run_tag [extra flags...]
  local arm="$1" patch="$2" tag="$3"; shift 3
  $PY spectro/scripts/spectro_train_heads.py --arm "$arm" --patch "$patch" --head "$DOWNSTREAM_HEAD" \
    --synth-dir "$DOWNSTREAM_DIR" --run-tag "$tag" \
    --sample-counts $SAMPLE_COUNTS --seeds $DOWNSTREAM_SEEDS --head-restarts "$HEAD_RESTARTS" \
    --val-frac "$VAL_FRAC" --test-frac "$TEST_FRAC" "$@" 2>&1 | tee -a "$LOG"
}

for PATCH in $PATCHES; do
  for arm in $BASELINES; do
    if [ "$arm" = random_init ]; then
      # run the untrained-MoE control once per architecture so each is a control for its arm
      for MA in mamba transformer; do
        echo "===== baseline random_init/$MA p$PATCH $(date) =====" | tee -a "$LOG"
        run_head random_init "$PATCH" "${DOWNSTREAM_TAG}_${MA}" --moe-arch "$MA"
      done
    else
      echo "===== baseline $arm p$PATCH $(date) =====" | tee -a "$LOG"
      run_head "$arm" "$PATCH" "$DOWNSTREAM_TAG"
    fi
  done
done

echo "=== 02 DONE $(date). Publish on the LOGIN node: bash cluster/03_publish.sh ===" | tee -a "$LOG"
