#!/usr/bin/env bash
# All-user (diverse) pretrain on the CLUSTER. Pulls the 85%-user corpus + 15%-user downstream eval
# from HF, pretrains one arch (default mamba) at patch 4 with the standard gridstft recipe + the
# in-corpus train/val downstream probe (--probe-heldout-frac), then runs the downstream sweep.
# Usage on a GPU compute node (see datascale_transformer.sbatch for the sbatch pattern):
#     ARCH=mamba bash cluster/run_alluser_pretrain.sh
# Env: ARCH (mamba|transformer, default mamba), HF_REPO, PY.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
ARCH="${ARCH:-mamba}"
LOG="cluster/logs/alluser_${ARCH}.log"
PY="${PY:-uv run python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
# xet transfer backend fails on restricted cluster networks (token-refresh over the xet CDN); force
# the classic HTTP download path. Best practice is still to pre-pull on the login node before sbatch.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_REPO="${HF_REPO:-tomerraviv95/lwm-spectro-alluser}"
CORPUS=spectro/outputs/spectro_deepmimo_alluser85_gridstft
EVAL=spectro/outputs/spectro_eval_alluser15_gridstft
PATCH=4; SUF=alluser
if [ "$ARCH" = transformer ]; then BATCH="--batch-size 8 --accum-steps 4"; else BATCH="--batch-size 32 --accum-steps 1"; fi
# Optional CPU-RAM safety valve: cap samples/expert (float16 build already cuts the peak ~4x; set
# MAX_PER_EXPERT=25000 etc. if the compute node still OOM-kills on the full ~44k/expert slice).
CAP=""; [ -n "${MAX_PER_EXPERT:-}" ] && CAP="--max-per-expert $MAX_PER_EXPERT"
RECIPE="--steps 12000 --eval-every 2000 --eval-task modulation --n-layers 12 --router-epochs 15 --mask-percent 0.7 --w-mlm 1.0 --w-cont 0.3 --temperature 0.2 --lr 5e-4 --min-lr 1e-8 --warmup-frac 0.1 --weight-decay 0.05 --seed 42 --weights-suffix $SUF --probe-heldout-frac 0.1 $CAP"
SWEEP="--sample-counts 50 100 250 500 1000 2500 4000 --seeds 42 43 44 --head-restarts 3 --project-dim 256"
done3(){ d="spectro/outputs/pretrained_models/spectro_${ARCH}_p${PATCH}_${SUF}_weights"; [ "$(ls "$d" 2>/dev/null|grep -c expert.pth)" = 3 ] && [ -f "$d/router.pth" ]; }
{
  echo "=== ALL-USER PRETRAIN ($ARCH p$PATCH) START $(date) ==="
  if [ ! -f "$CORPUS/manifest.json" ] || [ ! -f "$EVAL/manifest.json" ]; then
    echo "### pulling all-user corpus+eval from HF ($HF_REPO) $(date)"
    $PY spectro/scripts/hf_download_gridstft.py --repo "$HF_REPO" \
      --corpus-dir "$CORPUS" --eval-dir "$EVAL" || { echo "HF DOWNLOAD FAILED"; exit 1; }
  fi
  if done3; then echo "## $ARCH p$PATCH $SUF already pretrained, skip"; else
    echo "===== PRETRAIN $ARCH p$PATCH $SUF $(date) ====="
    $PY spectro/scripts/spectro_pretrain_real.py --arch "$ARCH" --patch "$PATCH" \
      --pretrain-dir "$CORPUS" $BATCH $RECIPE || { echo "PRETRAIN FAILED"; exit 1; }
  fi
  ARM=$([ "$ARCH" = transformer ] && echo transformer_synth || echo mamba)
  echo "===== SWEEP $ARM p$PATCH $SUF $(date) ====="
  $PY spectro/scripts/spectro_train_heads.py --arm "$ARM" --patch "$PATCH" --pool meanstd_t \
    --synth-dir "$EVAL" --weights-suffix "$SUF" --seed 42 $SWEEP || echo "SWEEP FAILED"
  echo "=== ALL-USER PRETRAIN ($ARCH p$PATCH) DONE $(date) ==="
} >> "$LOG" 2>&1
