#!/usr/bin/env bash
# LOCAL chained driver for the all-user (diverse) transformer run.
# Waits for the all-user corpus+eval gen (already running) to finish, then pretrains the Transformer
# MoE at patch 4 on the 85%-user corpus (weights-suffix alluser, 1 pretrain seed) with the SAME recipe
# as every other gridstft pretrain, and runs the downstream sweep on the 15%-user corpus. Detached +
# idempotent. GPU0 only. Mamba runs separately on the cluster (see run_alluser_pretrain.sh).
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG=cluster/logs/alluser_transformer_local.log
PY="${PY:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CORPUS=spectro/outputs/spectro_deepmimo_alluser85_gridstft
EVAL=spectro/outputs/spectro_eval_alluser15_gridstft
PATCH=4; SUF=alluser
BATCH="--batch-size 8 --accum-steps 4"
RECIPE="--steps 12000 --eval-every 2000 --eval-task modulation --n-layers 12 --router-epochs 15 --mask-percent 0.7 --w-mlm 1.0 --w-cont 0.3 --temperature 0.2 --lr 5e-4 --min-lr 1e-8 --warmup-frac 0.1 --weight-decay 0.05 --seed 42 --weights-suffix $SUF"
SWEEP="--sample-counts 50 100 250 500 1000 2500 4000 --seeds 42 43 44 --head-restarts 3 --project-dim 256"
done3(){ d="spectro/outputs/pretrained_models/spectro_transformer_p${PATCH}_${SUF}_weights"; [ "$(ls "$d" 2>/dev/null|grep -c expert.pth)" = 3 ] && [ -f "$d/router.pth" ]; }
{
  echo "=== ALL-USER TRANSFORMER (local) START $(date) ==="
  # wait for the corpus+eval gen (separate process) to finish
  waited=0
  while [ ! -f "$CORPUS/manifest.json" ] || [ ! -f "$EVAL/manifest.json" ]; do
    sleep 60; waited=$((waited+1))
    [ $((waited % 10)) -eq 0 ] && echo "   ...waiting for gen ($((waited)) min): corpus=$([ -f "$CORPUS/manifest.json" ] && echo Y || echo N) eval=$([ -f "$EVAL/manifest.json" ] && echo Y || echo N) $(date)"
    [ $waited -gt 240 ] && { echo "TIMEOUT waiting for gen (4h)"; exit 1; }
  done
  echo "### corpus+eval ready $(date)"
  if done3; then echo "## transformer p$PATCH $SUF already pretrained, skip"; else
    echo "===== PRETRAIN transformer p$PATCH $SUF $(date) ====="
    $PY spectro/scripts/spectro_pretrain_real.py --arch transformer --patch "$PATCH" \
      --pretrain-dir "$CORPUS" $BATCH $RECIPE || { echo "PRETRAIN FAILED"; exit 1; }
  fi
  echo "===== SWEEP transformer_synth p$PATCH $SUF $(date) ====="
  $PY spectro/scripts/spectro_train_heads.py --arm transformer_synth --patch "$PATCH" --pool meanstd_t \
    --synth-dir "$EVAL" --weights-suffix "$SUF" --seed 42 $SWEEP || echo "SWEEP FAILED"
  echo "=== ALL-USER TRANSFORMER (local) DONE $(date) ==="
} >> "$LOG" 2>&1
