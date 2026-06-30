#!/usr/bin/env bash
# Chained in-domain eval: WAIT for the p4 transformer pretrain to finish (frees GPU0), then GENERATE a
# held-out-cities eval set (asu/boston/o1 — disjoint from the 20 pretrain cities, same recipe), then run
# the in-domain downstream sweep (frozen embedding -> small MLP head: train fits, val early-stops, test
# reported). Detached + idempotent so it survives session teardowns.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG=cluster/logs/m6_indomain_eval.log
EVAL=spectro/outputs/spectro_eval_heldout_cities
PY=.venv/bin/python
{
  echo "=== in-domain eval driver START $(date) ==="
  P4=spectro/outputs/pretrained_models/spectro_transformer_p4_weights
  echo "waiting for p4 transformer (3 experts + router) and a free GPU ..."
  while true; do
    n=$(ls "$P4" 2>/dev/null | grep -c expert.pth || echo 0); [ -f "$P4/router.pth" ] && r=1 || r=0
    if [ "$n" = "3" ] && [ "$r" = "1" ] && ! pgrep -f spectro_pretrain_real >/dev/null; then
      echo "p4 complete + GPU free $(date)"; break
    fi
    sleep 120
  done
  export CUDA_VISIBLE_DEVICES=0
  # 1. held-out-cities eval set (same mult8/vary recipe, new seed, disjoint cities)
  if [ ! -f "$EVAL/manifest.json" ]; then
    echo "generating held-out eval set $(date)"
    $PY spectro/datagen/generate_deepmimo_spectro.py --out "$EVAL" \
      --cities asu_campus_3p5:1,boston5g_3p5:2,o1_3p5:3 --symbol-mult 8 --vary-speed \
      --per-city 2000 --seed 1234 --batch 8 || { echo "GEN FAILED $(date)"; exit 1; }
  else
    echo "eval set already present, skipping gen"
  fi
  # 2. in-domain downstream sweep (frozen features + MLP head; train/val/test)
  for P in 4 6 8; do
    for arm in transformer_synth mamba random_init raw; do
      echo "===== in-domain $arm p$P $(date) ====="
      $PY spectro/scripts/spectro_train_heads.py --arm "$arm" --patch "$P" \
        --pool meanstd_t --synth-dir "$EVAL" --seed 42 || echo "$arm p$P FAILED"
    done
  done
  echo "=== IN-DOMAIN EVAL DONE $(date) ==="
} >> "$LOG" 2>&1
