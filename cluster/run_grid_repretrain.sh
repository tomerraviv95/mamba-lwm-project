#!/usr/bin/env bash
# GRID re-pretrain pipeline (M8): regenerate corpus+eval in --repr grid (modulation-encoding
# representation), re-pretrain BOTH arches x patches 4/6/8 on grid (weights-suffix grid, so STFT
# checkpoints are preserved), run the in-domain grid sweep, and plot score-vs-patch. Detached +
# idempotent (skips completed gens/pretrains). Apples-to-apples recipe matches the STFT runs.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG=cluster/logs/m8_grid_repretrain.log
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CORPUS=spectro/outputs/spectro_deepmimo_mult8_vary_grid
EVAL=spectro/outputs/spectro_eval_heldout_cities_grid
RECIPE="--steps 12000 --eval-every 2000 --eval-task modulation --n-layers 12 --router-epochs 15 --mask-percent 0.7 --w-mlm 1.0 --w-cont 0.3 --temperature 0.2 --lr 5e-4 --min-lr 1e-8 --warmup-frac 0.1 --weight-decay 0.05 --seed 42 --weights-suffix grid"
done3(){ d="spectro/outputs/pretrained_models/spectro_$1_p$2_grid_weights"; [ "$(ls "$d" 2>/dev/null|grep -c expert.pth)" = 3 ] && [ -f "$d/router.pth" ]; }
{
  echo "=== GRID RE-PRETRAIN PIPELINE START $(date) ==="
  if [ ! -f "$CORPUS/manifest.json" ]; then
    echo "### gen grid pretrain corpus (20 cities) $(date)"
    $PY spectro/datagen/generate_deepmimo_spectro.py --out "$CORPUS" --symbol-mult 8 --vary-speed \
      --per-city 2000 --seed 42 --batch 4 --repr grid || { echo "CORPUS GEN FAILED"; exit 1; }
  fi
  if [ ! -f "$EVAL/manifest.json" ]; then
    echo "### gen grid held-out eval (nearest) $(date)"
    $PY spectro/datagen/generate_deepmimo_spectro.py --out "$EVAL" \
      --cities asu_campus_3p5:1,boston5g_3p5:2,o1_3p5:3 --symbol-mult 8 --vary-speed \
      --per-city 2000 --seed 1234 --batch 2 --repr grid || { echo "EVAL GEN FAILED"; exit 1; }
  fi
  for arch in mamba transformer; do
    if [ "$arch" = transformer ]; then BATCH="--batch-size 8 --accum-steps 4"; else BATCH="--batch-size 32 --accum-steps 1"; fi
    for P in 4 6 8; do
      if done3 "$arch" "$P"; then echo "## $arch p$P grid done, skip"; continue; fi
      echo "===== PRETRAIN $arch p$P grid $(date) ====="
      $PY spectro/scripts/spectro_pretrain_real.py --arch "$arch" --patch "$P" --pretrain-dir "$CORPUS" $BATCH $RECIPE \
        || echo "$arch p$P PRETRAIN FAILED"
    done
  done
  for P in 4 6 8; do
    for arm in transformer_synth mamba random_init raw; do
      echo "===== SWEEP $arm p$P grid $(date) ====="
      $PY spectro/scripts/spectro_train_heads.py --arm "$arm" --patch "$P" --pool meanstd_t \
        --synth-dir "$EVAL" --weights-suffix grid --seed 42 || echo "$arm p$P SWEEP FAILED"
    done
  done
  $PY spectro/scripts/spectro_plot_vs_patch.py --suffix _heldout_grid || echo "PLOT FAILED"
  echo "=== GRID RE-PRETRAIN PIPELINE DONE $(date) ==="
} >> "$LOG" 2>&1
