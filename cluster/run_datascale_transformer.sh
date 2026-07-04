#!/usr/bin/env bash
# DATA-SCALING ablation (transformer only): does more PRETRAIN data help?
# Re-pretrains the Transformer MoE at patch 4 on 10k and 20k subsamples of the existing 40k dual
# [STFT|grid] corpus (--max-samples), using the SAME recipe as the full 40k run so corpus size is the
# ONLY variable. Then runs the in-domain held-out sweep for each with the SAME downstream protocol
# (7 sample counts x 3 seeds x 3 head-restarts, project-dim 256) as the 40k baseline, so the new
# points drop straight onto the existing accuracy-vs-samples figure. Compare 10k/20k vs the existing
# 40k (submission_spectro_transformer_synth_p4_heldout_gridstft) to decide if more data is worth it.
# Detached + idempotent (skips a size whose 3 experts + router already exist). GPU0 only.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG=cluster/logs/datascale_transformer.log
PY="${PY:-uv run python}"   # cluster uses uv; override e.g. PY=.venv/bin/python locally
# respect SLURM's GPU allocation; default to 0 only for local runs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CORPUS=spectro/outputs/spectro_deepmimo_mult8_vary_gridstft
EVAL=spectro/outputs/spectro_eval_heldout_cities_gridstft
HF_REPO="${HF_REPO:-tomerraviv95/lwm-spectro-gridstft}"   # dual [STFT|grid] corpus+eval on HF
PATCH=4
SIZES="10000 20000"
# transformer recipe, identical to the 40k gridstft pretrain (eff batch 8*4=32, 12k steps)
BATCH="--batch-size 8 --accum-steps 4"
RECIPE="--steps 12000 --eval-every 2000 --eval-task modulation --n-layers 12 --router-epochs 15 --mask-percent 0.7 --w-mlm 1.0 --w-cont 0.3 --temperature 0.2 --lr 5e-4 --min-lr 1e-8 --warmup-frac 0.1 --weight-decay 0.05 --seed 42"
SWEEP="--sample-counts 50 100 250 500 1000 2500 4000 --seeds 42 43 44 --head-restarts 3 --project-dim 256"
done3(){ d="spectro/outputs/pretrained_models/spectro_transformer_p${PATCH}_$1_weights"; [ "$(ls "$d" 2>/dev/null|grep -c expert.pth)" = 3 ] && [ -f "$d/router.pth" ]; }
{
  echo "=== DATA-SCALING (transformer p$PATCH) START $(date) ==="
  # Prefer pulling the prebuilt corpus+eval from HF (fast, no DeepMIMO scenarios needed).
  if [ ! -f "$CORPUS/manifest.json" ] || [ ! -f "$EVAL/manifest.json" ]; then
    echo "### fetching dual corpus+eval from HF ($HF_REPO) $(date)"
    $PY spectro/scripts/hf_download_gridstft.py --repo "$HF_REPO" || echo "HF download failed; falling back to local gen"
  fi
  # Fallback: generate from DeepMIMO scenarios if still missing (same commands as run_gridstft_repretrain.sh).
  if [ ! -f "$CORPUS/manifest.json" ]; then
    echo "### gen grid_stft pretrain corpus (20 cities) $(date)"
    $PY spectro/datagen/generate_deepmimo_spectro.py --out "$CORPUS" --symbol-mult 8 --vary-speed \
      --per-city 2000 --seed 42 --batch 4 --repr grid_stft || { echo "CORPUS GEN FAILED"; exit 1; }
  fi
  if [ ! -f "$EVAL/manifest.json" ]; then
    echo "### gen grid_stft held-out eval (asu/boston/o1) $(date)"
    $PY spectro/datagen/generate_deepmimo_spectro.py --out "$EVAL" \
      --cities asu_campus_3p5:1,boston5g_3p5:2,o1_3p5:3 --symbol-mult 8 --vary-speed \
      --per-city 2000 --seed 1234 --batch 2 --repr grid_stft || { echo "EVAL GEN FAILED"; exit 1; }
  fi
  for N in $SIZES; do
    K=$((N/1000)); SUF="gridstft_${K}k"
    if done3 "$SUF"; then echo "## transformer p$PATCH $SUF already done, skip pretrain"; else
      echo "===== PRETRAIN transformer p$PATCH $SUF (corpus=$N) $(date) ====="
      $PY spectro/scripts/spectro_pretrain_real.py --arch transformer --patch "$PATCH" \
        --pretrain-dir "$CORPUS" --max-samples "$N" --weights-suffix "$SUF" $BATCH $RECIPE \
        || echo "transformer p$PATCH $SUF PRETRAIN FAILED"
    fi
    echo "===== SWEEP transformer_synth p$PATCH $SUF $(date) ====="
    $PY spectro/scripts/spectro_train_heads.py --arm transformer_synth --patch "$PATCH" --pool meanstd_t \
      --synth-dir "$EVAL" --weights-suffix "$SUF" --seed 42 $SWEEP \
      || echo "transformer_synth p$PATCH $SUF SWEEP FAILED"
  done
  echo "--- data-scaling summary: transformer p$PATCH modulation/snr/mobility @4000 ---"
  $PY - <<'PYEOF' || true
import json, os
base='spectro/outputs/submissions'
for suf in ['gridstft_10k','gridstft_20k','gridstft']:
    p=os.path.join(base,f'submission_spectro_transformer_synth_p4_heldout_{suf}','aggregated_results.json')
    if not os.path.exists(p): print(f'  {suf:14s} (missing)'); continue
    d=json.load(open(p)); row=[]
    for t in ['task_modulation','task_snr','task_mobility']:
        r=d['results_by_task'][t]['results']; k=[kk for kk in r if int(kk)==4000][0]
        row.append(f"{t.split('_')[1]}={r[k]['score']:.3f}")
    tag={'gridstft':'40k','gridstft_10k':'10k','gridstft_20k':'20k'}[suf]
    print(f'  {tag:4s}  '+'  '.join(row))
PYEOF
  echo "=== DATA-SCALING (transformer p$PATCH) DONE $(date) ==="
} >> "$LOG" 2>&1
