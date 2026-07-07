#!/usr/bin/env bash
# LOCAL mamba run on the all-user (diverse) corpus. Pretrains WiMamba MoE at patch 4 on the 85%-user
# corpus (1 seed) + in-corpus train/val probe, then sweeps on the 15%-user downstream corpus.
# --max-per-expert caps each expert's slice so the masked-tensor build fits this box's 25 GB RAM
# (full corpus needs the cluster's larger RAM; the cap sits well past the ~13k/expert saturation
# point measured in the data-scaling ablation, so it costs no accuracy). GPU0 only.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG=cluster/logs/alluser_mamba_local.log
PY="${PY:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CORPUS=spectro/outputs/spectro_deepmimo_alluser85_gridstft
EVAL=spectro/outputs/spectro_eval_alluser15_gridstft
PATCH=4; SUF=alluser; CAP="${MAX_PER_EXPERT:-25000}"
RECIPE="--steps 12000 --eval-every 2000 --eval-task modulation --n-layers 12 --router-epochs 15 --mask-percent 0.7 --w-mlm 1.0 --w-cont 0.3 --temperature 0.2 --lr 5e-4 --min-lr 1e-8 --warmup-frac 0.1 --weight-decay 0.05 --seed 42 --weights-suffix $SUF --probe-heldout-frac 0.1 --max-per-expert $CAP"
SWEEP="--sample-counts 50 100 250 500 1000 2500 4000 --seeds 42 43 44 --head-restarts 3 --project-dim 256"
done3(){ d="spectro/outputs/pretrained_models/spectro_mamba_p${PATCH}_${SUF}_weights"; [ "$(ls "$d" 2>/dev/null|grep -c expert.pth)" = 3 ] && [ -f "$d/router.pth" ]; }
{
  echo "=== ALL-USER MAMBA (local) START $(date) cap/expert=$CAP ==="
  [ -f "$CORPUS/manifest.json" ] || { echo "MISSING CORPUS $CORPUS"; exit 1; }
  [ -f "$EVAL/manifest.json" ] || { echo "MISSING EVAL $EVAL"; exit 1; }
  if done3; then echo "## mamba p$PATCH $SUF already pretrained, skip"; else
    echo "===== PRETRAIN mamba p$PATCH $SUF $(date) ====="
    $PY spectro/scripts/spectro_pretrain_real.py --arch mamba --patch "$PATCH" \
      --pretrain-dir "$CORPUS" --batch-size 32 --accum-steps 1 $RECIPE || { echo "PRETRAIN FAILED"; exit 1; }
  fi
  echo "===== SWEEP mamba p$PATCH $SUF $(date) ====="
  $PY spectro/scripts/spectro_train_heads.py --arm mamba --patch "$PATCH" --pool meanstd_t \
    --synth-dir "$EVAL" --weights-suffix "$SUF" --seed 42 $SWEEP || echo "SWEEP FAILED"
  echo "=== ALL-USER MAMBA (local) DONE $(date) ==="
} >> "$LOG" 2>&1
