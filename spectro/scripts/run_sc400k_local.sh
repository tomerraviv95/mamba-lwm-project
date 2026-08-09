#!/usr/bin/env bash
# SCALE TEST: does a 3x larger pretraining corpus improve the downstream embedding?
#
# Design — compute-matched. Identical recipe to the 132k `sc2` run (patch 8, seed 1, 3000 steps,
# recon-only, batch 32 x accum 2, pos-embed std 0.02, [x,x^2] token embedding); the ONLY change is
# --pretrain-dir. So both runs see the same NUMBER of samples during pretraining (3000 steps x 64 =
# 192k) but the 400k run draws them from a 3x more diverse pool with far less repetition:
#     132k corpus -> ~1.4 epochs   |   400k corpus -> ~0.5 epochs
# A win here is therefore attributable to data DIVERSITY, not to extra compute — the cleanest form
# of the scale claim, and the one the SSL literature says should matter (published wireless SSL
# corpora are 1M-9.2M; the reference paper uses 9.2M against our 132k).
#
# Baselines are deliberately NOT weakened: frozen backbone + 150-epoch head + best-of-3 restarts
# stays. The reference paper fine-tunes its ImageNet baselines end-to-end for only 8 epochs, which
# collapses them few-shot (MobileNetV3-S = 7.12 macro-F1 @5/cls in its Table II). We beat a properly
# trained baseline or we do not claim a win.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0

CORPUS=spectro/outputs/spectro_corpus_sc400k_s1
EVAL_SEEN=spectro/outputs/spectro_eval_sc_indist_s1
EVAL_UNSEEN=spectro/outputs/spectro_eval_sc_heldout_s1
SUFFIX=sc400k
PATCH=8
SEED=1
STEPS=${STEPS:-3000}
TAG=sc400k_p8

[ -f "$CORPUS/manifest.json" ] || { echo "ERROR: $CORPUS missing — run the generator first"; exit 1; }
$PY -c "
import json;m=json.load(open('$CORPUS/manifest.json'))
print('corpus:', {k:m.get(k) for k in ('n_samples','draws','unique_users','waveform','sc_norm')})
assert m.get('waveform')=='sc' and m.get('sc_norm')=='global', 'wrong corpus recipe'
" || exit 1

echo "############ PRETRAIN 400k (patch $PATCH, seed $SEED, steps $STEPS, recon-only) ############"
for ARCH in mamba transformer; do
  D=spectro/outputs/pretrained_models/spectro_${ARCH}_p${PATCH}_${SUFFIX}_weights
  if [ -f "$D/router.pth" ] && [ "$(ls "$D"/*expert.pth 2>/dev/null | wc -l)" = "3" ]; then
    echo "--- $ARCH: already pretrained, skipping"; continue
  fi
  # ONE EXPERT PER INVOCATION. The corpus is loaded filtered to that protocol, so peak RAM is the
  # protocol's third (~4.4 GB) instead of the whole 398k corpus (13.5 GB). Costs ~2 min of extra
  # shard reading per expert; buys enough headroom to train on the FULL ~133k per expert instead of
  # a --max-per-expert truncation, which is the point of a scale test.
  for PROTO in LTE WiFi 5G; do
    echo "--- pretrain $ARCH / $PROTO"
    $PY spectro/scripts/spectro_pretrain_real.py \
        --arch "$ARCH" --patch "$PATCH" --seed "$SEED" --steps "$STEPS" \
        --pretrain-dir "$CORPUS" --weights-suffix "$SUFFIX" --protocols "$PROTO" \
        --w-mlm 1.0 --w-cont 0.0 --mask-percent 0.7 \
        --batch-size 32 --accum-steps 2 --early-stop-patience 0 \
        --eval-every 500 || { echo "PRETRAIN FAILED ($ARCH/$PROTO)"; exit 1; }
  done
  echo "--- router $ARCH"
  $PY spectro/scripts/spectro_pretrain_real.py \
      --arch "$ARCH" --patch "$PATCH" --seed "$SEED" --steps 1 \
      --pretrain-dir "$CORPUS" --weights-suffix "$SUFFIX" --router-only \
      --batch-size 32 || { echo "ROUTER FAILED ($ARCH)"; exit 1; }
done

echo "############ DOWNSTREAM ############"
# Only the arms whose features depend on the pretrained checkpoints need re-running; the CV / raw /
# deepcnn arms are corpus-independent and are reused from the sc2 run.
for EV in seen unseen; do
  [ "$EV" = seen ] && DIR=$EVAL_SEEN || DIR=$EVAL_UNSEEN
  for ARM in mamba transformer_synth; do
    echo "--- $ARM / $EV"
    $PY spectro/scripts/spectro_train_heads.py \
        --arm "$ARM" --head cnn1d --patch "$PATCH" --seed "$SEED" \
        --synth-dir "$DIR" --weights-suffix "$SUFFIX" --routing oracle \
        --run-tag "${TAG}_${EV}_s${SEED}" \
        --per-class-counts 2 5 10 20 50 100 --head-restarts 3 \
      || echo "ARM FAILED: $ARM/$EV"
  done
done
echo "ALL DONE"
