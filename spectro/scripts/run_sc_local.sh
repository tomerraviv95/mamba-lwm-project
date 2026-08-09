#!/usr/bin/env bash
# First local end-to-end validation of the single-carrier corpus: pretrain both MoE arches at
# patch 8 (seed 1), then run every downstream arm on both eval sets.
#
# Recipe notes:
#   --w-cont 0        recon-only pretraining, matching LWM-Spectro eq. (21). SupCon was previously
#                     folded into pretraining at 0.3 purely to compensate for a DEAD MLM objective
#                     (it sat at its trivial floor on the complex grid). On this corpus 4-neighbour
#                     interpolation cuts token MSE by 88-92%, so reconstruction carries real
#                     gradient and the crutch is unnecessary. It also removes a fair criticism:
#                     SupCon ran on the modulation/mobility labels, i.e. the downstream label sets.
#   --patch 8         seq 257 (vs 1025 at patch 4) -> ~6x cheaper, and the audit showed patch was
#                     never the cause of the modulation failure (a bigger patch is a BETTER moment
#                     estimate: one 8x8 patch carried 0.474 vs one 4x4 at 0.333).
#   --early-stop-patience 0   run the full step budget; the mean-pool probe understates per-token
#                     structure, so it is a poor stopping signal.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0

CORPUS=spectro/outputs/spectro_corpus_sc_s1
EVAL_SEEN=spectro/outputs/spectro_eval_sc_indist_s1
EVAL_UNSEEN=spectro/outputs/spectro_eval_sc_heldout_s1
SUFFIX=sc2
PATCH=8
SEED=1
STEPS=${STEPS:-3000}
TAG=sc2_p8

echo "############ PRETRAIN (patch $PATCH, seed $SEED, steps $STEPS, recon-only) ############"
for ARCH in mamba transformer; do
  D=spectro/outputs/pretrained_models/spectro_${ARCH}_p${PATCH}_${SUFFIX}_weights
  if [ -f "$D/router.pth" ] && [ "$(ls "$D"/*expert.pth 2>/dev/null | wc -l)" = "3" ]; then
    echo "--- $ARCH: already pretrained, skipping"; continue
  fi
  echo "--- pretrain $ARCH"
  $PY spectro/scripts/spectro_pretrain_real.py \
      --arch "$ARCH" --patch "$PATCH" --seed "$SEED" --steps "$STEPS" \
      --pretrain-dir "$CORPUS" --weights-suffix "$SUFFIX" \
      --w-mlm 1.0 --w-cont 0.0 --mask-percent 0.7 \
      --batch-size 32 --accum-steps 2 --early-stop-patience 0 \
      --eval-every 500 || { echo "PRETRAIN FAILED ($ARCH)"; exit 1; }
done

echo "############ DOWNSTREAM ############"
FROZEN="mamba transformer_synth random_init resnet18 mobilenet_v3_small raw"
E2E="deepcnn"
for EVAL_NAME in seen unseen; do
  [ "$EVAL_NAME" = seen ] && DIR=$EVAL_SEEN || DIR=$EVAL_UNSEEN
  for ARM in $FROZEN; do
    echo "--- $ARM / $EVAL_NAME"
    $PY spectro/scripts/spectro_train_heads.py \
        --arm "$ARM" --head cnn1d --patch "$PATCH" --seed "$SEED" \
        --synth-dir "$DIR" --weights-suffix "$SUFFIX" --moe-arch mamba \
        --routing oracle --run-tag "${TAG}_${EVAL_NAME}_s${SEED}" \
        --per-class-counts 2 5 10 20 50 100 --head-restarts 3 \
      || echo "ARM FAILED: $ARM/$EVAL_NAME"
  done
  echo "--- random_init(transformer) / $EVAL_NAME"
  $PY spectro/scripts/spectro_train_heads.py \
      --arm random_init --head cnn1d --patch "$PATCH" --seed "$SEED" \
      --synth-dir "$DIR" --weights-suffix "$SUFFIX" --moe-arch transformer \
      --routing oracle --run-tag "${TAG}_${EVAL_NAME}_s${SEED}_tfrand" \
      --per-class-counts 2 5 10 20 50 100 --head-restarts 3 \
    || echo "ARM FAILED: random_init_tf/$EVAL_NAME"
  for ARM in $E2E; do
    echo "--- $ARM / $EVAL_NAME (end-to-end)"
    $PY spectro/scripts/spectro_train_heads.py \
        --arm "$ARM" --head cnn1d --patch "$PATCH" --seed "$SEED" \
        --synth-dir "$DIR" --run-tag "${TAG}_${EVAL_NAME}_s${SEED}" \
        --per-class-counts 2 5 10 20 50 100 --head-restarts 3 --epochs 60 \
      || echo "ARM FAILED: $ARM/$EVAL_NAME"
  done
done
echo "ALL DONE"
