#!/usr/bin/env bash
###############################################################################
# M3 LOCAL launcher: patch-parameterized pretrain + downstream, GPU0 only.
#
# The validated M1 recipe is baked in as defaults: downstream pooling is
# `meanstd_t` (mean ++ per-freq temporal std) and the pretrain corpus is
# expected to have been generated with `--symbol-mult 8 --vary-speed` (so
# mobility clears the raw floor). PATCH is persisted in every output name:
#   weights   -> spectro/outputs/pretrained_models/spectro_{arch}_p{PATCH}_weights/
#   downstream -> spectro/outputs/submissions/submission_spectro_{arm}_p{PATCH}/
#   plot      -> spectro/outputs/plots/spectro_performance_vs_samples_p{PATCH}.png
#
# Usage (env-overridable):
#   PATCH=6 bash spectro/scripts/run_patch_study.sh                 # full: pretrain+downstream, patch 6
#   PATCH=8 MODE=pretrain ARCHES=transformer bash .../run_patch_study.sh
#   PATCH=4 MODE=downstream bash .../run_patch_study.sh
#   PATCH=4 STEPS=200 MODE=pretrain ARCHES=transformer bash .../run_patch_study.sh   # smoke
###############################################################################
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PATCH="${PATCH:-4}"                       # 4 | 6 | 8
ARCHES="${ARCHES:-transformer mamba}"     # which backbones to pretrain
MODE="${MODE:-all}"                       # pretrain | downstream | all
STEPS="${STEPS:-12000}"                   # optimizer steps per expert
PRETRAIN_DIR="${PRETRAIN_DIR:-spectro/outputs/spectro_deepmimo_150k}"  # M1: gen w/ --symbol-mult 8 --vary-speed
POOL="${POOL:-meanstd_t}"                 # M1 recipe downstream pooling
SEED="${SEED:-42}"
ARMS="${ARMS:-transformer transformer_synth mamba random_init raw}"
export CUDA_VISIBLE_DEVICES=0             # GPU0 only (GPU1 is too slow)
PY="${PY:-.venv/bin/python}"
EVAL_EVERY=$(( STEPS / 6 > 0 ? STEPS / 6 : 1 ))

echo "M3 patch study: PATCH=$PATCH MODE=$MODE ARCHES='$ARCHES' STEPS=$STEPS POOL=$POOL SEED=$SEED (GPU0)"

if [[ "$MODE" == pretrain || "$MODE" == all ]]; then
    for arch in $ARCHES; do
        echo "===== pretrain $arch  patch=$PATCH ====="
        "$PY" spectro/scripts/spectro_pretrain_real.py --arch "$arch" --patch "$PATCH" \
            --pretrain-dir "$PRETRAIN_DIR" --steps "$STEPS" --eval-every "$EVAL_EVERY" \
            --eval-task modulation --seed "$SEED"
    done
fi

if [[ "$MODE" == downstream || "$MODE" == all ]]; then
    for arm in $ARMS; do
        echo "===== downstream $arm  patch=$PATCH pool=$POOL ====="
        "$PY" spectro/scripts/spectro_train_heads.py --arm "$arm" --patch "$PATCH" \
            --pool "$POOL" --seed "$SEED"
    done
    "$PY" spectro/scripts/spectro_plot_sample_variation.py --patch "$PATCH"
    echo "Plot -> spectro/outputs/plots/spectro_performance_vs_samples_p${PATCH}.png"
fi
