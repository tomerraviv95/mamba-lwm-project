#!/usr/bin/env bash
# Regenerate the full single-carrier (LWM-Spectro eq. 4) dataset family.
#
#   corpus   : 85% user split, 20 LWM cities        -> pretraining
#   indist   : 15% user split, SAME cities          -> downstream "seen" eval (user-disjoint)
#   heldout  : 3 unseen scenarios                   -> downstream "unseen" cross-environment eval
#
# The 85/15 user partition uses a FIXED split_seed=777 inside build_pdp_pool, independent of
# --seed, so corpus and indist are guaranteed disjoint at the user level across separate runs.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=spectro/outputs
GEN="$PY spectro/datagen/generate_deepmimo_spectro.py --waveform sc --vary-speed --batch 8"
export CUDA_VISIBLE_DEVICES=0

echo "=== [1/3] pretraining corpus (85% users, 20 cities) ==="
$GEN --all-users --user-split-frac 0.85 --user-split-part train \
     --shard-size 2000 --seed 1 --out "$OUT/spectro_corpus_sc_s1" || exit 1

echo "=== [2/3] downstream in-distribution eval (15% users, same cities) ==="
$GEN --all-users --user-split-frac 0.85 --user-split-part downstream \
     --shard-size 2000 --seed 2 --out "$OUT/spectro_eval_sc_indist_s1" || exit 1

echo "=== [3/3] downstream held-out-cities eval (cross-environment) ==="
$GEN --per-city 2000 --cities asu_campus_3p5:1,boston5g_3p5:2,o1_3p5:3 \
     --shard-size 2000 --seed 3 --out "$OUT/spectro_eval_sc_heldout_s1" || exit 1

echo "=== manifests ==="
for d in spectro_corpus_sc_s1 spectro_eval_sc_indist_s1 spectro_eval_sc_heldout_s1; do
  $PY -c "
import json;m=json.load(open('$OUT/$d/manifest.json'))
print('$d', {k:m.get(k) for k in ('n_samples','waveform','sc_win','sc_norm','freq_jitter','channels','seed','user_split_part')})"
done
echo "DONE"
