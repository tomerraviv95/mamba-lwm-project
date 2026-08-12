#!/usr/bin/env bash
# Maximum-diversity single-carrier corpus: ALL users x ALL BS positions in the 20 training cities.
#
# Why this and not simply "more draws": the 132k -> 398k scale test raised every downstream cell,
# but those three draws shared all 132,748 ray-traced geometries -- they varied labels, noise,
# symbols and sub-ray phases only. This run adds the axis that could not be tested there:
# each LWM city exposes THREE base stations (BS1/BS2/BS3), and only one was ever used, so ~2/3 of
# the available propagation geometry was being discarded. Pooling all three takes the pool from
# 156k to ~346k unique (user, BS) links before any draw multiplier.
#
# Split integrity: the 85/15 user partition is keyed on the RAW DeepMIMO grid index
# (generate_deepmimo_spectro.user_split_mask), NOT on each BS's filtered valid-user list. A UE with
# no link to BS1 may have one to BS3, so per-BS permutation would put the same physical location in
# train for one BS and eval for another. Verified: 0 user overlap across all three BSs.
# City-level split is unchanged -- the 20 LWM cities train, and asu_campus / boston5g / o1 stay
# entirely held out for the cross-environment eval.
set -uo pipefail
cd "$(dirname "$0")/../.."
# PY / CUDA_VISIBLE_DEVICES are overridable so cluster/05_datagen.sbatch can reuse this
# recipe verbatim (uv-provided interpreter, GPU already bound by Slurm).
PY="${PY:-.venv/bin/python}"
OUT=spectro/outputs
GEN="$PY spectro/datagen/generate_deepmimo_spectro.py --waveform sc --vary-speed --batch 8 --bs-list 1 2 3"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== [1/3] pretraining corpus (85% users x 3 BS x 20 cities, 3 draws) ==="
$GEN --all-users --draws 3 --user-split-frac 0.85 --user-split-part train \
     --shard-size 2000 --seed 1 --out "$OUT/spectro_corpus_scmax_s1" || exit 1

echo "=== [2/3] downstream in-distribution eval (15% users x 3 BS, same cities) ==="
$GEN --all-users --draws 1 --user-split-frac 0.85 --user-split-part downstream \
     --shard-size 2000 --seed 2 --out "$OUT/spectro_eval_scmax_indist_s1" || exit 1

echo "=== [3/3] downstream held-out-cities eval (cross-environment) ==="
# --bs-list is applied to every scenario; combinations that do not exist are skipped with a notice
# (asu_campus has only BS1, boston5g only BS2, o1 exposes TX 3..20).
$GEN --per-city 2000 --cities asu_campus_3p5:1,boston5g_3p5:2,o1_3p5:3 \
     --shard-size 2000 --seed 3 --out "$OUT/spectro_eval_scmax_heldout_s1" || exit 1

echo "=== manifests ==="
for d in spectro_corpus_scmax_s1 spectro_eval_scmax_indist_s1 spectro_eval_scmax_heldout_s1; do
  $PY -c "
import json;m=json.load(open('$OUT/$d/manifest.json'))
print('$d', {k:m.get(k) for k in ('n_samples','draws','unique_users','waveform','sc_norm','user_split_part')})"
done
echo "DONE"
