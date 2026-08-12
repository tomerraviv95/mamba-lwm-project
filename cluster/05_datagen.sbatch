#!/usr/bin/env bash
#SBATCH --job-name=lwm-datagen
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
# Generation is GPU-bound (pulse shaping + channel + STFT) and streams to disk shard-by-shard, so
# host RAM is dominated by the ray-traced PDP pool for all 20 cities x 3 BS (~442k links) plus one
# shard buffer (2000 x 98 KB = ~200 MB at --sc-channels 3). Measured peak locally: ~11 GB.
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/05_datagen_%j.out
# COMPUTE-NODE step — generate the corpus + both eval sets ON the cluster.
#
# Requires cluster/04_fetch_scenarios.sh to have run on the LOGIN node first (this node may have no
# internet, and a missing scenario would otherwise fail 40 minutes in).
#
#     export SC_CHANNELS=3
#     sbatch --chdir="$PWD" --output="$PWD/cluster/logs/05_datagen_%j.out" cluster/05_datagen.sbatch
#
# See the --output note in 10_pretrain_grid.sbatch: the relative default above is resolved against
# the job's working directory, which a site submit plugin can change, and a job whose log cannot be
# created dies with ExitCode 0:53 and no log at all. Always pass absolute --chdir/--output.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env
cd "$REPO_ROOT"
export SC_CHANNELS="${SC_CHANNELS:-1}"
PY="${PY:-uv run --no-sync python}"

echo "=== datagen: SC_CHANNELS=$SC_CHANNELS -> $(basename "$CORPUS_DIR") ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h "$REPO_ROOT" | tail -1

if [ "$SC_CHANNELS" = "3" ]; then
  RECIPE=spectro/datagen/regen_sc3_maxdiv.sh
else
  RECIPE=spectro/datagen/regen_sc_maxdiv.sh
fi
echo "recipe: $RECIPE"

# The recipe scripts hardcode CUDA_VISIBLE_DEVICES=0 and the local .venv python; on the cluster the
# interpreter comes from uv and Slurm has already bound the GPU, so override both.
PY="$PY" bash "$RECIPE"
rc=$?

echo
if [ "$rc" != 0 ]; then
  echo "DATAGEN FAILED (rc=$rc)"; exit "$rc"
fi
du -sh "$CORPUS_DIR" "$EVAL_INDIST_DIR" "$EVAL_XENV_DIR" 2>/dev/null
echo "DATAGEN OK — next: bash cluster/00_preflight.sh && bash cluster/run_study.sh"
