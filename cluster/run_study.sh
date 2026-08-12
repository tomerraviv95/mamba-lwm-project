#!/usr/bin/env bash
# LOGIN-NODE orchestrator (STUDY): submit the whole multi-seed / multi-patch grid in one shot.
#   1) 10_pretrain_grid  (array over arch x patch x seed)
#   2) 11_downstream_grid (array over patch x seed), dependency=afterANY on (1).
#      afterok would cancel EVERY downstream task if a single pretrain array task times out
#      (the transformer at patch 4 exceeds the 24h wall) -- including combos that finished.
#      The per-combo checkpoint guard in 11_* is the real gate.
# Then run cluster/12_publish_study.sh MANUALLY once both arrays finish (login-node network I/O).
#
#   bash cluster/run_study.sh
#
# Array ranges are computed from PATCHES / STUDY_SEEDS in config.env and passed via `sbatch --array`
# (overrides the #SBATCH defaults). Requires setup_env.sh + download_data.sh done first.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$ROOT/cluster/config.env" ] || { echo "ERROR: run from the repo root — cluster/config.env not found under $ROOT"; exit 1; }
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env

# preflight: data present?
for d in "$CORPUS_DIR" "$EVAL_INDIST_DIR" "$EVAL_XENV_DIR"; do
  [ -f "$d/manifest.json" ] || { echo "ERROR: MISSING $d/manifest.json — run 'bash cluster/download_data.sh' first."; exit 1; }
done

n_patch=$(echo $PATCHES | wc -w)
n_seed=$(echo $STUDY_SEEDS | wc -w)
N_PRE=$(( 2 * n_patch * n_seed ))      # arches (mamba, transformer) x patch x seed
N_DOWN=$(( n_patch * n_seed ))         # patch x seed
echo "grid: patches=[$PATCHES] seeds=[$STUDY_SEEDS] -> pretrain array 0-$((N_PRE-1)) ($N_PRE), downstream array 0-$((N_DOWN-1)) ($N_DOWN)"

# Slurm creates the --output file BEFORE the batch script runs. If cluster/logs/ is missing, not
# writable, or the filesystem is full, the job is assigned an ID and then dies at 00:00:00 with NO
# log at all -- it simply "disappears". Fail here, where the message is visible, instead.
mkdir -p cluster/logs
if ! touch cluster/logs/_submit_writetest 2>/dev/null; then
  echo "ERROR: cannot write cluster/logs/ -- Slurm will not be able to create job output files."
  echo "       Check free space and quota:  df -h .   /   quota -s"
  exit 1
fi
rm -f cluster/logs/_submit_writetest
_avail=$(df -Pk . | awk 'NR==2{print $4}')
if [ "${_avail:-0}" -lt 5242880 ]; then          # < 5 GB free
  echo "WARNING: only $((_avail/1024/1024)) GB free here. Jobs die at 00:00:00 with no log when the"
  echo "         filesystem is full. The HF cache (~/.cache/huggingface) is safe to delete --"
  echo "         hf_download_sc.py writes straight to the target and does not need it."
fi

JID_PRE=$(sbatch --parsable --export=ALL --array=0-$((N_PRE-1)) cluster/10_pretrain_grid.sbatch)
echo "submitted 10_pretrain_grid  -> job $JID_PRE"
JID_DOWN=$(sbatch --parsable --export=ALL --array=0-$((N_DOWN-1)) --dependency=afterany:"$JID_PRE" cluster/11_downstream_grid.sbatch)
echo "submitted 11_downstream_grid -> job $JID_DOWN (starts after $JID_PRE completes OK)"

cat <<EOF

Submitted. Watch with:  squeue --me
When BOTH arrays finish, publish on THIS (login) node:
    bash cluster/12_publish_study.sh
Then plot locally (your workstation):
    python spectro/scripts/plot_from_csv.py --hf-repo $HF_STUDY_REPO
EOF
