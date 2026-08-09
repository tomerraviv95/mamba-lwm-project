#!/usr/bin/env bash
# Retry cluster/download_data.sh until every shard is present, or --max-attempts is exhausted.
#
# Safe to loop because every fetch is RESUMABLE: hf_download_sc.py skips files whose local size
# already matches the remote, so each pass only pulls what is still missing and a failed attempt
# costs nothing but the retry delay. download_data.sh exits non-zero when any dataset is
# incomplete, which is what drives the loop.
#
#   bash cluster/fetch_until_done.sh                    # defaults: 40 attempts, 60s backoff
#   MAX_ATTEMPTS=100 SLEEP_S=120 bash cluster/fetch_until_done.sh
#   nohup bash cluster/fetch_until_done.sh > fetch.log 2>&1 &     # unattended
#
# Progress is reported per attempt as shards-present/shards-in-manifest, so a run that is
# advancing slowly is distinguishable from one that is stuck.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env >/dev/null 2>&1
cd "$REPO_ROOT"
PY="${PY:-uv run --no-sync python}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
SLEEP_S="${SLEEP_S:-60}"

progress() {
  for d in "$CORPUS_DIR" "$EVAL_INDIST_DIR" "$EVAL_XENV_DIR"; do
    $PY - "$d" <<'PY' 2>/dev/null || echo "    $(basename "$d"): no manifest yet"
import json, os, sys
d = sys.argv[1]; m = os.path.join(d, 'manifest.json')
if not os.path.isfile(m):
    print(f"    {os.path.basename(d)}: no manifest yet"); sys.exit(0)
man = json.load(open(m)); sh = man['shards']
have = sum(1 for s in sh if os.path.isfile(os.path.join(d, s)))
gb = sum(os.path.getsize(os.path.join(d, s)) for s in sh
         if os.path.isfile(os.path.join(d, s))) / 1e9
print(f"    {os.path.basename(d)}: {have}/{len(sh)} shards ({gb:.1f} GB) n={man.get('n_samples')}")
PY
  done
}

for i in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=========== fetch attempt $i/$MAX_ATTEMPTS  $(date) ==========="
  if bash cluster/download_data.sh; then
    echo "=== ALL DATA PRESENT after $i attempt(s) $(date) ==="
    progress
    exit 0
  fi
  echo "--- attempt $i incomplete; progress so far:"
  progress
  echo "--- retrying in ${SLEEP_S}s (Ctrl-C to stop; re-running later resumes) ---"
  sleep "$SLEEP_S"
done

echo "=== STILL INCOMPLETE after $MAX_ATTEMPTS attempts $(date) ==="
progress
echo "If the shard count is not advancing between attempts, the failure is not transient:"
echo "  * 'No space left'      -> check quota; the corpus needs 44 GB (+2.6 GB eval)"
echo "  * 401/403              -> HF_TOKEN not set/expired (these repos are public, so unlikely)"
echo "  * repo/path not found  -> check HF_CORPUS_REPO / HF_GRIDSTFT_REPO in cluster/config.env"
exit 1
