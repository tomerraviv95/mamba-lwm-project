#!/usr/bin/env bash
# STEP 1 — pull the paper-aligned data from HF. RUN THIS ON THE LOGIN NODE (it has internet;
# compute nodes may not, and the xet backend is disabled so this uses classic HTTP).
#
#     bash cluster/download_data.sh            # normal (skips dirs that already verify)
#     FORCE=1 bash cluster/download_data.sh    # re-download everything
#
# Fetches (and VERIFIES every shard listed in the manifest exists + is non-empty; a partial/interrupted
# earlier download leaves a tiny manifest.json but truncated shards -> torch.load EOFError at pretrain,
# so we re-fetch with --force when verification fails):
#   - 85%-user pretrain corpus + 15%-user in-distribution eval  (HF_CORPUS_REPO: corpus/ + eval/)
#   - held-out-cities cross-environment eval                    (HF_GRIDSTFT_REPO: eval/ only)
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$ROOT/cluster/config.env" ] || { echo "ERROR: run from the repo root — cluster/config.env not found under $ROOT"; exit 1; }
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
PY="${PY:-uv run --no-sync python}"

# verify_dir <dir>: 0 if manifest present AND every listed shard exists and is >1KB; nonzero otherwise.
verify_dir() {
  $PY - "$1" <<'PY'
import json, os, sys
d = sys.argv[1]; m = os.path.join(d, 'manifest.json')
if not os.path.isfile(m): sys.exit(1)
try:
    shards = json.load(open(m)).get('shards', [])
except Exception:
    sys.exit(1)
if not shards: sys.exit(1)
for s in shards:
    p = os.path.join(d, s)
    if not os.path.isfile(p) or os.path.getsize(p) < 1024:
        print(f"  bad/missing shard: {s}"); sys.exit(2)
sys.exit(0)
PY
}

# fetch <repo> <only> <corpus_dir> <eval_dir> <verify_dir> : download, then verify+force-retry once.
fetch() {
  local repo="$1" only="$2" cdir="$3" edir="$4" vdir="$5"
  local force=""; [ "${FORCE:-0}" = 1 ] && force="--force"
  $PY spectro/scripts/hf_download_gridstft.py --repo "$repo" --only "$only" \
      --corpus-dir "$cdir" --eval-dir "$edir" $force
  if ! verify_dir "$vdir"; then
    echo "  !! $vdir failed verification — re-fetching with --force ..."
    $PY spectro/scripts/hf_download_gridstft.py --repo "$repo" --only "$only" \
        --corpus-dir "$cdir" --eval-dir "$edir" --force
  fi
}

echo "=== download_data $(date) ==="
echo "corpus    <- $HF_CORPUS_REPO   -> $CORPUS_DIR (+ $EVAL_INDIST_DIR)"
echo "xenv eval <- $HF_GRIDSTFT_REPO -> $EVAL_XENV_DIR"

# all-user corpus (corpus/) + in-distribution 15% eval (eval/) — one repo, both subfolders
fetch "$HF_CORPUS_REPO" both "$CORPUS_DIR" "$EVAL_INDIST_DIR" "$CORPUS_DIR"
verify_dir "$EVAL_INDIST_DIR" || fetch "$HF_CORPUS_REPO" eval "$CORPUS_DIR" "$EVAL_INDIST_DIR" "$EVAL_INDIST_DIR"
# held-out-cities eval only (skip that repo's large corpus)
fetch "$HF_GRIDSTFT_REPO" eval "$CORPUS_DIR" "$EVAL_XENV_DIR" "$EVAL_XENV_DIR"

echo "--- final verification ---"
ok=1
for d in "$CORPUS_DIR" "$EVAL_INDIST_DIR" "$EVAL_XENV_DIR"; do
  if verify_dir "$d" >/dev/null; then echo "  OK  $d"; else echo "  BAD $d (shards missing/empty)" >&2; ok=0; fi
done
[ "$ok" = 1 ] && echo "=== download_data DONE $(date) ===" || { echo "=== download_data INCOMPLETE — see above $(date) ==="; exit 1; }
