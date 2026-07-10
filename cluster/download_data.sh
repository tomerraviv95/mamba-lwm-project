#!/usr/bin/env bash
# STEP 1 — pull the paper-aligned data from HF. RUN THIS ON THE LOGIN NODE (it has internet;
# compute nodes may not, and the xet backend is disabled so this uses classic HTTP).
#
#     bash cluster/download_data.sh
#
# Fetches (idempotent — skips a dir that already has a manifest):
#   - the 85%-user pretrain corpus + 15%-user in-distribution eval  (HF_CORPUS_REPO: corpus/ + eval/)
#   - the held-out-cities cross-environment eval                    (HF_GRIDSTFT_REPO: eval/ only)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
PY="${PY:-uv run --no-sync python}"

echo "=== download_data $(date) ==="
echo "corpus  <- $HF_CORPUS_REPO       -> $CORPUS_DIR (+ $EVAL_INDIST_DIR)"
echo "xenv eval <- $HF_GRIDSTFT_REPO   -> $EVAL_XENV_DIR"

# all-user corpus (corpus/) + in-distribution 15% eval (eval/)
$PY spectro/scripts/hf_download_gridstft.py --repo "$HF_CORPUS_REPO" \
    --corpus-dir "$CORPUS_DIR" --eval-dir "$EVAL_INDIST_DIR"

# held-out-cities eval only (skip that repo's 2.5 GB corpus)
$PY spectro/scripts/hf_download_gridstft.py --repo "$HF_GRIDSTFT_REPO" \
    --eval-dir "$EVAL_XENV_DIR" --only eval

echo "--- manifests present? ---"
for d in "$CORPUS_DIR" "$EVAL_INDIST_DIR" "$EVAL_XENV_DIR"; do
    if [ -f "$d/manifest.json" ]; then echo "  OK  $d"; else echo "  MISSING $d/manifest.json" >&2; fi
done
echo "=== download_data DONE $(date) ==="
