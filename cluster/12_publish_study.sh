#!/usr/bin/env bash
# LOGIN-NODE step (STUDY): publish the multi-seed study outputs to Hugging Face.
# Compute nodes have no outbound internet, so ALL network I/O lives here. Run from the repo root
# AFTER 10_pretrain_grid + 11_downstream_grid have finished:
#
#   bash cluster/12_publish_study.sh
#
# Uploads: (1) all pretrained checkpoints in CKPT_DIR -> HF_MODEL_REPO (per-(arch,patch,seed) subfolders
# preserved, no clobber); (2) the collated CSVs ($STUDY_CSV_DIR/study_csv/*.csv) -> HF_STUDY_REPO under
# study_csv/. Each step is best-effort and independent. Download + plot locally with
# `spectro/scripts/plot_from_csv.py --hf-repo $HF_STUDY_REPO`.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$ROOT/cluster/config.env" ] || { echo "ERROR: run from the repo root — cluster/config.env not found under $ROOT"; exit 1; }
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_XET=1
PY="${PY:-uv run --no-sync python}"
PRIV=$([ "${HF_PRIVATE:-1}" = 1 ] && echo --private || echo "")

# 1) study checkpoints -> HF_MODEL_REPO
if [ -d "$CKPT_DIR" ]; then
  # Auth preflight: whoami fails fast and unambiguously, instead of a 401 traceback mid-upload.
if ! $PY -c "
import os,sys
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from huggingface_hub import HfApi
try:
    print('  HF auth OK as', HfApi().whoami()['name'])
except Exception as e:
    print('  HF AUTH FAILED:', type(e).__name__, str(e)[:120]); sys.exit(1)
"; then
  echo "ERROR: not authenticated to Hugging Face. Set a WRITE token and retry:"
  echo "    export HF_TOKEN=hf_xxx     (https://huggingface.co/settings/tokens)"
  echo "    # or: hf auth login"
  exit 1
fi

echo "== [1/2] push checkpoints $CKPT_DIR -> $HF_MODEL_REPO =="
  $PY spectro/scripts/hf_sync.py push-ckpts --repo "$HF_MODEL_REPO" --dir "$CKPT_DIR" $PRIV \
    || { echo "ERROR: checkpoint push FAILED"; PUSH_FAILED=1; }
else
  echo "== [1/2] no checkpoint dir ($CKPT_DIR) — skip =="
fi

# 2) collated CSVs -> HF_STUDY_REPO (uploads the parent dir so each study_csv_{head}/ subfolder is preserved)
if ls "$STUDY_CSV_DIR"/study_csv*/*.csv >/dev/null 2>&1; then
  N=$(ls "$STUDY_CSV_DIR"/study_csv*/*.csv | wc -l)
  echo "== [2/2] push $N CSV(s) from $STUDY_CSV_DIR/study_csv*/ -> $HF_STUDY_REPO =="
  $PY spectro/scripts/hf_sync.py push-dataset --repo "$HF_STUDY_REPO" --dir "$STUDY_CSV_DIR" $PRIV \
    || { echo "ERROR: CSV push FAILED"; PUSH_FAILED=1; }
else
  echo "== [2/2] no CSVs under $STUDY_CSV_DIR/study_csv*/ — did 11_downstream_grid run? — skip =="
fi

# A failed push previously only WARNed, so the script still printed "study publish done" and the
# stale (empty) CSVs stayed on the Hub looking like a valid result. Any push failure is now fatal.
if [ "${PUSH_FAILED:-0}" = 1 ]; then
  echo
  echo "PUBLISH FAILED — nothing was uploaded. The remote still holds whatever was there before."
  echo "Most common cause is auth: 401 Unauthorized / 'no HF_TOKEN set'. Fix with either"
  echo "    export HF_TOKEN=hf_xxx        # a WRITE token from https://huggingface.co/settings/tokens"
  echo "    hf auth login                 # or huggingface-cli login"
  echo "then re-run. Remember to pass the same env as the study, e.g. SPECTRO_CONV_STEM=1,"
  echo "or STUDY_VARIANT resolves to the wrong namespace ($STUDY_VARIANT here)."
  exit 1
fi

# Refuse to publish header-only CSVs. hf_sync pushes whatever is on disk, so a failed study
# silently overwrites good remote results with empty files that look valid.
_bad=0
for f in "$STUDY_CSV_DIR"/study_csv_"${STUDY_VARIANT:-$STUDY_HEAD}"/*.csv; do
  [ -e "$f" ] || continue
  _rows=$(( $(wc -l < "$f") - 1 ))
  if [ "$_rows" -lt 1 ]; then echo "ERROR: $f has $_rows data rows — refusing to publish"; _bad=1
  else echo "  ok $(basename "$f"): $_rows rows"; fi
done
if [ "$_bad" = 1 ]; then
  echo "Nothing published. Check the downstream logs and submission dirs first:"
  echo "  ls -d spectro/outputs/submissions/*${STUDY_VARIANT:-}* | wc -l"
  exit 1
fi

echo "study publish done ($(date)). Plot locally: python spectro/scripts/plot_from_csv.py --hf-repo $HF_STUDY_REPO --variant ${STUDY_VARIANT:-$STUDY_HEAD}"
