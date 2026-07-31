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
  echo "== [1/2] push checkpoints $CKPT_DIR -> $HF_MODEL_REPO =="
  $PY spectro/scripts/hf_sync.py push-ckpts --repo "$HF_MODEL_REPO" --dir "$CKPT_DIR" $PRIV \
    || echo "WARN: checkpoint push failed"
else
  echo "== [1/2] no checkpoint dir ($CKPT_DIR) — skip =="
fi

# 2) collated CSVs -> HF_STUDY_REPO (uploads the parent dir so the study_csv/ subfolder is preserved)
if ls "$STUDY_CSV_DIR"/study_csv/*.csv >/dev/null 2>&1; then
  N=$(ls "$STUDY_CSV_DIR"/study_csv/*.csv | wc -l)
  echo "== [2/2] push $N CSV(s) $STUDY_CSV_DIR/study_csv -> $HF_STUDY_REPO (study_csv/) =="
  $PY spectro/scripts/hf_sync.py push-dataset --repo "$HF_STUDY_REPO" --dir "$STUDY_CSV_DIR" $PRIV \
    || echo "WARN: CSV push failed"
else
  echo "== [2/2] no CSVs under $STUDY_CSV_DIR/study_csv/ — did 11_downstream_grid run? — skip =="
fi

echo "study publish done ($(date)). Plot locally: python spectro/scripts/plot_from_csv.py --hf-repo $HF_STUDY_REPO"
