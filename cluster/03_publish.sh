#!/usr/bin/env bash
# LOGIN-NODE step: publish everything produced by 01/02 to Hugging Face + Weights & Biases.
# Compute nodes on this cluster have no usable outbound internet, so ALL network I/O lives here.
# Run from the repo root on the login node AFTER 01/02 have finished:
#
#   bash cluster/03_publish.sh
#
# Uploads: (1) all pretrained checkpoints -> HF_MODEL_REPO, (2) all downstream submission dirs ->
# HF_RESULTS_REPO + W&B, (3) offline W&B pretraining runs via `wandb sync`. Each step is best-effort
# and independent; a failure in one is reported but does not abort the others.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$ROOT/cluster/config.env" ] || { echo "ERROR: run from the repo root — cluster/config.env not found under $ROOT"; exit 1; }
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_XET=1                       # classic HTTP upload works on the cluster; xet doesn't
PY="${PY:-uv run --no-sync python}"
PRIV=$([ "${HF_PRIVATE:-1}" = 1 ] && echo --private || echo "")

# 1) pretrained checkpoints (push the parent dir so per-(arch,patch) subfolders are preserved, no clobber)
if [ -d "$CKPT_DIR" ]; then
  echo "== [1/3] push checkpoints $CKPT_DIR -> $HF_MODEL_REPO =="
  $PY spectro/scripts/hf_sync.py push-ckpts --repo "$HF_MODEL_REPO" --dir "$CKPT_DIR" $PRIV \
    || echo "WARN: checkpoint push failed"
else
  echo "== [1/3] no checkpoint dir ($CKPT_DIR) — skip =="
fi

# 2) downstream results (all submission dirs) -> HF dataset + W&B
SUBS=$(ls -d "$REPO_ROOT"/spectro/outputs/submissions/submission_spectro_* 2>/dev/null || true)
if [ -n "$SUBS" ]; then
  echo "== [2/3] push $(echo "$SUBS" | wc -l) result dir(s) -> $HF_RESULTS_REPO + W&B =="
  # shellcheck disable=SC2086
  $PY spectro/scripts/upload_results.py --submissions $SUBS --hf-repo "$HF_RESULTS_REPO" $PRIV \
    --wandb-project "$WANDB_PROJECT" || echo "WARN: results upload failed"
else
  echo "== [2/3] no submission dirs under spectro/outputs/submissions/ — skip =="
fi

# 3) sync offline W&B runs (the pretraining curves logged on the compute node)
if ls cluster/logs/wandb/offline-* >/dev/null 2>&1; then
  echo "== [3/3] wandb sync offline runs =="
  # shellcheck disable=SC2086
  $PY -m wandb sync cluster/logs/wandb/offline-* || echo "WARN: wandb sync failed"
else
  echo "== [3/3] no offline W&B runs to sync =="
fi

echo "publish done ($(date))."
