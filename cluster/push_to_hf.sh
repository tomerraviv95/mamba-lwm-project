#!/bin/bash
# Publish the corpus and/or checkpoints to your Hugging Face account.
# RUN ON THE LOGIN NODE (it has internet; compute nodes may not). Needs a WRITE token
# (cluster/secrets.env or a prior `huggingface-cli login`).
#
#     bash cluster/push_to_hf.sh dataset    # upload the synthetic corpus
#     bash cluster/push_to_hf.sh ckpts      # upload the pretrained checkpoints
#     bash cluster/push_to_hf.sh all        # both
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HOME/lwm-competition-2025/cluster/config.env"

export PATH="$HOME/.local/bin:$PATH"     # uv
cd "$REPO_ROOT"

WHAT="${1:-all}"
PRIV=""; [ "${HF_PRIVATE:-1}" = "1" ] && PRIV="--private"

if [ "$HF_USER" = "CHANGE_ME" ]; then
    echo "ERROR: set HF_USER in cluster/config.env first." >&2; exit 1
fi

if [ "$WHAT" = "dataset" ] || [ "$WHAT" = "all" ]; then
    uv run --no-sync python spectro/scripts/hf_sync.py push-dataset \
        --repo "$HF_DATASET_REPO" --dir "$DATA_DIR" $PRIV
fi
if [ "$WHAT" = "ckpts" ] || [ "$WHAT" = "all" ]; then
    uv run --no-sync python spectro/scripts/hf_sync.py push-ckpts \
        --repo "$HF_MODEL_REPO" --dir "$CKPT_DIR" $PRIV
fi
echo "Published: $WHAT"
