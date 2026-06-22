#!/bin/bash
# Publish the spectro checkpoints (and optionally the dataset) to your Hugging Face account.
# RUN ON THE LOGIN NODE (it has internet; compute nodes may not). Needs a WRITE token
# (cluster/secrets.env or a prior `huggingface-cli login`).
#
#     bash cluster/push_to_hf.sh ckpts      # upload the spectro MoE checkpoints (default)
#     bash cluster/push_to_hf.sh dataset    # upload the DeepMIMO-spectrogram dataset
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"

export PATH="$HOME/.local/bin:$PATH"     # uv
cd "$REPO_ROOT"

WHAT="${1:-ckpts}"
PRIV=""; [ "${HF_PRIVATE:-1}" = "1" ] && PRIV="--private"

if [ "$WHAT" = "ckpts" ]; then
    uv run --no-sync python spectro/scripts/hf_sync.py push-ckpts \
        --repo "$HF_MODEL_REPO" --dir "$CKPT_DIR" $PRIV
elif [ "$WHAT" = "dataset" ]; then
    uv run --no-sync python spectro/scripts/hf_sync.py push-dataset \
        --repo "$SPECTRO_DM_REPO" --dir "$SPECTRO_DM_DIR" $PRIV
else
    echo "usage: $0 {ckpts|dataset}" >&2; exit 1
fi
echo "Published: $WHAT"
