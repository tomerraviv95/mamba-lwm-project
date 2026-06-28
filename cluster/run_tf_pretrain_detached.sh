#!/usr/bin/env bash
# Detached transformer patch-study pretrain (p4/6/8), micro-batch 8 x accum 4 = eff 32.
# Survives Claude-session teardown (launch with: setsid bash cluster/run_tf_pretrain_detached.sh </dev/null &>/dev/null &).
# Idempotent-ish: skips a patch whose weights dir already has 3 experts + router.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG=cluster/logs/m3_transformer_pretrain_p468.log
{
  echo "=== DETACHED transformer pretrain (batch 8 x accum 4 = eff 32) $(date) ==="
  for P in 4 6 8; do
    D="spectro/outputs/pretrained_models/spectro_transformer_p${P}_weights"
    if [ "$(ls "$D" 2>/dev/null | grep -c expert.pth)" = "3" ] && [ -f "$D/router.pth" ]; then
      echo "############ PATCH $P already complete — skipping ############"; continue
    fi
    echo "############ TRANSFORMER PRETRAIN PATCH $P ############"
    PATCH=$P MODE=pretrain ARCHES=transformer BATCH=8 ACCUM=4 \
      PRETRAIN_DIR=spectro/outputs/spectro_deepmimo_mult8_vary bash spectro/scripts/run_patch_study.sh \
      || { echo "PATCH $P FAILED"; break; }
  done
  echo "ALL TRANSFORMER PRETRAINS DONE $(date)"
} >> "$LOG" 2>&1
