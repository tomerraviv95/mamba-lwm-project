#!/usr/bin/env bash
# Fail-fast checks before submitting the study. Every item here corresponds to a failure that has
# actually cost this project time:
#   * a stale/truncated corpus that verified "clean" because only the local manifest was checked
#   * a repr/waveform mismatch between pretraining and eval (identical tensor SHAPES, so it loaded
#     fine and produced noise)
#   * train/eval user overlap once multiple BS positions were pooled
#   * checkpoints silently reused across recipe changes via the done3 guard
#
#   bash cluster/00_preflight.sh          # after download_data.sh, before run_study.sh
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env
cd "$REPO_ROOT"
PY="${PY:-uv run --no-sync python}"
fail=0

echo "=== config ==="
printf '  %-16s %s\n' REPR "${REPR:-sc}" PATCHES "$PATCHES" SEEDS "$STUDY_SEEDS" \
  SUFFIX "$STUDY_SUFFIX" VARIANT "$STUDY_VARIANT" W_CONT "$SPECTRO_W_CONT" STEPS "$SPECTRO_STEPS"

echo "=== datasets ==="
for d in "$CORPUS_DIR" "$EVAL_INDIST_DIR" "$EVAL_XENV_DIR"; do
  $PY - "$d" <<'PY' || fail=1
import json, os, sys
d = sys.argv[1]; m = os.path.join(d, 'manifest.json')
if not os.path.isfile(m):
    print(f"  MISSING {d}"); sys.exit(1)
man = json.load(open(m))
n_sh = len(man.get('shards', []))
present = sum(1 for s in man['shards'] if os.path.isfile(os.path.join(d, s)))
small = [s for s in man['shards'] if os.path.isfile(os.path.join(d, s))
         and os.path.getsize(os.path.join(d, s)) < 1024]
ok = present == n_sh and not small
print(f"  {'OK ' if ok else 'BAD'} {os.path.basename(d):34s} n={man.get('n_samples')} "
      f"shards={present}/{n_sh} waveform={man.get('waveform')} sc_norm={man.get('sc_norm')} "
      f"draws={man.get('draws')} split={man.get('user_split_part')}")
# a repr mismatch between pretrain and eval is UNDETECTABLE downstream (same shapes) -> assert here
if man.get('waveform') != 'sc' or man.get('sc_norm') != 'global':
    print(f"     ^ expected waveform=sc sc_norm=global"); sys.exit(1)
sys.exit(0 if ok else 1)
PY
done

echo "=== corpus/eval consistency ==="
$PY - "$CORPUS_DIR" "$EVAL_INDIST_DIR" "$EVAL_XENV_DIR" <<'PY' || fail=1
import json, os, sys
ms = [json.load(open(os.path.join(d, 'manifest.json'))) for d in sys.argv[1:]]
keys = ('waveform', 'sc_win', 'sc_norm', 'channels')
base = {k: ms[0].get(k) for k in keys}
bad = False
for d, m in zip(sys.argv[1:], ms):
    got = {k: m.get(k) for k in keys}
    if got != base:
        print(f"  MISMATCH {os.path.basename(d)}: {got} != {base}"); bad = True
print(f"  {'OK ' if not bad else 'BAD'} generation recipe identical across corpus and both evals: {base}")
sys.exit(1 if bad else 0)
PY

echo "=== checkpoint collisions ==="
for P in $PATCHES; do for S in $STUDY_SEEDS; do for A in mamba transformer; do
  W="$REPO_ROOT/spectro/outputs/pretrained_models/spectro_${A}_p${P}_${STUDY_SUFFIX}_s${S}_weights"
  if [ -d "$W" ]; then
    n=$(ls "$W" 2>/dev/null | grep -c expert.pth)
    echo "  EXISTS $(basename "$W") (${n} experts) — the done3 guard will SKIP it; rm -rf or bump STUDY_SUFFIX for a new recipe"
  fi
done; done; done
echo "  (nothing listed above = clean slate)"

echo
[ "$fail" = 0 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT FAILED"
exit "$fail"
