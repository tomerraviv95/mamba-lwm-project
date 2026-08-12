#!/usr/bin/env bash
# STEP 0 (login node) — fetch the DeepMIMO ray-tracing scenarios so the corpus can be GENERATED on
# the cluster instead of uploaded to it.
#
# Why bother: the generated corpus is ~130 GB at --sc-channels 3, while the scenarios it is
# generated FROM are ~24 GB and never change. Fetching them once makes every future representation
# change a compute job rather than a 130 GB upload + 130 GB download.
#
# Run on the LOGIN node (compute nodes may have no internet). Resumable: scenarios already present
# are skipped, so re-run after an interruption.
#
#     bash cluster/04_fetch_scenarios.sh
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
# shellcheck disable=SC1091
source cluster/config.env
cd "$REPO_ROOT"
PY="${PY:-uv run --no-sync python}"

# DeepMIMO is recorded in this repo as a GITLINK (mode 160000) with no .gitmodules, so a fresh
# clone/pull leaves an EMPTY DeepMIMO/ directory and `import deepmimo` fails with nothing obviously
# wrong in the tree. Restore it at the pinned commit rather than tracking upstream HEAD, so the
# generated corpus stays reproducible.
DEEPMIMO_COMMIT=f640e86e66ab0b83129eefcb9078c539119438ad
if [ ! -f "$REPO_ROOT/DeepMIMO/pyproject.toml" ]; then
  echo "=== DeepMIMO source missing (empty gitlink) -> cloning at $DEEPMIMO_COMMIT ==="
  rm -rf "$REPO_ROOT/DeepMIMO"
  git clone -q https://github.com/DeepMIMO/DeepMIMO.git "$REPO_ROOT/DeepMIMO" || {
    echo "FAILED to clone DeepMIMO (login node has internet; compute nodes may not)"; exit 1; }
  git -C "$REPO_ROOT/DeepMIMO" checkout -q "$DEEPMIMO_COMMIT" || exit 1
  echo "  cloned -> installing editable"
  uv pip install -q -e "$REPO_ROOT/DeepMIMO" || { echo "FAILED to install DeepMIMO"; exit 1; }
fi

echo "=== datagen dependencies ==="
# NOTE: the `|| {...}` must stay on ONE line -- anything after it is heredoc body, not shell.
$PY - <<'PY' || { printf '%s\n' "FAILED: install the missing datagen deps, e.g." "    uv pip install 'sionna>=2.0'" "    uv pip install -e ./DeepMIMO"; exit 1; }
import importlib, sys
bad = []
for m in ('deepmimo', 'sionna', 'torch'):
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  MISSING {m}: {type(e).__name__}: {e}"); bad.append(m)
sys.exit(1 if bad else 0)
PY

echo "=== scenarios ==="
# The 20 LWM training cities + the three held-out cross-environment scenarios. Kept as a literal
# list rather than imported from deepmimo_channel so this script runs before any GPU/torch import.
SCENARIOS="city_0_newyork_3p5_lwm city_1_losangeles_3p5_lwm city_2_chicago_3p5_lwm \
city_3_houston_3p5_lwm city_4_phoenix_3p5_lwm city_5_philadelphia_3p5_lwm city_6_miami_3p5_lwm \
city_7_sandiego_3p5_lwm city_8_dallas_3p5_lwm city_9_sanfrancisco_3p5_lwm city_10_austin_3p5_lwm \
city_11_santaclara_3p5_lwm city_12_fortworth_3p5_lwm city_13_columbus_3p5_lwm \
city_14_charlotte_3p5_lwm city_15_indianapolis_3p5_lwm city_16_sanfrancisco_3p5_lwm \
city_17_seattle_3p5_lwm city_18_denver_3p5_lwm city_19_oklahoma_3p5_lwm \
asu_campus_3p5 boston5g_3p5 o1_3p5"

fail=0
for s in $SCENARIOS; do
  $PY - "$s" <<'PY' || fail=1
import os, sys
import deepmimo as dm
name = sys.argv[1]
folder = dm.get_scenario_folder(name)
# "present" = the folder exists and holds real files; an interrupted download leaves an empty dir.
if os.path.isdir(folder) and any(
        os.path.getsize(os.path.join(r, f)) > 0
        for r, _, fs in os.walk(folder) for f in fs):
    print(f"  have {name}")
    sys.exit(0)
print(f"  fetching {name} ...", flush=True)
try:
    dm.download(name)
    print(f"  OK   {name}")
except Exception as e:
    print(f"  FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1)
PY
done

echo
du -sh "$REPO_ROOT/deepmimo_scenarios" 2>/dev/null
[ "$fail" = 0 ] && echo "SCENARIOS OK — next: sbatch cluster/05_datagen.sbatch" \
               || echo "SOME SCENARIOS FAILED — re-run this script (it skips what it already has)"
exit "$fail"
