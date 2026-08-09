#!/usr/bin/env bash
# Collate the local single-carrier run into a tidy CSV and render the study figures.
#
# Two conventions collate_csv.py enforces that run_sc_local.sh does not produce directly:
#   * random_init dirs must end in '_mambarand' / '_tfrand' (that suffix is how the collator knows
#     WHICH architecture the random-init control instantiated). We stamp it here.
#   * the plotter defaults to --metric accuracy, but the study's primary metric is macro-F1.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PATCH=8
SEED=1
VARIANT=sc2
CSV_DIR=spectro/outputs/results_csv_sc

# random_init was run with --moe-arch mamba -> tag its dirs accordingly
for d in spectro/outputs/submissions/submission_spectro_random_init_p${PATCH}_heldout_*_s${SEED}; do
  [ -d "$d" ] || continue
  case "$d" in *mambarand|*tfrand) continue;; esac
  mv "$d" "${d}_mambarand" && echo "renamed $(basename "$d") -> $(basename "${d}")_mambarand"
done

$PY spectro/scripts/collate_csv.py --patch $PATCH --seed $SEED --variant $VARIANT \
    --submissions spectro/outputs/submissions --out "$CSV_DIR/study_csv_${VARIANT}" || exit 1

$PY spectro/scripts/plot_from_csv.py --csv-dir "$CSV_DIR" --variant "$VARIANT" \
    --metric macro_f1 --out-dir spectro/outputs/plots || exit 1

echo "--- figures:"
ls -la spectro/outputs/plots/study_${VARIANT}_p${PATCH}_*.png
