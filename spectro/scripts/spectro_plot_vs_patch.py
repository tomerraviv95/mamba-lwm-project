"""Plot downstream score vs. PATCH SIZE, one subplot per task (modulation / SNR / mobility).

Patch-axis companion to ``spectro_plot_sample_variation.py`` (which fixes patch and sweeps #samples).
Modeled on the repo's ``scripts/plot_task_scores.py`` (patch on x, score on y, one line per arm).
Reads each arm's ``submission_spectro_{arm}_p{patch}{suffix}/aggregated_results.json`` and plots the
score at a chosen training percentage (default 100%) against patch size {4,6,8}.

Examples::

    python spectro/scripts/spectro_plot_vs_patch.py --suffix _heldout      # in-domain held-out cities
    python spectro/scripts/spectro_plot_vs_patch.py --suffix '' --pct 100  # the demo sweep
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_SUBMISSIONS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'submissions')
_PLOTS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'plots')

PATCHES = [4, 6, 8]
# arm -> (label, color, linestyle, marker). 'transformer' (published baseline) only exists for the demo.
ARMS = [
    ('transformer',       'LWM-Spectro baseline (Transformer MoE)', '#1f77b4', '-',  'o'),
    ('transformer_synth', 'Transformer MoE (ours)',                 '#2ca02c', '-.', 'D'),
    ('mamba',             'WiMamba MoE (ours)',                     '#d62728', '--', 's'),
    ('random_init',       'Random-init MoE (no pretraining)',       '#9467bd', ':',  'v'),
    ('raw',               'Raw patches',                            '#7f7f7f', ':',  '^'),
]
# task key -> (display name, chance level = 1/n_classes)
TASKS = [('task_modulation', 'Modulation', 1 / 5),
         ('task_snr', 'SNR', 1 / 7),
         ('task_mobility', 'Mobility', 1 / 3)]


def _score(arm, patch, suffix, pct):
    """Score for (arm, patch) at training percentage ``pct`` -> {task_key: score} or None."""
    path = os.path.join(_SUBMISSIONS, f'submission_spectro_{arm}_p{patch}{suffix}', 'aggregated_results.json')
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    out = {}
    for task_key, *_ in TASKS:
        r = d.get('results_by_task', {}).get(task_key, {}).get('results', {})
        if str(pct) in r:
            out[task_key] = r[str(pct)]['score']
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='_heldout',
                    help="submission-dir suffix: '_heldout' (in-domain held-out cities) or '' (demo).")
    ap.add_argument('--pct', type=int, default=100, help='training-data percentage to read (default 100).')
    args = ap.parse_args()
    os.makedirs(_PLOTS, exist_ok=True)

    # arm -> {patch: {task: score}}
    data = {arm: {p: _score(arm, p, args.suffix, args.pct) for p in PATCHES} for arm, *_ in ARMS}
    have = [arm for arm, *_ in ARMS if any(data[arm][p] for p in PATCHES)]
    if not have:
        print(f"No results found for suffix={args.suffix!r} pct={args.pct}.")
        return

    fig, axes = plt.subplots(1, len(TASKS), figsize=(6 * len(TASKS), 5.5))
    for ax, (task_key, task_name, chance) in zip(axes, TASKS):
        for arm, label, color, ls, marker in ARMS:
            if arm not in have:
                continue
            xs = [p for p in PATCHES if data[arm][p] and task_key in data[arm][p]]
            ys = [data[arm][p][task_key] for p in xs]
            if xs:
                ax.plot(xs, ys, color=color, linestyle=ls, marker=marker, label=label,
                        linewidth=2, markersize=9)
        ax.axhline(chance, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(PATCHES[0], chance + 0.01, 'chance', color='gray', fontsize=8, va='bottom')
        ax.set_title(task_name)
        ax.set_xlabel('Patch size')
        ax.set_ylabel('Test accuracy')
        ax.set_xticks(PATCHES)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc='best', fontsize=9)
    scope = ('in-domain held-out cities, dual [STFT|grid] repr' if 'gridstft' in args.suffix
             else 'in-domain held-out cities, grid repr' if 'grid' in args.suffix
             else 'in-domain held-out cities' if '_heldout' in args.suffix else 'demo')
    fig.suptitle(f'Downstream accuracy vs. patch size ({scope}, {args.pct}% train)', y=1.02, fontsize=14)
    fig.tight_layout()
    out = os.path.join(_PLOTS, f'spectro_score_vs_patch{args.suffix or "_demo"}.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
