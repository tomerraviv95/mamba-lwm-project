"""Plot accuracy vs. number of training samples for the spectro arms.

Adapted from ``scripts/plot_sample_variation.py``: no patch-size loop (fixed 4x4 geometry),
all tasks are classification so every y-axis is accuracy in [0,1]. Reads each arm's
``submission_spectro_{arm}/aggregated_results.json`` and writes a single figure with one
subplot per task plus a dual x-axis (#samples and training %).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_SUBMISSIONS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'submissions')
_PLOTS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'plots')

ARMS = [
    ('transformer', 'LWM-Spectro (Transformer MoE)', '#1f77b4', '-', 'o'),
    ('mamba',       'WiMamba MoE',                    '#d62728', '--', 's'),
    ('raw',         'Raw patches',                    '#7f7f7f', ':', '^'),
]
TASKS = [('task_modulation', 'Modulation'), ('task_snr', 'SNR'), ('task_mobility', 'Mobility')]


def _load(arm):
    path = os.path.join(_SUBMISSIONS, f'submission_spectro_{arm}', 'aggregated_results.json')
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    os.makedirs(_PLOTS, exist_ok=True)
    loaded = {arm: _load(arm) for arm, *_ in ARMS}
    if not any(loaded.values()):
        print("No results found. Run spectro_train_heads.py for at least one arm first.")
        return

    ref = next(v for v in loaded.values() if v)
    pcts = ref['experiment_config']['sample_percentages']

    fig, axes = plt.subplots(1, len(TASKS), figsize=(6 * len(TASKS), 5))
    if len(TASKS) == 1:
        axes = [axes]

    for ax, (task_key, task_name) in zip(axes, TASKS):
        for arm, label, color, ls, marker in ARMS:
            agg = loaded.get(arm)
            if not agg or task_key not in agg['results_by_task']:
                continue
            res = agg['results_by_task'][task_key]['results']
            xs = [res[str(p)]['n_samples'] for p in pcts]
            ys = [res[str(p)]['score'] for p in pcts]
            ax.plot(xs, ys, color=color, linestyle=ls, marker=marker, label=label, linewidth=2)
        ax.set_title(task_name)
        ax.set_xlabel('Number of training samples')
        ax.set_ylabel('Test accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # secondary x-axis: training data %
        sec = ax.secondary_xaxis('top')
        ref_res = ref['results_by_task'][task_key]['results']
        sec.set_xticks([ref_res[str(p)]['n_samples'] for p in pcts])
        sec.set_xticklabels([f'{p}%' for p in pcts])
        sec.set_xlabel('Training data %')

    axes[0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Spectrogram downstream transfer: accuracy vs. #training samples', y=1.02)
    fig.tight_layout()
    out = os.path.join(_PLOTS, 'spectro_performance_vs_samples.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
