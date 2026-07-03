"""Accuracy vs. #training-samples, one FIGURE PER PATCH SIZE (4/6/8), 3 task subplots each.

This is the headline transfer-learning plot (paper Fig-3 style): for each patch size it produces a
figure with three subplots (modulation / SNR / mobility), and in every subplot each method is a curve
of test accuracy against the number of downstream training samples. Our two pretrained backbones
(Transformer MoE, WiMamba MoE) are drawn on top (thick, high z-order) of the baselines
(random-init MoE, random patches, and the ImageNet ResNet-18/50 generic-vision references).

Reads ``submission_spectro_{arm}_p{patch}{suffix}/aggregated_results.json`` written by
``spectro_train_heads.py``. ImageNet arms are patch-independent (a fixed CNN, no patchify), so the
same resnet18/resnet50 results are reused across all patch figures.

Example::

    python spectro/scripts/spectro_plot_samples_per_patch.py --suffix _heldout_gridstft
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
# arm -> (label, color, linestyle, marker, on_top). on_top => thick line drawn above the baselines.
ARMS = [
    ('transformer_synth', 'Transformer MoE (ours)',       '#2ca02c', '-',  'D', True),
    ('mamba',             'WiMamba MoE (ours)',            '#d62728', '-',  's', True),
    ('random_init',       'Random-init MoE',               '#9467bd', '--', 'v', False),
    ('raw',               'Random patches',                '#7f7f7f', ':',  '^', False),
    ('resnet18',          'ImageNet ResNet-18',            '#1f77b4', '-.', 'o', False),
    ('resnet50',          'ImageNet ResNet-50',            '#17becf', '-.', 'P', False),
]
# task key -> (display name, chance level = 1/n_classes)
TASKS = [('task_modulation', 'Modulation (5-way)', 1 / 5),
         ('task_snr', 'SNR (7-way)', 1 / 7),
         ('task_mobility', 'Mobility (3-way)', 1 / 3)]

# ImageNet arms are patch-independent: fall back to the patch-4 dir if a per-patch dir is missing.
_PATCH_INDEPENDENT = {'resnet18', 'resnet50'}


def _curve(arm, patch, suffix, task_key):
    """(#samples, mean-acc, std) lists for (arm, patch, task), sorted by sample count, or None."""
    path = os.path.join(_SUBMISSIONS, f'submission_spectro_{arm}_p{patch}{suffix}', 'aggregated_results.json')
    if not os.path.exists(path) and arm in _PATCH_INDEPENDENT:
        path = os.path.join(_SUBMISSIONS, f'submission_spectro_{arm}_p4{suffix}', 'aggregated_results.json')
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    r = d.get('results_by_task', {}).get(task_key, {}).get('results', {})
    pts = sorted((v['n_samples'], v['score'], v.get('score_std', 0.0)) for v in r.values())
    if not pts:
        return None
    xs, ys, es = zip(*pts)
    return list(xs), list(ys), list(es)


def _scope(suffix):
    if 'gridstft' in suffix:
        return 'in-domain held-out cities, dual [STFT|grid] representation'
    if 'grid' in suffix:
        return 'in-domain held-out cities, grid representation'
    if '_heldout' in suffix:
        return 'in-domain held-out cities'
    return 'demo'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='_heldout_gridstft',
                    help="submission-dir suffix, e.g. '_heldout_gridstft' (dual) or '_heldout' or ''.")
    args = ap.parse_args()
    os.makedirs(_PLOTS, exist_ok=True)
    scope = _scope(args.suffix)

    saved = []
    for patch in PATCHES:
        fig, axes = plt.subplots(1, len(TASKS), figsize=(6 * len(TASKS), 5.5))
        drew_any = False
        for ax, (task_key, task_name, chance) in zip(axes, TASKS):
            for arm, label, color, ls, marker, on_top in ARMS:
                c = _curve(arm, patch, args.suffix, task_key)
                if not c:
                    continue
                drew_any = True
                xs, ys, es = c
                ax.plot(xs, ys, color=color, linestyle=ls, marker=marker, label=label,
                        linewidth=3 if on_top else 1.8, markersize=9 if on_top else 6,
                        zorder=5 if on_top else 3, alpha=1.0 if on_top else 0.85)
                if any(e > 0 for e in es):
                    lo = [y - e for y, e in zip(ys, es)]
                    hi = [y + e for y, e in zip(ys, es)]
                    ax.fill_between(xs, lo, hi, color=color, alpha=0.18 if on_top else 0.10,
                                    zorder=(4 if on_top else 2), linewidth=0)
            ax.axhline(chance, color='gray', linestyle=':', linewidth=1, alpha=0.7)
            ax.text(0.02, chance + 0.01, 'chance', color='gray', fontsize=8,
                    va='bottom', transform=ax.get_yaxis_transform())
            ax.set_title(task_name)
            ax.set_xlabel('# training samples (log scale)')
            ax.set_ylabel('Test accuracy')
            ax.set_ylim(0, 1)
            ax.set_xscale('log')
            ax.grid(True, which='both', alpha=0.3)
        if not drew_any:
            plt.close(fig)
            print(f"patch {patch}: no results for suffix={args.suffix!r} — skipped.")
            continue
        axes[0].legend(loc='best', fontsize=9)
        fig.suptitle(f'Downstream accuracy vs. #training samples — patch {patch}×{patch}\n({scope})',
                     y=1.03, fontsize=14)
        fig.tight_layout()
        out = os.path.join(_PLOTS, f'spectro_samples_p{patch}{args.suffix}.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        saved.append(out)
        print(f"saved -> {out}")

    if not saved:
        print("No figures produced — check that submission dirs exist for the given suffix.")


if __name__ == '__main__':
    main()
