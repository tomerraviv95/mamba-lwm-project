"""Sample-variation sweep over frozen spectrogram features.

For a given arm's feature matrix (N, d) and the shared train/val/test split, sweep training
data percentages, train a fresh classification head per (task, percentage), evaluate test
accuracy, and emit ``aggregated_results.json`` (schema compatible with the channel pipeline's
plotter) plus a per-task radar chart.
"""
from __future__ import annotations

import json
import os
import sys
from math import pi
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from shared.finetune import finetune, subsample_training_data  # noqa: E402
from spectro_data import TASKS, SpectroData  # noqa: E402
from spectro_train_heads_config import TASK_CONFIGS, ClassificationHead, Conv1dHead  # noqa: E402

SAMPLE_PERCENTAGES = [0.2, 0.4, 0.6, 0.8, 1.0]
# absolute-count sweep (used by the per-patch accuracy-vs-samples figures); spreads the low end to
# expose the few-shot regime where pretraining helps most.
SAMPLE_COUNTS = [50, 100, 250, 500, 1000, 2500, 4000]
# paper's few-shot axis: samples PER CLASS (Table II/III use 5..400/cls; modulation 2..256/cls).
PER_CLASS_COUNTS = [2, 4, 8, 16, 32, 64, 128, 256]


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro-averaged F1 (paper's primary metric) — mean per-class F1, robust to class imbalance."""
    from sklearn.metrics import f1_score
    preds = logits.argmax(dim=1).cpu().numpy()
    return float(f1_score(labels.cpu().numpy(), preds, average='macro', zero_division=0))


def _subsample_count(n_total: int, count: int, seed: int) -> np.ndarray:
    """Deterministic seeded pick of ``min(count, n_total)`` training indices (>=1), sorted."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    k = min(max(1, count), n_total)
    return np.sort(perm[:k])


def _subsample_per_class(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Seeded pick of up to ``k`` positions PER CLASS from ``labels`` (positions into ``labels``), sorted."""
    rng = np.random.RandomState(seed)
    picks = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        rng.shuffle(ci)
        picks.extend(ci[:min(k, len(ci))])
    return np.sort(np.array(picks, dtype=np.int64))


def _build_head(features: torch.Tensor, n_classes: int, backbone=None):
    """Pick the downstream head: paper 1-D CNN over token sequences (3-D features), else MLP probe.
    When an end-to-end ``backbone`` is given (e.g. Deep CNN), the MLP head sits on its feature dim."""
    if backbone is not None:
        return ClassificationHead(backbone.feat_dim, n_classes)
    if features.dim() == 3:                       # (N, T, d) token sequence -> paper residual CNN head
        import spectro_train_heads_config as _cfg
        return Conv1dHead(features.shape[2], n_classes, second_order=_cfg.HEAD_SECOND_ORDER)
    return ClassificationHead(features.shape[1], n_classes)


def run_sweep(arm: str, features: torch.Tensor, data: SpectroData, out_dir: str, *,
              sample_counts=None, per_class_counts=None, percentages=SAMPLE_PERCENTAGES,
              seeds=None, seed: int = 42, head_restarts: int = 1, device: str = "cuda",
              epochs_override: int | None = None, backbone_factory=None, embed_fn=None) -> Dict:
    """Sample-variation sweep for one arm; write aggregated_results.json + radar chart.

    Reports BOTH macro-F1 (paper's primary metric, drives val model-selection) and accuracy per point.
    X-axis modes (priority): ``per_class_counts`` (paper's samples/class), ``sample_counts`` (absolute),
    or legacy ``percentages``. Each point averages over ``seeds`` (reseeding subsample + head init).

    ``features`` is either a pooled matrix (N, d) -> MLP head, a token sequence (N, T, d) -> 1-D CNN
    head (paper), or raw spectrograms (N, C, H, W) when ``backbone_factory`` is given (end-to-end
    baseline trained via ``embed_fn``). ``head_restarts`` keeps the best-VAL-F1 head per point to
    reject degenerate collapses at tiny sample counts."""
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(seeds) if seeds else [seed]
    head_restarts = max(1, head_restarts)
    feat_dim = tuple(features.shape[1:])
    train_idx, val_idx, test_idx = data.train_idx, data.val_idx, data.test_idx
    if per_class_counts is not None:
        mode, x_points = 'per_class', list(per_class_counts)
    elif sample_counts is not None:
        mode, x_points = 'counts', list(sample_counts)
    else:
        mode, x_points = 'pct', list(percentages)

    results_by_task = {}
    results_by_x = {}
    for task in TASKS:
        cfg = dict(TASK_CONFIGS[task])
        if epochs_override:
            cfg['epochs'] = epochs_override
        y = torch.as_tensor(data.labels[task], dtype=torch.long)
        n_classes = data.n_classes(task)
        y_train = y[train_idx].numpy()
        val_feats, val_y = features[val_idx], y[val_idx]
        test_feats, test_y = features[test_idx], y[test_idx]

        task_results = {}
        for x in x_points:
            f1s, accs = [], []
            n_samples = None
            for sd in seeds:
                if mode == 'per_class':
                    sub = _subsample_per_class(y_train, x, seed=sd)
                elif mode == 'counts':
                    sub = _subsample_count(len(train_idx), x, seed=sd)
                else:
                    sub = subsample_training_data(len(train_idx), x, seed=sd)
                sel = train_idx[sub]
                n_samples = len(sel)
                tr_feats, tr_y = features[sel], y[sel]
                # best-of-N heads by val macro-F1: reject degenerate (collapsed-to-chance) inits at low n
                best_val, best_f1, best_acc = -1.0, 0.0, 0.0
                for r in range(head_restarts):
                    torch.manual_seed(sd * 1000 + r); np.random.seed(sd * 1000 + r)
                    bb = backbone_factory() if backbone_factory else None
                    head = _build_head(features, n_classes, backbone=bb)
                    _, history, f1, tlab, tout = finetune(
                        head, tr_feats, tr_y, val_feats, val_y, test_feats, test_y,
                        score_fn=_macro_f1, backbone=bb, embed_fn=embed_fn,
                        fine_tune_layers=('full' if bb is not None else None),
                        epochs=cfg['epochs'], lr=cfg['lr'],
                        weight_decay=cfg['weight_decay'], batch_size=cfg['batch_size'],
                        patience=cfg['patience'], scheduler_step=cfg['scheduler_step'],
                        scheduler_gamma=cfg['scheduler_gamma'], device=device)
                    v = max(history['val_score']) if history['val_score'] else -1.0
                    if v > best_val:
                        best_val, best_f1, best_acc = v, f1, _accuracy(tout, tlab)
                f1s.append(best_f1); accs.append(best_acc)

            mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s))
            mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
            key = (f"{x}pc" if mode == 'per_class' else str(n_samples) if mode == 'counts'
                   else str(int(x * 100)))
            task_results[key] = {'score': mean_f1, 'score_std': std_f1, 'scores': f1s,
                                 'accuracy': mean_acc, 'accuracy_std': std_acc, 'accuracies': accs,
                                 'n_samples': n_samples, 'per_class': x if mode == 'per_class' else None}
            results_by_x.setdefault(key, {})[f'task_{task}'] = mean_f1
            print(f"  [{arm}] {task:12s} n={n_samples:5d}{f' ({x}/cls)' if mode=='per_class' else ''}  "
                  f"F1={mean_f1:.4f}+/-{std_f1:.4f}  acc={mean_acc:.4f} ({len(seeds)} seeds)")

        results_by_task[f'task_{task}'] = {'name': TASKS[task]['name'], 'results': task_results}

    aggregated = {
        'experiment_config': {
            'arm': arm, 'tasks': list(TASKS.keys()), 'feature_dim': feat_dim, 'seeds': seeds,
            'head_restarts': head_restarts, 'metric': 'macro_f1 (primary) + accuracy',
            'head': ('deepcnn_e2e' if backbone_factory else 'conv1d' if features.dim() == 3 else 'mlp'),
            'x_axis': {'per_class': 'per_class_counts', 'counts': 'sample_counts',
                       'pct': 'percentages'}[mode],
            'per_class_counts': list(per_class_counts) if mode == 'per_class' else None,
            'sample_counts': list(sample_counts) if mode == 'counts' else None,
            'sample_percentages': [int(p * 100) for p in percentages] if mode == 'pct' else None,
        },
        'results_by_task': results_by_task,
        'results_by_percentage': {
            pk: {**scores, 'composite_score': float(np.mean(list(scores.values())))}
            for pk, scores in results_by_x.items()
        },
    }
    with open(os.path.join(out_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)
    try:
        _radar_chart(aggregated, os.path.join(out_dir, 'radar_chart.png'), arm)
    except Exception as e:
        print(f"  (radar skipped: {e})")
    return aggregated


def _radar_chart(aggregated: Dict, save_path: str, arm: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    tasks = aggregated['experiment_config']['tasks']
    # x-keys in stored order (absolute counts or percentages), read off the first task's results
    pcts = list(aggregated['results_by_task'][f'task_{tasks[0]}']['results'].keys())
    names = [TASKS[t]['name'] for t in tasks]
    N = len(names)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(7, 7))
    cmap = plt.cm.viridis
    for i, pk in enumerate(pcts):
        scores = [aggregated['results_by_task'][f'task_{t}']['results'][pk]['score'] for t in tasks]
        ax.plot(angles, scores + scores[:1], 'o-', color=cmap(i / max(len(pcts) - 1, 1)),
                label=f'n={pk}', linewidth=1.5)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(names)
    ax.set_ylim(0, 1); ax.set_title(f'{arm}: macro-F1 by task')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1)); ax.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  radar chart -> {save_path}")
