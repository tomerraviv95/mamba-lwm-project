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
from spectro_train_heads_config import TASK_CONFIGS, ClassificationHead  # noqa: E402

SAMPLE_PERCENTAGES = [0.2, 0.4, 0.6, 0.8, 1.0]


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def run_sweep(arm: str, features: torch.Tensor, data: SpectroData, out_dir: str, *,
              percentages=SAMPLE_PERCENTAGES, seed: int = 42, device: str = "cuda",
              epochs_override: int | None = None) -> Dict:
    """Run the percentage sweep for one arm; write aggregated_results.json + radar chart."""
    os.makedirs(out_dir, exist_ok=True)
    input_dim = features.shape[1]
    train_idx, val_idx, test_idx = data.train_idx, data.val_idx, data.test_idx

    results_by_task = {}
    results_by_pct = {}
    for task in TASKS:
        cfg = dict(TASK_CONFIGS[task])
        if epochs_override:
            cfg['epochs'] = epochs_override
        y = torch.as_tensor(data.labels[task], dtype=torch.long)
        n_classes = data.n_classes(task)
        val_feats, val_y = features[val_idx], y[val_idx]
        test_feats, test_y = features[test_idx], y[test_idx]

        task_results = {}
        for pct in percentages:
            sub = subsample_training_data(len(train_idx), pct, seed=seed)
            sel = train_idx[sub]
            tr_feats, tr_y = features[sel], y[sel]

            head = ClassificationHead(input_dim, n_classes)
            _, _, score, _, _ = finetune(
                head, tr_feats, tr_y, val_feats, val_y, test_feats, test_y,
                score_fn=_accuracy, epochs=cfg['epochs'], lr=cfg['lr'],
                weight_decay=cfg['weight_decay'], batch_size=cfg['batch_size'],
                patience=cfg['patience'], scheduler_step=cfg['scheduler_step'],
                scheduler_gamma=cfg['scheduler_gamma'], device=device)

            n_samples = len(sel)
            pct_key = str(int(pct * 100))
            task_results[pct_key] = {'score': score, 'n_samples': n_samples}
            results_by_pct.setdefault(pct_key, {})[f'task_{task}'] = score
            print(f"  [{arm}] {task:10s} pct={pct_key:>3s}% n={n_samples:5d}  acc={score:.4f}")

        results_by_task[f'task_{task}'] = {'name': TASKS[task]['name'], 'results': task_results}

    aggregated = {
        'experiment_config': {
            'arm': arm, 'sample_percentages': [int(p * 100) for p in percentages],
            'tasks': list(TASKS.keys()), 'feature_dim': input_dim, 'seed': seed,
        },
        'results_by_task': results_by_task,
        'results_by_percentage': {
            pk: {**scores, 'composite_score': float(np.mean(list(scores.values())))}
            for pk, scores in results_by_pct.items()
        },
    }
    with open(os.path.join(out_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)
    _radar_chart(aggregated, os.path.join(out_dir, 'radar_chart.png'), arm)
    return aggregated


def _radar_chart(aggregated: Dict, save_path: str, arm: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    tasks = aggregated['experiment_config']['tasks']
    pcts = [str(p) for p in aggregated['experiment_config']['sample_percentages']]
    names = [TASKS[t]['name'] for t in tasks]
    N = len(names)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(7, 7))
    cmap = plt.cm.viridis
    for i, pk in enumerate(pcts):
        scores = [aggregated['results_by_task'][f'task_{t}']['results'][pk]['score'] for t in tasks]
        ax.plot(angles, scores + scores[:1], 'o-', color=cmap(i / max(len(pcts) - 1, 1)),
                label=f'{pk}%', linewidth=1.5)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(names)
    ax.set_ylim(0, 1); ax.set_title(f'{arm}: accuracy vs training %')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1)); ax.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  radar chart -> {save_path}")
