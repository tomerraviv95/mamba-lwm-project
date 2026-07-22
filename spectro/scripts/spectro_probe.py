"""Representation-quality PROBE: frozen backbone + mean-pool + a LINEAR probe (logistic regression).

This is the clean, readout-agnostic metric of how good the pretrained representation is — unlike the
downstream Conv1d-sequence head + finetuning, which can extract the task from *random* features and mask
the pretraining lift (see the 2026-07-22 debug). Run it for the pretrained checkpoint AND with
``--random`` (same architecture, untrained) so the pretraining lift = pretrained_probe - random_probe on
identical footing. Writes ``aggregated_results.json`` in the sweep schema so it plots with everything else.

Example:
    python spectro/scripts/spectro_probe.py --arch mamba --patch 4 --weights-suffix alluser_15k \
        --synth-dir spectro/outputs/spectro_eval_alluser15_gridstft --run-tag indist15 \
        --sample-counts 50 100 200 400 600 --seeds 42 43 44
    # + the random-init control:
    python spectro/scripts/spectro_probe.py --arch mamba ... --random
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectro_data import PROTOCOLS, TASKS, load_synthetic_data, load_spectro_data  # noqa: E402
from spectro_moe import SpectroMoE  # noqa: E402
from spectro_patchify import patch_geometry  # noqa: E402
from spectro_pretrain import weights_dir  # noqa: E402
from spectro_sweep import _subsample_count, _subsample_per_class  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_SUBMISSIONS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'submissions')


def _build_moe(arch, patch, suffix, pool, random_init, channels):
    """Build the MoE; load pretrained experts+router unless random_init (then leave untrained).

    Geometry follows the EVAL's channel count (1 = magnitude, 2 = dual [STFT|grid]) so the random-init
    control matches the input shape; the pretrained path additionally reads d_model/n_layers/geometry
    from the checkpoint (authoritative)."""
    geom = patch_geometry(patch, channels=channels)
    el, ml, in_ch, d_model, n_layers = geom['element_length'], geom['max_len'], channels, 128, 12
    if not random_init:
        wdir = weights_dir(arch, patch, suffix)
        s = torch.load(os.path.join(wdir, f'{PROTOCOLS[0]}_expert.pth'), map_location='cpu', weights_only=False)
        el = s.get('element_length', el); ml = s.get('max_len', ml)
        in_ch = max(1, el // (patch * patch)); d_model = s.get('d_model', 128); n_layers = s.get('n_layers', 12)
    if random_init:
        torch.manual_seed(42)
    moe = SpectroMoE(PROTOCOLS, d_model=d_model, arch=arch, n_layers=n_layers, pool=pool, patch=patch,
                     element_length=el, max_len=ml, in_channels=in_ch)
    if not random_init:
        for p in PROTOCOLS:
            moe.load_expert(p, torch.load(os.path.join(wdir, f'{p}_expert.pth'), map_location='cpu',
                                          weights_only=False)['state_dict'])
        moe.router.load_state_dict(torch.load(os.path.join(wdir, 'router.pth'), map_location='cpu',
                                              weights_only=False)['state_dict'])
    return moe


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['mamba', 'transformer'], default='mamba')
    ap.add_argument('--patch', type=int, default=4, choices=[4, 6, 8])
    ap.add_argument('--weights-suffix', default='alluser_15k')
    ap.add_argument('--synth-dir', default=None, help='eval corpus (70/10/20 split); omit for the demo set')
    ap.add_argument('--random', action='store_true', help='untrained backbone control (isolates the lift)')
    ap.add_argument('--pool', choices=['mean', 'meanstd_t', 'cls'], default='mean')
    ap.add_argument('--tasks', nargs='+', default=list(TASKS.keys()))
    ap.add_argument('--sample-counts', type=int, nargs='+', default=[50, 100, 200, 400, 600])
    ap.add_argument('--per-class-counts', type=int, nargs='+', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--val-frac', type=float, default=0.10)
    ap.add_argument('--test-frac', type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=42, help='data-split seed')
    ap.add_argument('--run-tag', default='')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = (load_synthetic_data(args.synth_dir, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac)
            if args.synth_dir else load_spectro_data(seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac))
    channels = data.spectrograms.shape[1] if data.spectrograms.ndim == 4 else 1
    moe = _build_moe(args.arch, args.patch, args.weights_suffix, args.pool, args.random, channels)
    X = moe.extract_embeddings(data.spectrograms, routing='oracle', protocol_idx=data.protocol,
                               device=device).numpy()                     # frozen pooled features (N, d)
    tr, te = data.train_idx, data.test_idx
    mode, x_points = (('per_class', args.per_class_counts) if args.per_class_counts
                      else ('counts', args.sample_counts))
    arm = f"probe_{args.arch}{'_random' if args.random else ''}"
    print(f"PROBE {arm} p{args.patch} [{args.weights_suffix}] pool={args.pool} feat={X.shape} "
          f"eval={os.path.basename((args.synth_dir or 'demo').rstrip('/'))} mode={mode} x={x_points}")

    results_by_task, results_by_x = {}, {}
    for task in args.tasks:
        y = data.labels[task].astype(int); y_tr = y[tr]
        task_results = {}
        for x in x_points:
            f1s, accs, n_samples = [], [], None
            for sd in args.seeds:
                sub = (_subsample_per_class(y_tr, x, seed=sd) if mode == 'per_class'
                       else _subsample_count(len(tr), x, seed=sd))
                sel = tr[sub]; n_samples = len(sel)
                if len(set(y[sel].tolist())) < 2:
                    f1s.append(0.0); accs.append(0.0); continue
                clf = LogisticRegression(max_iter=300).fit(X[sel], y[sel])
                pred = clf.predict(X[te])
                f1s.append(float(f1_score(y[te], pred, average='macro', zero_division=0)))
                accs.append(float(accuracy_score(y[te], pred)))
            key = f"{x}pc" if mode == 'per_class' else str(n_samples)
            task_results[key] = {'score': float(np.mean(f1s)), 'score_std': float(np.std(f1s)), 'scores': f1s,
                                 'accuracy': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
                                 'accuracies': accs, 'n_samples': n_samples,
                                 'per_class': x if mode == 'per_class' else None}
            results_by_x.setdefault(key, {})[f'task_{task}'] = float(np.mean(f1s))
            print(f"  [{arm}] {task:12s} n={n_samples:5d}  F1={np.mean(f1s):.4f}  acc={np.mean(accs):.4f}", flush=True)
        results_by_task[f'task_{task}'] = {'name': TASKS[task]['name'], 'results': task_results}

    aggregated = {
        'experiment_config': {'arm': arm, 'tasks': args.tasks, 'seeds': args.seeds,
                              'metric': 'macro_f1 (primary) + accuracy', 'head': f'linear_probe_{args.pool}',
                              'pool': args.pool, 'random_init': args.random,
                              'x_axis': 'per_class_counts' if mode == 'per_class' else 'sample_counts',
                              'sample_counts': x_points if mode == 'counts' else None,
                              'per_class_counts': x_points if mode == 'per_class' else None},
        'results_by_task': results_by_task,
        'results_by_percentage': {k: {**v, 'composite_score': float(np.mean(list(v.values())))}
                                  for k, v in results_by_x.items()},
    }
    tag = (('_heldout' if args.synth_dir else '') + (f'_{args.weights_suffix}' if args.weights_suffix else '')
           + (f'_{args.run_tag}' if args.run_tag else ''))
    out_dir = os.path.join(_SUBMISSIONS, f'submission_spectro_{arm}_p{args.patch}{tag}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nDone. probe results -> {out_dir}/aggregated_results.json")


if __name__ == '__main__':
    main()
