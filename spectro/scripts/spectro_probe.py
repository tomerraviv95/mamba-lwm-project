"""Unified FROZEN linear-probe — the apples-to-apples representation-quality metric.

Every extractor is FROZEN and read by the SAME standardized logistic-regression probe, so it measures
*representation quality* on equal footing — unlike the downstream Conv1d-seq head + finetuning, and unlike
comparing a frozen LWM against an END-TO-END DeepCNN (debug 2026-07-22: frozen random DeepCNN scores 0.40
mod vs its end-to-end 0.52 — the gap was adaptation, not representation; and the pretrained LWM frozen-probe
ties ResNet). Covers ``--extractor`` in {lwm, deepcnn (frozen random), resnet18/50, efficientnet_b0,
mobilenet_v3_small, raw}. StandardScaler matters: without it the LWM's 128-d features under-probe (0.38->0.44).

Examples:
    python spectro/scripts/spectro_probe.py --extractor lwm --arch mamba --patch 4 \
        --weights-suffix alluser_15k --synth-dir <eval> --run-tag indist15 [--random]
    python spectro/scripts/spectro_probe.py --extractor deepcnn --synth-dir <eval> --run-tag indist15
    python spectro/scripts/spectro_probe.py --extractor resnet18 --synth-dir <eval> --run-tag indist15
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectro_data import PROTOCOLS, TASKS, load_synthetic_data, load_spectro_data  # noqa: E402
from spectro_moe import SpectroMoE  # noqa: E402
from spectro_patchify import patch_geometry, spectrogram_patchify  # noqa: E402
from spectro_pretrain import weights_dir  # noqa: E402
from spectro_sweep import _subsample_count, _subsample_per_class  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_SUBMISSIONS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'submissions')
_IMAGENET = ('resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_small')


def _build_moe(arch, patch, suffix, pool, random_init, channels):
    """Build the MoE; load pretrained experts+router unless random_init. Geometry follows the EVAL's
    channel count so the random control matches the input shape; pretrained reads dims from the ckpt."""
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


@torch.no_grad()
def _extract(args, data, device) -> np.ndarray:
    """Frozen features (N, d) for the chosen extractor."""
    ex = args.extractor
    ch = data.spectrograms.shape[1] if data.spectrograms.ndim == 4 else 1
    if ex == 'lwm':
        moe = _build_moe(args.arch, args.patch, args.weights_suffix, args.pool, args.random, ch)
        return moe.extract_embeddings(data.spectrograms, routing='oracle', protocol_idx=data.protocol,
                                      device=device).numpy()
    if ex == 'raw':
        return spectrogram_patchify(data.spectrograms, patch=args.patch, normalize=True).mean(axis=1)
    if ex == 'deepcnn':                                   # frozen RANDOM CNN (fair frozen counterpart to e2e DeepCNN)
        from spectro_train_heads import DeepCNN
        torch.manual_seed(42); net = DeepCNN(in_channels=ch).to(device).eval()
        sp = data.spectrograms; sp = sp if torch.is_tensor(sp) else torch.as_tensor(np.asarray(sp), dtype=torch.float32)
        out = [net(sp[s:s + 128].float().to(device)).cpu() for s in range(0, sp.shape[0], 128)]
        return torch.cat(out).numpy()
    if ex in _IMAGENET:
        from spectro_train_heads import _imagenet_features
        return _imagenet_features(data, ex, device).numpy()
    raise ValueError(f"unknown extractor {ex!r}")


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score
    ap = argparse.ArgumentParser()
    ap.add_argument('--extractor', default='lwm',
                    choices=['lwm', 'deepcnn', 'raw', *_IMAGENET],
                    help="frozen feature extractor. 'deepcnn' = frozen RANDOM DeepCNN (the fair counterpart "
                         "to the end-to-end DeepCNN baseline).")
    ap.add_argument('--arch', choices=['mamba', 'transformer'], default='mamba', help='(lwm only)')
    ap.add_argument('--patch', type=int, default=4, choices=[4, 6, 8])
    ap.add_argument('--weights-suffix', default='alluser_15k', help='(lwm only)')
    ap.add_argument('--random', action='store_true', help='(lwm only) untrained backbone control')
    ap.add_argument('--pool', choices=['mean', 'meanstd_t', 'cls'], default='mean', help='(lwm only)')
    ap.add_argument('--synth-dir', default=None, help='eval corpus (70/10/20 split); omit for the demo set')
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
    X = _extract(args, data, device)                                  # frozen features (N, d)
    tr, te = data.train_idx, data.test_idx
    scaler = StandardScaler().fit(X[tr]); Xs = scaler.transform(X)      # standardize (fit on train only)
    mode, x_points = (('per_class', args.per_class_counts) if args.per_class_counts else ('counts', args.sample_counts))
    arm = 'probe_' + (f"{args.arch}{'_random' if args.random else ''}" if args.extractor == 'lwm'
                      else (args.extractor + ('_frozen' if args.extractor == 'deepcnn' else '')))
    print(f"PROBE {arm} feat={X.shape} eval={os.path.basename((args.synth_dir or 'demo').rstrip('/'))} "
          f"mode={mode} x={x_points}")

    results_by_task, results_by_x = {}, {}
    for task in args.tasks:
        y = data.labels[task].astype(int); y_tr = y[tr]; task_results = {}
        for x in x_points:
            f1s, accs, n_samples = [], [], None
            for sd in args.seeds:
                sub = (_subsample_per_class(y_tr, x, seed=sd) if mode == 'per_class'
                       else _subsample_count(len(tr), x, seed=sd))
                sel = tr[sub]; n_samples = len(sel)
                if len(set(y[sel].tolist())) < 2:
                    f1s.append(0.0); accs.append(0.0); continue
                pred = LogisticRegression(max_iter=1000).fit(Xs[sel], y[sel]).predict(Xs[te])
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
        'experiment_config': {'arm': arm, 'extractor': args.extractor, 'tasks': args.tasks, 'seeds': args.seeds,
                              'metric': 'macro_f1 (primary) + accuracy', 'head': 'standardized_linear_probe',
                              'pool': args.pool if args.extractor == 'lwm' else None,
                              'random_init': args.random if args.extractor == 'lwm' else None,
                              'x_axis': 'per_class_counts' if mode == 'per_class' else 'sample_counts',
                              'sample_counts': x_points if mode == 'counts' else None,
                              'per_class_counts': x_points if mode == 'per_class' else None},
        'results_by_task': results_by_task,
        'results_by_percentage': {k: {**v, 'composite_score': float(np.mean(list(v.values())))}
                                  for k, v in results_by_x.items()},
    }
    tag = (('_heldout' if args.synth_dir else '')
           + (f'_{args.weights_suffix}' if args.extractor == 'lwm' and args.weights_suffix else '')
           + (f'_{args.run_tag}' if args.run_tag else ''))
    out_dir = os.path.join(_SUBMISSIONS, f'submission_spectro_{arm}_p{args.patch}{tag}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nDone. probe results -> {out_dir}/aggregated_results.json")


if __name__ == '__main__':
    main()
