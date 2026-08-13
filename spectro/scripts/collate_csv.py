#!/usr/bin/env python3
"""Collate the study's per-arm ``aggregated_results.json`` files into ONE tidy CSV per (patch, seed).

The multi-seed / multi-patch cluster study (cluster/11_downstream_grid.sbatch) writes one submission dir per
(arm, eval, seed) via ``spectro_train_heads.py``. This script globs those dirs for a given (patch, seed),
reads the macro-F1 / accuracy at each per-class count for the two plotted tasks, and emits a single tidy CSV:

    patch,seed,arm,arm_label,eval,task,per_class,n_samples,macro_f1,accuracy

Study run-tag convention (see 11_downstream_grid.sbatch):
  * LWM arms  ``mamba`` / ``transformer_synth``  -> weights-suffix ``study_s{seed}``, run-tag ``{eval}_s{seed}``
  * random_init -> run-tag ``{eval}_s{seed}_mambarand`` / ``{eval}_s{seed}_tfrand``
  * resnet18 / raw / deepcnn -> run-tag ``{eval}_s{seed}``
where ``{eval}`` is ``seen`` (in-dist alluser15) or ``unseen`` (held-out cities).

Usage:
    python spectro/scripts/collate_csv.py --patch 4 --seed 1 \
        --submissions spectro/outputs/submissions --out spectro/outputs/results_csv/study_csv
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, sys, time

# canonical arm key -> display label used by the plotter
ARM_LABELS = {
    'mamba':                 'LWM Mamba (pretrained)',
    'transformer_synth':     'LWM Transformer (pretrained)',
    'random_init_mambarand': 'Mamba (random init)',
    'random_init_tfrand':    'Transformer (random init)',
    'resnet18':              'ResNet-18 (frozen)',
    'mobilenet_v3_small':    'MobileNetV3-S (frozen, param-matched)',
    'raw':                   'raw patches',
    'deepcnn':               'DeepCNN (end-to-end)',
}
PLOT_TASKS = ('modulation', 'modulation3', 'snr_doppler')


def _classify(dirname: str, patch: int, seed: int, variant: str = ''):
    """Return (arm_key, eval_name) for a study submission dir, or None if it isn't part of this
    (patch, seed[, variant]) run. ``variant`` is the head tag (e.g. 'cnn1d'/'mlp') stamped into the run-tag;
    when given, only dirs carrying that token are collated (so mlp and cnn1d runs never mix)."""
    m = re.match(rf'^submission_spectro_(.+?)_p{patch}_heldout_(.+)$', dirname)
    if not m:
        return None
    arm_token, rest = m.group(1), m.group(2)
    toks = rest.split('_')
    if f's{seed}' not in toks:                            # this run's seed must appear as an _s{seed}_ token
        return None
    if variant and variant not in toks:                  # and the head-variant token, if filtering
        return None
    if 'unseen' in rest:
        ev = 'unseen'
    elif 'seen' in rest:
        ev = 'seen'
    else:
        return None
    if arm_token == 'random_init':
        if rest.endswith('mambarand'):
            return 'random_init_mambarand', ev
        if rest.endswith('tfrand'):
            return 'random_init_tfrand', ev
        return None
    if arm_token in ('mamba', 'transformer_synth', 'resnet18', 'mobilenet_v3_small', 'raw', 'deepcnn'):
        return arm_token, ev
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--variant', default='', help="head tag ('cnn1d'/'mlp') to filter dirs by; '' = no filter")
    ap.add_argument('--submissions', default='spectro/outputs/submissions')
    ap.add_argument('--out', default='spectro/outputs/results_csv/study_csv',
                    help='output directory; writes results_p{patch}_s{seed}.csv there')
    ap.add_argument('--newer-than', type=float, default=0.0, metavar='EPOCH',
                    help='reject aggregated_results.json older than this unix time. A submission dir '
                         'is NOT cleared between runs, so an arm that dies before writing leaves the '
                         'PREVIOUS run\'s json in place and this script silently collates it as if '
                         'it were current. That shipped stale mamba rows into a published study.')
    args = ap.parse_args()

    rows = []
    stale, arm_tasks = [], {}
    for path in sorted(glob.glob(os.path.join(args.submissions, 'submission_spectro_*'))):
        if not os.path.isdir(path):
            continue
        hit = _classify(os.path.basename(path), args.patch, args.seed, args.variant)
        if not hit:
            continue
        arm_key, ev = hit
        jf = os.path.join(path, 'aggregated_results.json')
        if not os.path.isfile(jf):
            print(f"  WARN no aggregated_results.json in {os.path.basename(path)}")
            continue
        if args.newer_than and os.path.getmtime(jf) < args.newer_than:
            age = (args.newer_than - os.path.getmtime(jf)) / 3600.0
            stale.append(f"{os.path.basename(path)} ({age:.1f}h older than this run)")
            continue
        with open(jf) as f:
            js = json.load(f)
        arm_tasks[(arm_key, ev)] = sorted(
            t for t in PLOT_TASKS if js.get('results_by_task', {}).get(f'task_{t}'))
        for task in PLOT_TASKS:
            tr = js.get('results_by_task', {}).get(f'task_{task}')
            if not tr:
                continue
            for v in tr['results'].values():
                rows.append({
                    'patch': args.patch, 'seed': args.seed, 'arm': arm_key,
                    'arm_label': ARM_LABELS[arm_key], 'eval': ev, 'task': task,
                    'per_class': v.get('per_class'), 'n_samples': v.get('n_samples'),
                    'macro_f1': round(float(v['score']), 6), 'accuracy': round(float(v['accuracy']), 6),
                })

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f'results_p{args.patch}_s{args.seed}.csv')
    fields = ['patch', 'seed', 'arm', 'arm_label', 'eval', 'task', 'per_class', 'n_samples', 'macro_f1', 'accuracy']
    rows.sort(key=lambda r: (r['eval'], r['task'], r['arm'], r['per_class'] or 0))
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    arms = sorted({r['arm'] for r in rows}); evals = sorted({r['eval'] for r in rows})
    print(f"wrote {len(rows)} rows -> {out_csv}  (arms={arms}, evals={evals})")
    bad = False
    for d in stale:
        print(f"  STALE (skipped): {d}"); bad = True
    # Arms disagreeing on which tasks they carry is the signature of a partially-updated run: the
    # task list changed (e.g. modulation3 was added) and only some arms were re-run. Averaging or
    # plotting across that silently compares different recipes, so refuse it.
    sets = {tuple(v) for v in arm_tasks.values()}
    if len(sets) > 1:
        print("  INCONSISTENT task sets across arms -- this run mixes results from different recipes:")
        for (a, e), t in sorted(arm_tasks.items()):
            print(f"    {a:24s} {e:6s} {list(t)}")
        bad = True
    missing = [(a, e) for a in ARM_LABELS for e in ('seen', 'unseen') if (a, e) not in arm_tasks]
    if missing:
        print(f"  MISSING {len(missing)} (arm, eval) cells: {missing}")
        bad = True
    if bad:
        print("  -> collate refuses to certify this CSV; the sbatch will fail the task.")
        sys.exit(2)


if __name__ == '__main__':
    main()
