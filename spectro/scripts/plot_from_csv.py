#!/usr/bin/env python3
"""Plot the multi-seed study CSVs: one figure per (patch, eval), each with 2 task subplots, mean +/- std bands.

Reads the tidy CSVs produced by ``collate_csv.py`` (columns: patch,seed,arm,arm_label,eval,task,per_class,
n_samples,macro_f1,accuracy), aggregates the metric across the 5 seeds (mean and std), and renders every arm
as a line + shaded std band. Produces 4 PNGs: ``study_p{patch}_{eval}.png`` for patch in {4,8}, eval in
{seen,unseen} (whatever combos are present).

Usage:
    # from a local CSV dir (e.g. after a manual pull, or the local smoke test):
    python spectro/scripts/plot_from_csv.py --csv-dir spectro/outputs/results_csv/study_csv

    # or pull the CSVs from HF first (needs HF_TOKEN for a private repo):
    python spectro/scripts/plot_from_csv.py --hf-repo tomerraviv95/lwm-spectro-results \
        --csv-dir spectro/outputs/results_csv
"""
from __future__ import annotations
import argparse, csv, glob, math, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASK_TITLES = {'modulation': 'Modulation Task', 'snr_doppler': 'SNR,Doppler Task'}
TASK_ORDER = ['modulation', 'snr_doppler']
# arm_label -> (color, linestyle, marker, linewidth); ordered for the legend
STYLE = {
    'LWM Mamba (pretrained)':       ('#d62728', '-',  's', 2.4),
    'LWM Transformer (pretrained)': ('#1f77b4', '-',  'o', 2.4),
    'Mamba (random init)':          ('#d62728', '--', 'x', 1.3),
    'Transformer (random init)':    ('#1f77b4', '--', 'x', 1.3),
    'DeepCNN (end-to-end)':         ('#2ca02c', '-.', 'D', 1.6),
    'ResNet-18 (frozen)':           ('#9467bd', ':',  'v', 1.4),
    'raw patches':                  ('#7f7f7f', ':',  '.', 1.0),
}


def maybe_download(hf_repo: str, csv_dir: str, subdir: str):
    from huggingface_hub import snapshot_download
    tok = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    print(f"downloading {subdir}/ CSVs from {hf_repo} -> {csv_dir}")
    snapshot_download(repo_id=hf_repo, repo_type='dataset', local_dir=csv_dir,
                      allow_patterns=[f'{subdir}/*.csv'], token=tok)


def load_rows(csv_dir: str):
    files = glob.glob(os.path.join(csv_dir, '**', '*.csv'), recursive=True)
    rows = []
    for fp in files:
        with open(fp) as f:
            for r in csv.DictReader(f):
                r['patch'] = int(r['patch']); r['seed'] = int(r['seed'])
                r['per_class'] = int(r['per_class']); r['n_samples'] = int(r['n_samples'])
                r['macro_f1'] = float(r['macro_f1']); r['accuracy'] = float(r['accuracy'])
                rows.append(r)
    print(f"loaded {len(rows)} rows from {len(files)} CSV file(s) in {csv_dir}")
    return rows


def agg(rows, metric):
    """(patch,eval,task,arm_label,per_class) -> (mean, std, n_seeds)."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r['patch'], r['eval'], r['task'], r['arm_label'], r['per_class'])].append(r[metric])
    out = {}
    for k, vals in buckets.items():
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
        out[k] = (m, sd, len(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-dir', default='spectro/outputs/results_csv',
                    help='local dir holding the study_csv[_variant]/ subfolder (or the CSVs directly)')
    ap.add_argument('--variant', default='cnn1d', help="head variant subfolder study_csv_{variant} to read")
    ap.add_argument('--hf-repo', default=None, help='if set, snapshot_download the variant CSVs first')
    ap.add_argument('--metric', choices=['accuracy', 'macro_f1'], default='accuracy')
    ap.add_argument('--out-dir', default='spectro/outputs/plots')
    args = ap.parse_args()

    subdir = f'study_csv_{args.variant}' if args.variant else 'study_csv'
    if args.hf_repo:
        maybe_download(args.hf_repo, args.csv_dir, subdir)
    # read the variant subfolder if present, else the given dir (supports pointing --csv-dir straight at CSVs)
    read_root = os.path.join(args.csv_dir, subdir)
    if not os.path.isdir(read_root):
        read_root = args.csv_dir
    rows = load_rows(read_root)
    if not rows:
        raise SystemExit(f"no CSV rows found under {args.csv_dir}")
    A = agg(rows, args.metric)

    patches = sorted({r['patch'] for r in rows})
    evals = [e for e in ('seen', 'unseen') if any(r['eval'] == e for r in rows)]
    os.makedirs(args.out_dir, exist_ok=True)

    for patch in patches:
        for ev in evals:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            tasks = [t for t in TASK_ORDER if any(k[:3] == (patch, ev, t) for k in A)]
            for ax, task in zip(axes, tasks):
                for lab, (c, ls, mk, lw) in STYLE.items():
                    pts = sorted((k[4], *A[k]) for k in A if k[0] == patch and k[1] == ev
                                 and k[2] == task and k[3] == lab)
                    if not pts:
                        continue
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; es = [p[2] for p in pts]
                    ax.plot(xs, ys, ls, color=c, marker=mk, label=lab, lw=lw, ms=5)
                    ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                                    [y + e for y, e in zip(ys, es)], color=c, alpha=0.15, linewidth=0)
                ax.set_xscale('log'); ax.set_xlabel('samples per class')
                ax.set_ylabel(args.metric.replace('_', '-'))
                ax.set_title(TASK_TITLES.get(task, task)); ax.grid(alpha=0.3); ax.legend(fontsize=7)
            n_seeds = max((A[k][2] for k in A if k[0] == patch and k[1] == ev), default=0)
            fig.suptitle(f"{args.variant} head  |  patch {patch}  |  {ev} cities  |  mean +/- std over {n_seeds} seeds",
                         fontsize=11)
            plt.tight_layout()
            vtag = f'{args.variant}_' if args.variant else ''
            out = os.path.join(args.out_dir, f'study_{vtag}p{patch}_{ev}.png')
            plt.savefig(out, dpi=130); plt.close(fig)
            print(f"saved -> {out}")


if __name__ == '__main__':
    main()
