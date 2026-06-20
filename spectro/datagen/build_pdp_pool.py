"""Extract a compact PDP pool from the 20 LWM cities for on-the-fly spectrogram streaming.

Saves one ``pdp_pool.pt`` holding per-(sampled-)user ray tables (delay/power/phase/AoA, padded
to a common path count) + a city index. This few-MB artifact replaces a 327 GB materialized
10M-spectrogram dataset: the streaming generator (stream.py) draws PDPs from it and synthesizes
spectrograms on the fly during pretraining. Upload it to HF; the cluster needs no DeepMIMO.

Usage::

    python spectro/datagen/build_pdp_pool.py --per-city 5000 --out spectro/outputs/pdp_pool
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepmimo_channel import CITY_SCENARIOS, extract_city_pdp  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-city', type=int, default=5000, help='max valid users sampled per city.')
    ap.add_argument('--out', default=os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'pdp_pool'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.per_city = 50

    rng = np.random.RandomState(args.seed)
    parts = {k: [] for k in ('delay', 'power_linear', 'phase', 'aoa_az')}
    city_idx, counts = [], []
    for ci, scn in enumerate(CITY_SCENARIOS):
        pdp = extract_city_pdp(scn, bs_idx=1)
        u = pdp['delay'].shape[0]
        idx = rng.permutation(u)[:min(args.per_city, u)]
        for k in parts:
            parts[k].append(pdp[k][idx])
        city_idx.append(np.full(len(idx), ci, dtype=np.int64))
        counts.append(len(idx))
        print(f"  {scn}: {u} valid -> {len(idx)}")

    kmax = max(a.shape[1] for a in parts['delay'])
    pad = lambda key: np.concatenate(
        [np.pad(a, ((0, 0), (0, kmax - a.shape[1]))) for a in parts[key]], axis=0).astype(np.float32)
    pool = {
        'delay': torch.from_numpy(pad('delay')),
        'power_linear': torch.from_numpy(pad('power_linear')),
        'phase': torch.from_numpy(pad('phase')),
        'aoa_az': torch.from_numpy(pad('aoa_az')),
        'city': torch.from_numpy(np.concatenate(city_idx)),
        'city_names': CITY_SCENARIOS, 'kmax': kmax, 'per_city': args.per_city, 'seed': args.seed,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'pdp_pool.pt')
    torch.save(pool, path)
    n = pool['delay'].shape[0]
    size_mb = os.path.getsize(path) / 1e6
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump({'n_users': int(n), 'kmax': int(kmax), 'per_city': args.per_city,
                   'cities': CITY_SCENARIOS, 'per_city_counts': counts,
                   'file': 'pdp_pool.pt', 'source': 'deepmimo-ray-traced-pdp'}, f, indent=2)
    print(f"\nSaved PDP pool: {n} users (kmax={kmax}) -> {path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
