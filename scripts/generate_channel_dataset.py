"""Generate the token-balanced DeepMIMO channel dataset (run locally, upload to HF).

Samples the 20 LWM cities so each contributes up to ~``tokens_per_city`` 4x4 patch-tokens
(default 500k -> ~10M total), pooling base stations TXset 1..3 to offset LoS-link dropout.
Splits each city's realizations 99/1 (train/val) — i.e. a per-city-stratified split — and saves
per-city ``{scenario}_{train,val}.pt`` channel shards + ``manifest.json``.

The cluster then just downloads this dataset (no DeepMIMO needed) and trains via
``scripts/pretrain_channel_sampled.py --dataset-dir ...``.

Usage::

    python scripts/generate_channel_dataset.py --out outputs/channel_dataset
    python scripts/generate_channel_dataset.py --smoke      # 2 small cities, fast sanity run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pretrain_channel_sampled as P  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens-per-city', type=int, default=500_000)
    ap.add_argument('--val-frac', type=float, default=0.01, help='per-city validation fraction (99/1).')
    ap.add_argument('--max-bs', type=int, default=P.N_BS, help='base stations to pool per city.')
    ap.add_argument('--out', default=os.path.join(_REPO_ROOT, 'outputs', 'channel_dataset'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true', help='2 small cities, tiny budget.')
    args = ap.parse_args()

    plan = P.plan_sampling(args.tokens_per_city)
    if args.smoke:
        plan = [p for p in plan if p[0] in ('city_0_newyork_3p5_lwm', 'city_6_miami_3p5_lwm')]
        plan = [(scn, a, s, min(n, 300), pp, min(n, 300) * pp) for scn, a, s, n, pp, _ in plan]

    os.makedirs(args.out, exist_ok=True)
    cities, tot_train_tok, tot_train, tot_val = [], 0, 0, 0
    print(f"Generating channel dataset -> {args.out} (target {args.tokens_per_city} tokens/city, "
          f"pool {args.max_bs} BS, {int((1-args.val_frac)*100)}/{int(args.val_frac*100)} split) ...")

    for i, (scn, n_ant, n_sub, n, pp, _) in enumerate(plan):
        ch, avail = P.gather_city_channels(scn, n_ant, n_sub, n, args.seed, max_bs=args.max_bs)
        m = ch.shape[0]
        perm = np.random.RandomState(args.seed + i).permutation(m)
        n_val = max(1, round(args.val_frac * m))
        val = ch[torch.as_tensor(np.sort(perm[:n_val]))]
        train = ch[torch.as_tensor(np.sort(perm[n_val:]))]
        tr_name, va_name = f'{scn}_train.pt', f'{scn}_val.pt'
        torch.save(train, os.path.join(args.out, tr_name))
        torch.save(val, os.path.join(args.out, va_name))

        train_tok = train.shape[0] * pp
        tot_train_tok += train_tok
        tot_train += train.shape[0]
        tot_val += val.shape[0]
        cities.append({
            'scenario': scn, 'n_ant': n_ant, 'n_subcarriers': n_sub, 'patches_per_realization': pp,
            'available_valid': int(avail), 'kept': int(m), 'target': int(n),
            'n_train': int(train.shape[0]), 'n_val': int(val.shape[0]),
            'train_shard': tr_name, 'val_shard': va_name, 'train_tokens': int(train_tok),
        })
        short = '' if m >= n else f'  (LoS-limited: {m}<{n})'
        print(f"  {scn:32s} {n_ant:3d}x{n_sub:<4d} avail={avail:6d} kept={m:6d} "
              f"train={train.shape[0]:6d} val={val.shape[0]:4d} tok={train_tok}{short}")

    manifest = {
        'tokens_per_city_target': args.tokens_per_city, 'val_frac': args.val_frac,
        'max_bs': args.max_bs, 'seed': args.seed, 'patch': P.PATCH,
        'element_length': P.ELEMENT_LENGTH, 'grid_idx': P.GRID_IDX,
        'n_train': tot_train, 'n_val': tot_val, 'train_tokens': tot_train_tok,
        'cities': cities,
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone: {tot_train} train + {tot_val} val realizations -> "
          f"~{tot_train_tok/1e6:.2f}M train tokens. Manifest: {args.out}/manifest.json")
    print(f"Upload with:  python spectro/scripts/hf_sync.py push-dataset "
          f"--repo <user>/lwm-channel-dataset --dir {args.out} --private")


if __name__ == '__main__':
    main()
