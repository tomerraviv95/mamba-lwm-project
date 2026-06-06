"""Generate a synthetic spectrogram corpus mirroring the demo label grid (scaled).

Sweeps (tech x modulation x snr x mobility), generates ``--per-combo`` samples each via the
Sionna chain, and writes sharded ``.pt`` files (lists of demo-format dicts) plus a
``manifest.json``. The output is consumable by ``spectro_data`` (minus precomputed embeddings).

Usage::

    python spectro/datagen/generate.py --per-combo 500            # ~157k samples
    python spectro/datagen/generate.py --per-combo 4 --smoke      # tiny sanity run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phy_params import (MOBILITIES, MODULATIONS, PROTOCOL_CONFIGS, PROTOCOLS, SNRS_DB,
                        snr_label)
from sionna_blocks import generate_iq_batch
from spectrogram import iq_batch_to_spectrogram

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_DEFAULT_OUT = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'synthetic')


def _make_combo_samples(tech, modulation, snr_db, mobility, n, batch):
    """Generate ``n`` samples for one label combo, in batches of ``batch``. Returns a list of dicts."""
    cfg = PROTOCOL_CONFIGS[tech]
    label = {'tech': tech, 'snr': snr_label(snr_db), 'mod': modulation, 'mob': mobility}
    out = []
    remaining = n
    while remaining > 0:
        b = min(batch, remaining)
        iq = generate_iq_batch(cfg, modulation, snr_db=snr_db, mobility=mobility, n=b)
        specs = iq_batch_to_spectrogram(iq).cpu()        # (b,1,128,128) float16
        for i in range(b):
            out.append({**label, 'data': specs[i]})
        remaining -= b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-combo', type=int, default=500,
                    help='samples per (tech,mod,snr,mobility) combo.')
    ap.add_argument('--out', default=_DEFAULT_OUT)
    ap.add_argument('--shard-size', type=int, default=2000, help='samples per .pt shard.')
    ap.add_argument('--batch', type=int, default=64,
                    help='per-combo generation batch (keep modest if sharing the GPU).')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true',
                    help='tiny grid (1 tech-subset) for a fast sanity run.')
    ap.add_argument('--techs', nargs='*', default=None)
    ap.add_argument('--mods', nargs='*', default=None)
    args = ap.parse_args()

    techs = args.techs or PROTOCOLS
    mods = args.mods or MODULATIONS
    snrs = SNRS_DB
    mobs = MOBILITIES
    if args.smoke:
        techs, mods, snrs, mobs = PROTOCOLS, ['QPSK'], [0, 20], ['static', 'vehicular']

    os.makedirs(args.out, exist_ok=True)
    total = len(techs) * len(mods) * len(snrs) * len(mobs) * args.per_combo
    print(f"Generating {total} samples -> {args.out} "
          f"({len(techs)}t x {len(mods)}m x {len(snrs)}snr x {len(mobs)}mob x {args.per_combo})")

    from sionna_blocks import DEVICE
    print(f"  device={DEVICE}, batch={args.batch}")
    torch.manual_seed(args.seed)
    buffer, shard_idx, made = [], 0, 0
    shard_paths = []
    t0 = time.time()
    for tech in techs:
        for modulation in mods:
            for snr_db in snrs:
                for mobility in mobs:
                    buffer.extend(_make_combo_samples(tech, modulation, snr_db, mobility,
                                                      args.per_combo, args.batch))
                    made += args.per_combo
                    while len(buffer) >= args.shard_size:
                        p = os.path.join(args.out, f'shard_{shard_idx:04d}.pt')
                        torch.save(buffer[:args.shard_size], p)
                        shard_paths.append(os.path.basename(p))
                        print(f"  shard {shard_idx:04d}: {made}/{total} "
                              f"({made/total*100:.1f}%, {time.time()-t0:.0f}s)")
                        buffer, shard_idx = buffer[args.shard_size:], shard_idx + 1
    if buffer:
        p = os.path.join(args.out, f'shard_{shard_idx:04d}.pt')
        torch.save(buffer, p); shard_paths.append(os.path.basename(p))
        print(f"  shard {shard_idx:04d}: {made}/{total} (100%, {time.time()-t0:.0f}s)")

    manifest = {
        'n_samples': made, 'shards': shard_paths, 'shard_size': args.shard_size,
        'grid': {'techs': techs, 'mods': mods, 'snrs_db': snrs, 'mobilities': mobs,
                 'per_combo': args.per_combo},
        'seed': args.seed, 'source': 'sionna-synthetic',
        'note': 'Approximate OFDM (Sionna PHY); LTE/WiFi via numerology, not spec-compliant.',
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Done: {made} samples in {len(shard_paths)} shards -> {args.out}/manifest.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
