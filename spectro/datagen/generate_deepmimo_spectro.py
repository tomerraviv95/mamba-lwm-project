"""Generate a spectrogram dataset whose channels come from DeepMIMO city ray-tracing.

For each of the 20 LWM cities, randomly sample ``--per-city`` user PDPs (ray-traced
delay/power/phase/AoA). Each sampled PDP becomes one spectrogram: a random
(tech, modulation, SNR, mobility) OFDM waveform is synthesized, propagated through that user's
site-specific channel (TDL-style fading w/ per-ray Doppler), AWGN added, then STFT -> 128x128.

Output matches the synthetic corpus format (sharded dicts {tech,snr,mod,mob,city,data} +
manifest.json), so it is drop-in for ``spectro_pretrain.py --data synthetic`` / the sweep.

Usage::

    CUDA_VISIBLE_DEVICES=1 python spectro/datagen/generate_deepmimo_spectro.py --per-city 1000
    python spectro/datagen/generate_deepmimo_spectro.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sionna.phy.channel import (ApplyTimeChannel, cir_to_time_channel,  # noqa: E402
                                 time_lag_discrete_time_channel)
from deepmimo_channel import CITY_SCENARIOS, deepmimo_tdl_cir, extract_city_pdp  # noqa: E402
from phy_params import (CARRIER_FREQUENCY_HZ, MOBILITIES, MOBILITY_SPEED_MS, MOD_BITS,  # noqa: E402
                        MODULATIONS, PROTOCOL_CONFIGS, PROTOCOLS, SNRS_DB, snr_label)
from sionna_blocks import DEVICE, _BINARY_SOURCE, _ofdm_chain  # noqa: E402
from spectrogram import iq_batch_to_spectrogram  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_DEFAULT_OUT = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'spectro_deepmimo')


def build_pdp_pool(per_city, seed):
    """Sample ``per_city`` valid user PDPs from each city; return stacked, K-padded tables + city idx."""
    rng = np.random.RandomState(seed)
    parts = defaultdict(list)
    for ci, scn in enumerate(CITY_SCENARIOS):
        pdp = extract_city_pdp(scn, bs_idx=1)
        u = pdp['delay'].shape[0]
        idx = rng.permutation(u)[:min(per_city, u)]
        for k in ('delay', 'power_linear', 'phase', 'aoa_az'):
            parts[k].append(pdp[k][idx])
        parts['city'].append(np.full(len(idx), ci, dtype=np.int64))
        print(f"  {scn}: {u} valid users -> sampled {len(idx)}")
    kmax = max(a.shape[1] for a in parts['delay'])

    def padcat(key):
        return np.concatenate([np.pad(a, ((0, 0), (0, kmax - a.shape[1]))) for a in parts[key]], axis=0)
    return (padcat('delay'), padcat('power_linear'), padcat('phase'), padcat('aoa_az'),
            np.concatenate(parts['city']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-city', type=int, default=1000)
    ap.add_argument('--out', default=_DEFAULT_OUT)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--shard-size', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.per_city = 20

    os.makedirs(args.out, exist_ok=True)
    print(f"Building PDP pool ({args.per_city}/city, device={DEVICE}) ...")
    delays, powers, phases, aoas, city = build_pdp_pool(args.per_city, args.seed)
    n = delays.shape[0]

    # random (tech, mod, snr, mobility) per sample
    rng = np.random.RandomState(args.seed + 1)
    techs = rng.choice(PROTOCOLS, n)
    mods = rng.choice(MODULATIONS, n)
    snrs = rng.choice(SNRS_DB, n)
    mobs = rng.choice(MOBILITIES, n)

    # group by (tech, mod) so each batch shares an OFDM waveform config
    groups = defaultdict(list)
    for i in range(n):
        groups[(techs[i], mods[i])].append(i)

    print(f"Generating {n} spectrograms over {len(groups)} (tech,mod) groups ...")
    buffer, shard_idx, made, shard_paths = [], 0, 0, []
    t0 = time.time()
    for (tech, mod), idxs in groups.items():
        cfg = PROTOCOL_CONFIGS[tech]
        sr = cfg.sample_rate
        rg, mapper, rg_mapper, modulator = _ofdm_chain(tech, mod)
        l_min, l_max = time_lag_discrete_time_channel(sr)
        l_tot = l_max - l_min + 1
        apply = None
        for s in range(0, len(idxs), args.batch):
            bi = idxs[s:s + args.batch]
            b = len(bi)
            bits = _BINARY_SOURCE([b, 1, 1, int(rg.num_data_symbols * MOD_BITS[mod])])
            x = modulator(rg_mapper(mapper(bits)))            # (b,1,1,T)
            num_time = x.shape[-1]
            if apply is None:
                apply = ApplyTimeChannel(num_time, l_tot=l_tot, add_awgn=False).to(DEVICE)
            speeds = np.array([MOBILITY_SPEED_MS[mobs[i]] for i in bi], dtype=np.float32)
            a, tau = deepmimo_tdl_cir(delays[bi], powers[bi], phases[bi], aoas[bi], speeds,
                                      num_time + l_tot - 1, sr, fc=CARRIER_FREQUENCY_HZ, device=DEVICE)
            h = cir_to_time_channel(sr, a, tau, l_min=l_min, l_max=l_max, normalize=True)
            y = apply(x, h).reshape(b, -1)                    # (b, T') complex, no AWGN yet
            # per-sample AWGN at each sample's target SNR
            p = y.abs().pow(2).mean(dim=1, keepdim=True)
            snr_lin = torch.tensor([10.0 ** (snrs[i] / 10.0) for i in bi],
                                   device=y.device).reshape(b, 1)
            no = p / snr_lin
            noise = torch.sqrt(no / 2) * torch.complex(torch.randn_like(y.real), torch.randn_like(y.real))
            specs = iq_batch_to_spectrogram(y + noise).cpu()  # (b,1,128,128) float16
            for j, i in enumerate(bi):
                buffer.append({'tech': tech, 'snr': snr_label(int(snrs[i])), 'mod': mod,
                               'mob': mobs[i], 'city': CITY_SCENARIOS[city[i]], 'data': specs[j]})
            made += b
            while len(buffer) >= args.shard_size:
                p_ = os.path.join(args.out, f'shard_{shard_idx:04d}.pt')
                torch.save(buffer[:args.shard_size], p_); shard_paths.append(os.path.basename(p_))
                print(f"  shard {shard_idx:04d}: {made}/{n} ({made/n*100:.1f}%, {time.time()-t0:.0f}s)")
                buffer, shard_idx = buffer[args.shard_size:], shard_idx + 1
    if buffer:
        p_ = os.path.join(args.out, f'shard_{shard_idx:04d}.pt')
        torch.save(buffer, p_); shard_paths.append(os.path.basename(p_))

    manifest = {'n_samples': made, 'shards': shard_paths, 'shard_size': args.shard_size,
                'per_city': args.per_city, 'cities': CITY_SCENARIOS, 'seed': args.seed,
                'source': 'deepmimo-channel-spectrograms',
                'note': 'OFDM waveforms through DeepMIMO ray-traced channels (delay/power/AoA) + AWGN -> STFT.'}
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone: {made} spectrograms in {len(shard_paths)} shards -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
