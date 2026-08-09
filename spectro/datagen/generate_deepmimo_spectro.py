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
from deepmimo_channel import (CITY_SCENARIOS, deepmimo_tdl_cir, extract_city_pdp,  # noqa: E402
                              sc_channel_apply)
from phy_params import (CARRIER_FREQUENCY_HZ, MOBILITIES, MOBILITY_SPEED_MS, MOBILITY_SPEED_RANGE,  # noqa: E402
                        MOD_BITS, MODULATIONS, PROTOCOL_CONFIGS, PROTOCOLS, SC_CONFIGS, SNRS_DB,
                        snr_label)
from pulse_shaping import apply_freq_offset, matched_filter, pulse_shape  # noqa: E402
from sionna.phy.mapping import Mapper  # noqa: E402
from sionna.phy.ofdm import OFDMDemodulator  # noqa: E402
from sionna_blocks import DEVICE, _BINARY_SOURCE, _ofdm_chain  # noqa: E402
from spectrogram import (iq_batch_to_spectrogram, iq_batch_to_complex_spectrogram,  # noqa: E402
                         grid_mag_to_spectrogram, grid_complex_to_spectrogram,
                         sc_power_spectrogram)

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_DEFAULT_OUT = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'spectro_deepmimo')


def user_split_mask(user_idx, city_i, split_frac, split_part, split_seed=777):
    """BS-INDEPENDENT train/downstream mask for a city's users, keyed on the RAW grid index.

    A city's valid-user SET differs per BS (a UE with no link to BS1 may have one to BS3), so
    permuting each BS's filtered list separately would put the same physical location in `train`
    for one BS and in `downstream` for another -- leakage that no split-seed can fix. Hashing the
    raw grid index instead gives every physical user one side of the split, for every BS, in every
    run, with no dependence on load order or on how many BSs are used.
    """
    h = ((user_idx.astype(np.uint64) + np.uint64(city_i) * np.uint64(1000003) +
          np.uint64(split_seed)) * np.uint64(2654435761)) % np.uint64(2 ** 32)
    u = h.astype(np.float64) / float(2 ** 32)
    return (u < split_frac) if split_part == 'train' else (u >= split_frac)


def build_pdp_pool(per_city, seed, cities=None, all_users=False,
                   split_frac=None, split_part='train', split_seed=777, bs_list=None):
    """Sample user PDPs from each city (optionally across several BS positions).

    ``cities``: list of scenario names (or (name, bs) tuples) to use instead of ``CITY_SCENARIOS``.
    ``bs_list``: BS/TX-set ids to pool per scenario -- the LWM cities each expose THREE (BS1/BS2/BS3,
    ids 1/2/3) and using only one discards ~2/3 of the available channel geometry. Each (user, BS)
    pair is an independent propagation link, so pooling them multiplies CHANNEL diversity, which is
    the axis that raw sample count cannot buy.
    ``all_users``: use every valid user rather than sampling ``per_city``.
    ``split_frac``: user-level train/downstream partition, applied via ``user_split_mask``.
    """
    scenarios = cities or [(s, 1) for s in CITY_SCENARIOS]
    rng = np.random.RandomState(seed)
    parts = defaultdict(list)
    for ci, item in enumerate(scenarios):
        scn, bs_default = item if isinstance(item, (tuple, list)) else (item, 1)
        bss = bs_list if bs_list else [bs_default]
        for bs in bss:
            try:
                pdp = extract_city_pdp(scn, bs_idx=bs)
            except Exception as e:
                print(f"  {scn} bs={bs}: SKIP ({type(e).__name__}: {e})")
                continue
            keep = np.arange(pdp['delay'].shape[0])
            if split_frac is not None:
                m = user_split_mask(pdp['user_idx'], ci, split_frac, split_part, split_seed)
                keep = keep[m]
            if not all_users:
                take = min(per_city, len(keep))
                keep = np.sort(keep[rng.permutation(len(keep))[:take]])
            if len(keep) == 0:
                continue
            for k in ('delay', 'power_linear', 'phase', 'aoa_az'):
                parts[k].append(pdp[k][keep])
            parts['city'].append(np.full(len(keep), ci, dtype=np.int64))
            parts['bs'].append(np.full(len(keep), bs, dtype=np.int64))
            parts['user'].append(pdp['user_idx'][keep])
            tag = f" [{split_part} {split_frac:.0%}]" if split_frac is not None else ""
            print(f"  {scn} bs={bs}: {pdp['delay'].shape[0]} valid -> using {len(keep)}{tag}", flush=True)
            del pdp
    if not parts['delay']:
        raise RuntimeError("no usable (scenario, bs) combinations")
    kmax = max(a.shape[1] for a in parts['delay'])

    def padcat(key):
        return np.concatenate([np.pad(a, ((0, 0), (0, kmax - a.shape[1]))) for a in parts[key]], axis=0)
    return (padcat('delay'), padcat('power_linear'), padcat('phase'), padcat('aoa_az'),
            np.concatenate(parts['city']), np.concatenate(parts['bs']), np.concatenate(parts['user']))


def _generate_sc_group(tech, mod, idxs, args, rng, delays, powers, phases, aoas, city,
                       mobs, snrs, snr_labels, used_cities, buffer):
    """Single-carrier generation for one (tech, modulation) group -> appends dicts to ``buffer``.

    Chain (LWM-Spectro sec. II-A):
        bits -> Mapper -> upsample x N_os + RRC pulse shape (eq. 4)
             -> random in-band frequency offset
             -> DeepMIMO ray-traced multipath with per-tap 3GPP-style sub-ray fading (eq. 5-6)
             -> AWGN at the sample's SNR
             -> receive matched filter
             -> |STFT|^2 -> log -> corpus-normalized 128x128 (eq. 7-10)
    """
    cfg = SC_CONFIGS[tech]
    bits_n = MOD_BITS[mod]
    mapper = Mapper("pam" if mod == "BPSK" else "qam", bits_n).to(DEVICE)
    room = max(0.0, 1.0 - cfg.occupied_frac) / 2.0
    for s in range(0, len(idxs), args.batch):
        bi = idxs[s:s + args.batch]
        b = len(bi)
        bits = _BINARY_SOURCE([b, 1, 1, cfg.num_symbols * bits_n])
        x = pulse_shape(mapper(bits).reshape(b, -1), cfg.sps, cfg.rolloff, cfg.span_symbols)
        if not args.no_freq_jitter:
            x = apply_freq_offset(x, torch.as_tensor(rng.uniform(-room, room, b), device=x.device))
        if args.vary_speed:
            speeds = np.array([rng.uniform(*MOBILITY_SPEED_RANGE[mobs[i]]) for i in bi], np.float32)
        else:
            speeds = np.array([MOBILITY_SPEED_MS[mobs[i]] for i in bi], np.float32)
        y = sc_channel_apply(x, delays[bi], powers[bi], phases[bi], aoas[bi], speeds,
                             cfg.sample_rate, fc=CARRIER_FREQUENCY_HZ, rng=rng)
        pw = y.abs().pow(2).mean(dim=1, keepdim=True)
        snr_lin = torch.as_tensor([10.0 ** (snrs[i] / 10.0) for i in bi],
                                  device=y.device, dtype=torch.float32).reshape(b, 1)
        no = pw / snr_lin
        y = y + torch.sqrt(no / 2) * torch.complex(torch.randn_like(y.real), torch.randn_like(y.real))
        y = matched_filter(y, cfg.sps, cfg.rolloff, cfg.span_symbols)
        specs = sc_power_spectrogram(y, n_fft=128, win_length=args.sc_win, out_size=128,
                                     norm=args.sc_norm).cpu()
        for j, i in enumerate(bi):
            buffer.append({'tech': tech, 'snr': snr_label(snr_labels[i]), 'mod': mod,
                           'mob': mobs[i], 'city': used_cities[city[i]],
                           'bs': int(args.bsid[i]), 'user': int(args.uidx[i]), 'data': specs[j]})
        del x, y, specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-city', type=int, default=1000)
    ap.add_argument('--all-users', action='store_true',
                    help='use EVERY valid user in each city (ignore --per-city). Maximizes channel '
                         'diversity — the 20 cities have ~156k users total vs the 40k we sampled before.')
    ap.add_argument('--user-split-frac', type=float, default=None,
                    help='partition each city\'s users into this train fraction + complement (disjoint '
                         'at the user level, same cities). Use with --user-split-part.')
    ap.add_argument('--user-split-part', choices=['train', 'downstream'], default='train',
                    help='which side of the --user-split-frac partition to emit (pretrain uses train, '
                         'the downstream eval uses downstream). Fixed split seed -> the two are disjoint.')
    ap.add_argument('--snr-range', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                    help='sample SNR CONTINUOUSLY ~U(MIN,MAX) dB instead of the 7 discrete SNRS_DB '
                         'values (wider input-condition diversity for pretraining; SNR is not a '
                         'pretrain label). Stored label is snapped to the nearest SNRS_DB bin.')
    ap.add_argument('--out', default=_DEFAULT_OUT)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--shard-size', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--vary-speed', action='store_true',
                    help='sample each sample speed from a per-class RANGE (MOBILITY_SPEED_RANGE) instead '
                         'of a fixed value -> broader mobility/Doppler distribution in train (helps the '
                         'train mobility signature overlap the test distribution).')
    ap.add_argument('--symbol-mult', type=int, default=1,
                    help='multiply OFDM symbols per burst -> longer slow-time window so Doppler/mobility '
                         'shows across STFT frames (burst must exceed Doppler coherence time). Memory '
                         'scales with this; lower --batch accordingly.')
    ap.add_argument('--complex', action='store_true',
                    help='store (2,128,128) [real,imag] complex spectrograms (element_length=32) '
                         'instead of (1,128,128) magnitude — the authors\' contrastive representation')
    ap.add_argument('--repr', choices=['stft', 'grid', 'grid_stft', 'grid_complex'], default='stft',
                    help="spectrogram representation. 'stft' = |STFT| of the time-domain OFDM waveform "
                         "(modulation NOT encoded — OFDM averages the constellation away). 'grid' = "
                         "|demodulated received resource grid| (subcarrier x symbol) where modulation "
                         "order IS PARTLY visible (amplitude only; BPSK/QPSK confusable). 'grid_complex' = "
                         "2-channel [Re(Y),Im(Y)] of that grid — KEEPS PHASE so the full constellation "
                         "(incl BPSK vs QPSK) is separable. Use 'grid' for the mod task, 'grid_complex' "
                         "for the phase-bearing IQ study.")
    ap.add_argument('--bs-list', nargs='+', default=None,
                    help='BS/TX-set ids to pool per scenario (LWM cities expose 1 2 3). Using only '
                         'one discards ~2/3 of the available channel geometry: 20 cities x 3 BS is '
                         '~346k unique (user, BS) links vs 156k at bs=1. The train/eval user split '
                         'is keyed on the RAW grid index so it stays consistent across BSs.')
    ap.add_argument('--draws', type=int, default=1,
                    help='independent samples PER USER. The DeepMIMO ray tables (the expensive part) '
                         'are reused, but each draw re-samples (tech, mod, snr, mobility), the payload '
                         'bits, the AWGN and the per-tap sub-ray phases -- so draws share a channel '
                         'geometry but are otherwise independent. Use to scale the corpus without '
                         'more ray-tracing: --all-users --draws 3 -> ~398k from the 132,748 users.')
    ap.add_argument('--waveform', choices=['ofdm', 'sc'], default='ofdm',
                    help="transmit waveform. 'sc' = SINGLE-CARRIER pulse-shaped (LWM-Spectro eq. 4, "
                         "what the paper actually uses) -> modulation survives in the MAGNITUDE "
                         "spectrogram. 'ofdm' = the legacy OFDM chain, where summing 52-624 "
                         "subcarriers makes the time signal ~Gaussian by CLT and erases the "
                         "constellation. Use 'sc'; 'ofdm' is kept only to reproduce old corpora.")
    ap.add_argument('--sc-win', type=int, default=8,
                    help='STFT analysis window in SAMPLES for --waveform sc (zero-padded to 128 '
                         'bins). Short = modulation visible, long = better SNR/Doppler but the '
                         'guard-band shortcut returns. 8 (=4 symbols) is the measured optimum.')
    ap.add_argument('--sc-norm', choices=['global', 'sample', 'none'], default='global',
                    help="spectrogram normalization. 'global' = corpus-level (the paper's "
                         "'normalize with pretrained statistics'), which PRESERVES the dB variance "
                         "that carries modulation. 'sample' = legacy per-sample z-score, which "
                         "divides it out.")
    ap.add_argument('--no-freq-jitter', action='store_true',
                    help='disable the random in-band frequency offset. Leave it ON: without it the '
                         'occupied band sits at fixed bins and 7-way SNR collapses to a guard-band '
                         'brightness ratio (measured 27-60 sigma on the OFDM corpus, <1 with it).')
    ap.add_argument('--cities', default=None,
                    help='comma-separated scenario names (optionally name:bs_idx) to use instead of the '
                         'default 20 CITY_SCENARIOS. Held-out cross-environment eval set with their BS sets: '
                         'asu_campus_3p5:1,boston5g_3p5:2,o1_3p5:3 (each scenario exposes a different TX set).')
    args = ap.parse_args()
    cities = None
    if args.cities:
        cities = []
        for c in args.cities.split(','):
            c = c.strip()
            name, bs = (c.rsplit(':', 1)[0], int(c.rsplit(':', 1)[1])) if ':' in c else (c, 1)
            cities.append((name, bs))
    if args.smoke:
        args.per_city = 20

    os.makedirs(args.out, exist_ok=True)
    used_cities = [c[0] for c in cities] if cities else CITY_SCENARIOS
    print(f"Building PDP pool ({args.per_city}/city x {len(used_cities)} cities, device={DEVICE}) ...")
    if cities:
        print(f"  cities override: {cities}")
    _bs = None
    if args.bs_list:
        _bs = [int(b) for b in args.bs_list]
    delays, powers, phases, aoas, city, bsid, uidx = build_pdp_pool(
        args.per_city, args.seed, cities, all_users=args.all_users,
        split_frac=args.user_split_frac, split_part=args.user_split_part, bs_list=_bs)
    if args.draws > 1:      # replicate the user pool; labels/noise/sub-rays are redrawn below
        delays = np.tile(delays, (args.draws, 1))
        powers = np.tile(powers, (args.draws, 1))
        phases = np.tile(phases, (args.draws, 1))
        aoas = np.tile(aoas, (args.draws, 1))
        city = np.tile(city, args.draws)
        bsid = np.tile(bsid, args.draws)
        uidx = np.tile(uidx, args.draws)
        print(f"  --draws {args.draws} -> {delays.shape[0]} samples from "
              f"{delays.shape[0] // args.draws} unique users")
    args.bsid, args.uidx = bsid, uidx      # AFTER tiling, so they match the replicated pool
    n = delays.shape[0]

    # Seed the TORCH global RNG too: _BINARY_SOURCE (payload bits) and torch.randn_like (AWGN)
    # both draw from it, so without this two runs with identical flags produce different data and
    # the corpus cannot be regenerated. --seed previously controlled only the numpy label/PDP draw.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # random (tech, mod, snr, mobility) per sample
    rng = np.random.RandomState(args.seed + 1)
    techs = rng.choice(PROTOCOLS, n)
    mods = rng.choice(MODULATIONS, n)
    if args.snr_range is not None:                      # continuous wide SNR (diversity), label -> nearest bin
        snrs = rng.uniform(args.snr_range[0], args.snr_range[1], n).astype(np.float32)
        _snr_bins = np.asarray(SNRS_DB, dtype=np.float32)
        snr_labels = [int(_snr_bins[int(np.argmin(np.abs(_snr_bins - v)))]) for v in snrs]
    else:
        snrs = rng.choice(SNRS_DB, n)
        snr_labels = [int(v) for v in snrs]
    mobs = rng.choice(MOBILITIES, n)

    # group by (tech, mod) so each batch shares an OFDM waveform config
    groups = defaultdict(list)
    for i in range(n):
        groups[(techs[i], mods[i])].append(i)

    print(f"Generating {n} spectrograms over {len(groups)} (tech,mod) groups ...")
    buffer, shard_idx, made, shard_paths = [], 0, 0, []
    t0 = time.time()
    if args.waveform == 'sc':
        # Emit batches in a globally SHUFFLED order across (tech, mod) groups. Generating group by
        # group makes every shard homogeneous in (tech, mod) -- harmless for the current full-load
        # path, but it silently turns any streaming / resume-from-shard-N / "load the first K shards
        # for a smoke test" into a corpus containing one or two modulations.
        tasks = [(tech, mod, idxs[s:s + args.batch])
                 for (tech, mod), idxs in groups.items()
                 for s in range(0, len(idxs), args.batch)]
        rng.shuffle(tasks)
        for tech, mod, bi in tasks:
            _generate_sc_group(tech, mod, bi, args, rng, delays, powers, phases, aoas, city,
                               mobs, snrs, snr_labels, used_cities, buffer)
            made += len(bi)
            while len(buffer) >= args.shard_size:
                p_ = os.path.join(args.out, f'shard_{shard_idx:04d}.pt')
                torch.save(buffer[:args.shard_size], p_); shard_paths.append(os.path.basename(p_))
                print(f"  shard {shard_idx:04d}: {made}/{n} ({made/n*100:.1f}%, {time.time()-t0:.0f}s)")
                del buffer[:args.shard_size]; shard_idx += 1
        groups = {}

    for (tech, mod), idxs in groups.items():
        cfg = PROTOCOL_CONFIGS[tech]
        sr = cfg.sample_rate
        rg, mapper, rg_mapper, modulator = _ofdm_chain(tech, mod, args.symbol_mult)
        l_min, l_max = time_lag_discrete_time_channel(sr)
        l_tot = l_max - l_min + 1
        apply = None
        demod = (OFDMDemodulator(cfg.fft_size, l_min, cfg.cyclic_prefix_length).to(DEVICE)
                 if args.repr in ('grid', 'grid_stft', 'grid_complex') else None)   # received-grid (modulation visible)
        for s in range(0, len(idxs), args.batch):
            bi = idxs[s:s + args.batch]
            b = len(bi)
            bits = _BINARY_SOURCE([b, 1, 1, int(rg.num_data_symbols * MOD_BITS[mod])])
            x = modulator(rg_mapper(mapper(bits)))            # (b,1,1,T)
            num_time = x.shape[-1]
            if apply is None:
                apply = ApplyTimeChannel(num_time, l_tot=l_tot, add_awgn=False).to(DEVICE)
            if args.vary_speed:
                speeds = np.array([rng.uniform(*MOBILITY_SPEED_RANGE[mobs[i]]) for i in bi], dtype=np.float32)
            else:
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
            yn = y + noise
            if args.repr == 'grid':                           # demod -> |received resource grid|
                Y = demod(yn.reshape(b, 1, 1, -1))            # (b,1,1,num_ofdm_symbols,fft_size)
                specs = grid_mag_to_spectrogram(Y.abs().reshape(b, -1, cfg.fft_size)).cpu()
            elif args.repr == 'grid_complex':                 # demod -> COMPLEX received grid [Re,Im] (keeps phase)
                Y = demod(yn.reshape(b, 1, 1, -1))
                specs = grid_complex_to_spectrogram(Y.reshape(b, -1, cfg.fft_size)).cpu()   # (b,2,128,128)
            elif args.repr == 'grid_stft':                    # 2ch [STFT (Doppler/mobility) | grid (modulation)]
                Y = demod(yn.reshape(b, 1, 1, -1))
                g = grid_mag_to_spectrogram(Y.abs().reshape(b, -1, cfg.fft_size))   # (b,1,128,128)
                st = iq_batch_to_spectrogram(yn)                                    # (b,1,128,128)
                specs = torch.cat([st, g], dim=1).cpu()       # (b,2,128,128): ch0=STFT, ch1=grid
            else:
                _spec_fn = iq_batch_to_complex_spectrogram if args.complex else iq_batch_to_spectrogram
                specs = _spec_fn(yn).cpu()                    # (b,1,128,128) mag or (b,2,128,128) complex
            for j, i in enumerate(bi):
                buffer.append({'tech': tech, 'snr': snr_label(snr_labels[i]), 'mod': mod,
                               'mob': mobs[i], 'city': used_cities[city[i]], 'data': specs[j]})
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
                'per_city': args.per_city, 'cities': used_cities, 'seed': args.seed,
                'complex': bool(args.complex), 'repr': args.repr, 'symbol_mult': args.symbol_mult,
                'vary_speed': bool(args.vary_speed),
                'snr_range': list(args.snr_range) if args.snr_range is not None else None,
                'all_users': bool(args.all_users), 'draws': args.draws,
                'unique_users': int(n // args.draws) if args.draws else n,
                'user_split_frac': args.user_split_frac, 'user_split_part': args.user_split_part,
                'waveform': args.waveform,
                'sc_win': args.sc_win if args.waveform == 'sc' else None,
                'sc_norm': args.sc_norm if args.waveform == 'sc' else None,
                'freq_jitter': (not args.no_freq_jitter) if args.waveform == 'sc' else None,
                'channels': 2 if (args.complex or args.repr in ('grid_stft', 'grid_complex'))
                            and args.waveform == 'ofdm' else 1,
                'sc_config': ({k: getattr(SC_CONFIGS['LTE'], k) for k in
                               ('sps', 'rolloff', 'span_symbols', 'num_symbols')}
                              if args.waveform == 'sc' else None),
                'source': 'deepmimo-channel-spectrograms',
                'note': ('Single-carrier pulse-shaped waveforms (LWM-Spectro eq. 4) through DeepMIMO '
                         'ray-traced channels with 3GPP sub-ray Doppler fading + AWGN -> |STFT|^2.'
                         if args.waveform == 'sc' else
                         'OFDM waveforms through DeepMIMO ray-traced channels (delay/power/AoA) + AWGN -> STFT.')}
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone: {made} spectrograms in {len(shard_paths)} shards -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
