"""Prototype: spectrogram through generic TDL-A vs DeepMIMO site-specific channel.

Generates one OFDM waveform (fixed tech/mod) and passes it through (a) the current TDL-A channel
and (b) a sampled city user's ray-traced DeepMIMO channel, at static and vehicular mobility, then
STFTs each to a 128x128 spectrogram and saves a comparison grid.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('SPECTRO_DATAGEN_DEVICE', 'cpu')  # prototype on CPU for determinism

from sionna.phy.channel import (ApplyTimeChannel, cir_to_time_channel,  # noqa: E402
                                 time_lag_discrete_time_channel)
from sionna_blocks import _ofdm_chain, MOD_BITS, _BINARY_SOURCE  # noqa: E402
from phy_params import PROTOCOL_CONFIGS, MOBILITY_SPEED_MS, CARRIER_FREQUENCY_HZ  # noqa: E402
from spectrogram import iq_batch_to_spectrogram  # noqa: E402
from deepmimo_channel import extract_city_pdp, deepmimo_tdl_cir, sample_user_pdp  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


def make_waveform(tech, modulation, seed=0):
    torch.manual_seed(seed)
    rg, mapper, rg_mapper, modulator = _ofdm_chain(tech, modulation)
    bits = _BINARY_SOURCE([1, 1, 1, int(rg.num_data_symbols * MOD_BITS[modulation])])
    x_time = modulator(rg_mapper(mapper(bits)))          # (1,1,1,T)
    return x_time, rg


def apply_cir(x_time, a, tau, sample_rate, snr_db):
    l_min, l_max = time_lag_discrete_time_channel(sample_rate)
    l_tot = l_max - l_min + 1
    num_time = x_time.shape[-1]
    h = cir_to_time_channel(sample_rate, a, tau, l_min=l_min, l_max=l_max, normalize=True)
    no = torch.tensor(10.0 ** (-snr_db / 10.0), dtype=torch.float32)
    y = ApplyTimeChannel(num_time, l_tot=l_tot, add_awgn=True)(x_time, h, no)
    return y.reshape(1, -1).cpu()   # Sionna apply defaults to cuda:0; bring back for STFT/plot


def tdl_cir(tech, mobility, num_time, sample_rate):
    from sionna_blocks import _channel
    tdl, l_min, l_max, l_tot, _ = _channel(tech, mobility, num_time)
    cir = tdl(batch_size=1, num_time_steps=num_time + l_tot - 1, sampling_frequency=sample_rate)
    return cir  # (a, tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tech', default='LTE')
    ap.add_argument('--mod', default='QAM16')
    ap.add_argument('--scenario', default='city_0_newyork_3p5_lwm')
    ap.add_argument('--snr', type=float, default=20.0)
    ap.add_argument('--user', type=int, default=None, help='DeepMIMO user index (default: a strong-LoS one)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'plots',
                                                  'deepmimo_vs_tdl_spectrogram.png'))
    args = ap.parse_args()

    cfg = PROTOCOL_CONFIGS[args.tech]
    sample_rate = cfg.sample_rate
    x_time, _ = make_waveform(args.tech, args.mod, args.seed)
    num_time = x_time.shape[-1]
    l_min, l_max = time_lag_discrete_time_channel(sample_rate)
    n_ts = num_time + (l_max - l_min + 1) - 1

    print(f"Extracting PDP from {args.scenario} ...")
    pdp = extract_city_pdp(args.scenario, bs_idx=1)
    # pick a user with several paths for a visibly multipath channel
    u = args.user if args.user is not None else int(np.argmax(pdp['num_paths']))
    print(f"  user {u}: {pdp['num_paths'][u]} paths, "
          f"delays(ns)={np.round(pdp['delay'][u][:5]*1e9,1)}, "
          f"powers(lin)={np.array2string(pdp['power_linear'][u][:5], precision=2)}")
    one = sample_user_pdp(pdp, u)

    panels = []  # (title, spectrogram)
    for mob in ('static', 'vehicular'):
        speed = MOBILITY_SPEED_MS[mob]
        # TDL-A baseline
        a_t, tau_t = tdl_cir(args.tech, mob, num_time, sample_rate)
        y_tdl = apply_cir(x_time, a_t, tau_t, sample_rate, args.snr)
        panels.append((f'TDL-A | {mob}', iq_batch_to_spectrogram(y_tdl)[0, 0].float().numpy()))
        # DeepMIMO site-specific
        a_d, tau_d = deepmimo_tdl_cir(one['delay'], one['power_linear'], one['phase'],
                                      one['aoa_az'], speed, n_ts, sample_rate,
                                      fc=CARRIER_FREQUENCY_HZ)
        y_dm = apply_cir(x_time, a_d, tau_d, sample_rate, args.snr)
        panels.append((f'DeepMIMO {args.scenario.split("_")[1]} | {mob}',
                       iq_batch_to_spectrogram(y_dm)[0, 0].float().numpy()))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (title, spec) in zip(axes.ravel(), panels):
        ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{args.tech} / {args.mod} @ {args.snr:.0f} dB  —  TDL-A vs DeepMIMO channel')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(); plt.savefig(args.out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
