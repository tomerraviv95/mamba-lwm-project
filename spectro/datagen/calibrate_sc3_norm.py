"""Measure corpus-level normalization constants for the two symbol-domain channels of --sc-channels 3.

The spectrogram channel already has calibrated constants (SC_DB_MEAN/STD). The histogram and
block-power channels live on entirely different scales, so they need their own -- and they must be
FIXED constants rather than per-sample statistics, for the same reason the spectrogram's are: a
per-sample z-score divides out the very variance that carries the label.

Run after any change to the numerology, SNR grid, block/bin counts, or symbol count::

    CUDA_VISIBLE_DEVICES=0 python spectro/datagen/calibrate_sc3_norm.py --n 600

then paste the printed constants into spectrogram.py.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepmimo_channel import sc_channel_apply  # noqa: E402
from phy_params import (CARRIER_FREQUENCY_HZ, MOBILITIES, MOBILITY_SPEED_RANGE, MOD_BITS,  # noqa: E402
                        MODULATIONS, SC_CONFIGS, SNRS_DB)
from pulse_shaping import apply_freq_offset, matched_filter, pulse_shape  # noqa: E402
from sionna.phy.mapping import Mapper  # noqa: E402
from sionna_blocks import DEVICE, _BINARY_SOURCE  # noqa: E402
from spectrogram import sc_amp_hist_channels, symbol_decimate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--gpu-frac', type=float, default=0.45, help='cap GPU memory (shared machine)')
    args = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac, 0)

    from sweep_sc_config import load_pdp
    rng = np.random.RandomState(11)
    pdp = load_pdp(max(args.n, 64), rng)   # real ray-traced fading: block power stats depend on it
    n_pdp = pdp['delay'].shape[0]
    h_sum = h_sq = b_sum = b_sq = 0.0
    cnt = 0
    for tech in SC_CONFIGS:
        cfg = SC_CONFIGS[tech]
        per_tech = args.n // len(SC_CONFIGS)
        for mod in MODULATIONS:
            mapper = Mapper('pam' if mod == 'BPSK' else 'qam', MOD_BITS[mod]).to(DEVICE)
            room = max(0.0, 1.0 - cfg.occupied_frac) / 2.0
            n_mod = max(args.batch, per_tech // len(MODULATIONS))
            for s in range(0, n_mod, args.batch):
                b = min(args.batch, n_mod - s)
                bits = _BINARY_SOURCE([b, 1, 1, cfg.num_symbols * MOD_BITS[mod]])
                x = pulse_shape(mapper(bits).reshape(b, -1), cfg.sps, cfg.rolloff, cfg.span_symbols)
                x = apply_freq_offset(x, torch.as_tensor(rng.uniform(-room, room, b), device=x.device))
                pi = rng.randint(0, n_pdp, b)
                mob = rng.choice(MOBILITIES, b)
                speeds = np.array([rng.uniform(*MOBILITY_SPEED_RANGE[m]) for m in mob], np.float32)
                y = sc_channel_apply(x, pdp['delay'][pi], pdp['power_linear'][pi], pdp['phase'][pi],
                                     pdp['aoa_az'][pi], speeds, cfg.sample_rate,
                                     fc=CARRIER_FREQUENCY_HZ, rng=rng)
                pw = y.abs().pow(2).mean(dim=1, keepdim=True)
                snr = rng.choice(SNRS_DB, b)
                snr_lin = torch.as_tensor(10.0 ** (snr / 10.0), device=y.device,
                                          dtype=torch.float32).reshape(b, 1)
                y = y + torch.sqrt(pw / snr_lin / 2) * torch.complex(torch.randn_like(y.real),
                                                                     torch.randn_like(y.real))
                y = matched_filter(y, cfg.sps, cfg.rolloff, cfg.span_symbols)
                ch = sc_amp_hist_channels(symbol_decimate(y, cfg.sps), norm='none').float()
                h, blk = ch[:, 0], ch[:, 1]
                h_sum += h.sum().item(); h_sq += h.pow(2).sum().item()
                b_sum += blk.sum().item(); b_sq += blk.pow(2).sum().item()
                cnt += h.numel()
                del x, y, ch
        print(f'  {tech} done ({cnt} elements)', flush=True)

    h_m = h_sum / cnt; h_s = (h_sq / cnt - h_m ** 2) ** 0.5
    b_m = b_sum / cnt; b_s = (b_sq / cnt - b_m ** 2) ** 0.5
    print('\npaste into spectro/datagen/spectrogram.py:')
    print(f'SC_HIST_MEAN = {h_m:.4f}')
    print(f'SC_HIST_STD = {h_s:.4f}')
    print(f'SC_BLKDB_MEAN = {b_m:.2f}')
    print(f'SC_BLKDB_STD = {b_s:.2f}')


if __name__ == '__main__':
    main()
