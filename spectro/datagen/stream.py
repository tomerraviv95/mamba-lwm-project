"""On-the-fly spectrogram streaming from a PDP pool (no stored dataset).

Given the compact PDP pool (build_pdp_pool.py), each call synthesizes a batch of spectrograms:
random OFDM waveform → a sampled city user's DeepMIMO channel (per-ray Doppler) → AWGN → STFT.
Used by spectro_pretrain_stream.py to pretrain on ~10M generated-on-the-fly spectrograms.

A batch fixes one (tech, mod) — OFDM length must be uniform — but draws PDPs from any city and
per-sample SNR/mobility. For expert pretraining we fix tech = the expert's protocol; for router
training we vary tech across batches and label by tech.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sionna.phy.channel import (ApplyTimeChannel, cir_to_time_channel,  # noqa: E402
                                 time_lag_discrete_time_channel)
from deepmimo_channel import deepmimo_tdl_cir  # noqa: E402
from phy_params import (CARRIER_FREQUENCY_HZ, MOBILITIES, MOBILITY_SPEED_MS, MOD_BITS,  # noqa: E402
                        MODULATIONS, PROTOCOL_CONFIGS, PROTOCOLS, SNRS_DB)
from sionna_blocks import DEVICE, _BINARY_SOURCE, _ofdm_chain  # noqa: E402
from spectro_patchify import build_masked_tensors, spectrogram_patchify, CLS_TOKEN  # noqa: E402
from spectrogram import iq_batch_to_spectrogram  # noqa: E402


def load_pool(path):
    """Load a pdp_pool.pt (or a directory containing it)."""
    if os.path.isdir(path):
        path = os.path.join(path, 'pdp_pool.pt')
    return torch.load(path, weights_only=False)


_APPLY_CACHE = {}


def _apply_layer(num_time, l_tot):
    key = (num_time, l_tot)
    if key not in _APPLY_CACHE:
        _APPLY_CACHE[key] = ApplyTimeChannel(num_time, l_tot=l_tot, add_awgn=False).to(DEVICE)
    return _APPLY_CACHE[key]


def gen_spectro_batch(pool, tech, b, rng, device=DEVICE, mod=None):
    """Generate ``b`` spectrograms for a fixed ``tech`` (random mod unless given) from pool PDPs.

    Returns (specs (b,128,128) float32 on cpu, snr_labels list[int], mob_labels list[str]).
    """
    cfg = PROTOCOL_CONFIGS[tech]
    sr = cfg.sample_rate
    mod = mod or MODULATIONS[rng.randint(len(MODULATIONS))]
    rg, mapper, rg_mapper, modulator = _ofdm_chain(tech, mod)

    n_pool = pool['delay'].shape[0]
    idx = rng.randint(0, n_pool, size=b)
    snrs = [int(SNRS_DB[i]) for i in rng.randint(0, len(SNRS_DB), size=b)]
    mobs = [MOBILITIES[i] for i in rng.randint(0, len(MOBILITIES), size=b)]
    speeds = np.array([MOBILITY_SPEED_MS[m] for m in mobs], dtype=np.float32)

    bits = _BINARY_SOURCE([b, 1, 1, int(rg.num_data_symbols * MOD_BITS[mod])])
    x = modulator(rg_mapper(mapper(bits)))                       # (b,1,1,T)
    num_time = x.shape[-1]
    l_min, l_max = time_lag_discrete_time_channel(sr)
    l_tot = l_max - l_min + 1

    a, tau = deepmimo_tdl_cir(pool['delay'][idx].numpy(), pool['power_linear'][idx].numpy(),
                              pool['phase'][idx].numpy(), pool['aoa_az'][idx].numpy(),
                              speeds, num_time + l_tot - 1, sr, fc=CARRIER_FREQUENCY_HZ, device=device)
    h = cir_to_time_channel(sr, a, tau, l_min=l_min, l_max=l_max, normalize=True)
    y = _apply_layer(num_time, l_tot)(x, h).reshape(b, -1)
    p = y.abs().pow(2).mean(dim=1, keepdim=True)
    snr_lin = torch.tensor([10.0 ** (s / 10.0) for s in snrs], device=y.device).reshape(b, 1)
    no = p / snr_lin
    y = y + torch.sqrt(no / 2) * torch.complex(torch.randn_like(y.real), torch.randn_like(y.real))
    specs = iq_batch_to_spectrogram(y).cpu().float().squeeze(1)   # (b,128,128)
    return specs, snrs, mobs


def gen_masked_batch(pool, tech, b, mask_percent, rng, device=DEVICE):
    """Generate a masked-spectrogram batch for expert pretraining of one protocol.

    Returns (input_ids (b,1025,16), masked_tokens (b,n_masks,16), masked_pos (b,n_masks)) on device.
    """
    specs, _, _ = gen_spectro_batch(pool, tech, b, rng, device=device)
    ids, toks, pos = build_masked_tensors(specs, mask_percent=mask_percent, seed=int(rng.randint(1 << 30)))
    return ids.to(device), toks.to(device), pos.to(device)


def gen_router_batch(pool, b, rng, device=DEVICE):
    """Generate a batch mixing protocols for router training. Returns (specs (b,128,128), tech_idx (b,))."""
    # one tech per call keeps OFDM uniform; cycle techs across calls for balance
    t = rng.randint(len(PROTOCOLS))
    tech = PROTOCOLS[t]
    specs, _, _ = gen_spectro_batch(pool, tech, b, rng, device=device)
    labels = torch.full((b,), t, dtype=torch.long)
    return specs, labels
