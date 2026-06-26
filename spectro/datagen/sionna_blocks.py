"""Thin builders around Sionna PHY for the synthetic spectrogram generator.

Delegates all DSP to Sionna 2.x (PyTorch-native): constellation mapping, OFDM modulation,
3GPP TDL fading with Doppler, and AWGN. We own only the per-protocol numerology (phy_params)
and the spectrogram step.

Chain: BinarySource -> Mapper -> ResourceGridMapper -> OFDMModulator (time domain)
       -> TDL CIR -> cir_to_time_channel -> ApplyTimeChannel(+AWGN) -> 1-D complex I/Q.
"""
from __future__ import annotations

import functools
import os

import torch

# Device for generation. Sionna requires an explicit index ('cuda:0'), not bare 'cuda'.
# Defaults to GPU when available; override with SPECTRO_DATAGEN_DEVICE (e.g. 'cpu' or 'cuda:1').
# Tip: launch with CUDA_VISIBLE_DEVICES=<free gpu> so 'cuda:0' maps to the idle GPU.
DEVICE = os.environ.get("SPECTRO_DATAGEN_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")

from sionna.phy.channel import (ApplyTimeChannel, cir_to_time_channel,
                                 time_lag_discrete_time_channel)
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.ofdm import OFDMModulator, ResourceGrid, ResourceGridMapper

from phy_params import (CARRIER_FREQUENCY_HZ, MOBILITY_SPEED_MS, MOD_BITS, ProtocolConfig)

_BINARY_SOURCE = BinarySource().to(DEVICE)


def build_resource_grid(cfg: ProtocolConfig, symbol_mult: int = 1) -> ResourceGrid:
    """No-pilot OFDM resource grid for a protocol (all subcarriers carry data; DC nulled).

    ``symbol_mult`` lengthens the burst (more OFDM symbols) so the STFT window spans enough slow-time
    for Doppler/mobility to appear across frames (burst must exceed the Doppler coherence time).
    """
    return ResourceGrid(
        num_ofdm_symbols=cfg.num_ofdm_symbols * symbol_mult,
        fft_size=cfg.fft_size,
        subcarrier_spacing=cfg.subcarrier_spacing,
        cyclic_prefix_length=cfg.cyclic_prefix_length,
        num_guard_carriers=cfg.num_guard_carriers,
        dc_null=True,
    )


@functools.lru_cache(maxsize=None)
def _ofdm_chain(tech: str, modulation: str, symbol_mult: int = 1):
    """Cache the (resource grid, mapper, rg-mapper, modulator) for a (tech, modulation, symbol_mult)."""
    from phy_params import PROTOCOL_CONFIGS
    cfg = PROTOCOL_CONFIGS[tech]
    rg = build_resource_grid(cfg, symbol_mult)
    bits = MOD_BITS[modulation]
    ctype = "pam" if modulation == "BPSK" else "qam"   # BPSK -> 2-PAM (1 bit), else QAM
    mapper = Mapper(ctype, bits).to(DEVICE)
    rg_mapper = ResourceGridMapper(rg).to(DEVICE)
    modulator = OFDMModulator(rg.cyclic_prefix_length).to(DEVICE)
    return rg, mapper, rg_mapper, modulator


@functools.lru_cache(maxsize=None)
def _channel(tech: str, mobility: str, num_time: int):
    """Cache (TDL, l_min, l_max, l_tot, ApplyTimeChannel) for a (tech, mobility, num_time)."""
    from phy_params import PROTOCOL_CONFIGS
    sample_rate = PROTOCOL_CONFIGS[tech].sample_rate
    l_min, l_max = time_lag_discrete_time_channel(sample_rate)
    l_tot = l_max - l_min + 1
    speed = MOBILITY_SPEED_MS[mobility]
    tdl = TDL("A", delay_spread=100e-9, carrier_frequency=CARRIER_FREQUENCY_HZ,
              min_speed=speed, max_speed=speed, device=DEVICE)
    apply = ApplyTimeChannel(num_time, l_tot=l_tot, add_awgn=True).to(DEVICE)
    return tdl, l_min, l_max, l_tot, apply


def generate_iq_batch(cfg: ProtocolConfig, modulation: str, snr_db: float, mobility: str,
                      n: int = 1) -> torch.Tensor:
    """Run the full Sionna chain for ``n`` samples at once -> (n, num_samples) complex tensor.

    Batching the chain (vs one sample at a time) is ~10-30x faster on CPU/GPU.

    Args:
        cfg: protocol numerology; modulation/snr_db/mobility as in ``generate_iq``.
        n: batch size (samples to generate in one forward pass).
    """
    rg, mapper, rg_mapper, modulator = _ofdm_chain(cfg.name, modulation)

    bits = _BINARY_SOURCE([n, 1, 1, int(rg.num_data_symbols * MOD_BITS[modulation])])
    symbols = mapper(bits)                                  # (n,1,1,num_data_symbols)
    grid = rg_mapper(symbols)                               # (n,1,1,num_ofdm_symbols,fft_size)
    x_time = modulator(grid)                                # (n,1,1,num_time_samples) complex

    num_time = x_time.shape[-1]
    tdl, l_min, l_max, l_tot, apply = _channel(cfg.name, mobility, num_time)
    cir = tdl(batch_size=n, num_time_steps=num_time + l_tot - 1, sampling_frequency=cfg.sample_rate)
    h_time = cir_to_time_channel(cfg.sample_rate, *cir, l_min=l_min, l_max=l_max, normalize=True)

    no = torch.tensor(10.0 ** (-snr_db / 10.0), dtype=torch.float32, device=DEVICE)
    y = apply(x_time, h_time, no)                           # (n,1,1,num_time + l_tot - 1) complex
    return y.reshape(n, -1)


def generate_iq(cfg: ProtocolConfig, modulation: str, snr_db: float, mobility: str) -> torch.Tensor:
    """Single-sample convenience wrapper -> 1-D complex baseband I/Q tensor."""
    return generate_iq_batch(cfg, modulation, snr_db, mobility, n=1).reshape(-1)
