"""Per-protocol PHY numerology + the label grid for the synthetic generator.

Sionna is 5G-NR/OFDM-centric, so LTE and WiFi are *approximated* by distinct OFDM numerologies
(subcarrier spacing / FFT size / cyclic prefix / occupied bandwidth). These differences give the
three techs visibly different time-frequency footprints, which is what makes the protocol router
learnable. This is intentionally not standard-compliant — see PLAN.md.

All values mirror the public PHY parameters of each standard closely enough to be distinctive:
- WiFi 802.11a/g : 20 MHz, 64-pt FFT, 312.5 kHz SCS, 52 used subcarriers, short symbols.
- LTE (10 MHz)   : 15.36 MHz rate, 1024-pt FFT, 15 kHz SCS, 600 used subcarriers, long symbols.
- 5G NR (mu=1)   : 30.72 MHz rate, 1024-pt FFT, 30 kHz SCS, 624 used subcarriers.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- label grid (mirrors demo_data) -------------------------------------------------------
PROTOCOLS = ["LTE", "WiFi", "5G"]
MODULATIONS = ["BPSK", "QPSK", "QAM16", "QAM64", "QAM256"]
# bits per symbol per modulation (BPSK handled specially as 1-bit PAM)
MOD_BITS = {"BPSK": 1, "QPSK": 2, "QAM16": 4, "QAM64": 6, "QAM256": 8}
SNRS_DB = [-5, 0, 5, 10, 15, 20, 25]
MOBILITIES = ["static", "pedestrian", "vehicular"]

# mobility -> UE speed in m/s (Doppler = speed * carrier_freq / c)
MOBILITY_SPEED_MS = {"static": 0.0, "pedestrian": 1.0, "vehicular": 30.0}

CARRIER_FREQUENCY_HZ = 3.5e9   # used for Doppler from speed
SPEED_OF_LIGHT = 3e8


def snr_label(snr_db: int) -> str:
    """Format an integer SNR (dB) as the demo's string label, e.g. -5 -> 'SNR-5dB'."""
    return f"SNR{snr_db}dB"


@dataclass(frozen=True)
class ProtocolConfig:
    name: str
    fft_size: int
    subcarrier_spacing: float      # Hz
    cyclic_prefix_length: int      # samples
    num_used_subcarriers: int      # occupied (data+pilot) subcarriers, centered
    num_ofdm_symbols: int          # symbols per generated burst

    @property
    def sample_rate(self) -> float:
        return self.fft_size * self.subcarrier_spacing

    @property
    def num_guard_carriers(self):
        """(left, right) guard carriers so used subcarriers sit centered in the FFT (DC null)."""
        total_guard = self.fft_size - self.num_used_subcarriers
        left = total_guard // 2
        right = total_guard - left
        return (left, right)


# Distinct numerologies per tech. num_ofdm_symbols sized so the time series is ~20-30k samples
# (>=128 STFT frames after resize) while keeping the TDL channel tensor small enough for GPU batches.
PROTOCOL_CONFIGS = {
    "WiFi": ProtocolConfig(
        name="WiFi", fft_size=64, subcarrier_spacing=312.5e3,
        cyclic_prefix_length=16, num_used_subcarriers=52, num_ofdm_symbols=160),   # ~12.8k samples
    "LTE": ProtocolConfig(
        name="LTE", fft_size=1024, subcarrier_spacing=15e3,
        cyclic_prefix_length=72, num_used_subcarriers=600, num_ofdm_symbols=12),    # ~13.2k samples
    "5G": ProtocolConfig(
        name="5G", fft_size=1024, subcarrier_spacing=30e3,
        cyclic_prefix_length=72, num_used_subcarriers=624, num_ofdm_symbols=12),    # ~13.2k samples
}


def doppler_hz(mobility: str) -> float:
    speed = MOBILITY_SPEED_MS[mobility]
    return speed * CARRIER_FREQUENCY_HZ / SPEED_OF_LIGHT
