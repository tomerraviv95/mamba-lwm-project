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
# Per-class speed RANGES (m/s) for --vary-speed: broaden the train mobility distribution so its
# Doppler signature isn't a single point (helps cover/overlap the test distribution). Sampled U[lo,hi].
MOBILITY_SPEED_RANGE = {"static": (0.0, 0.0), "pedestrian": (0.5, 5.0), "vehicular": (8.0, 50.0)}

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


# ---- single-carrier numerology (LWM-Spectro eq. 4) ----------------------------------------
# The reference paper uses a single-carrier PULSE-SHAPED waveform, not OFDM. "WiFi"/"LTE"/"5G"
# there denote different bandwidths, MCS sets and pulse/coding parameters -- not OFDM
# numerologies. These configs mirror that: distinct sample rates + roll-offs (so the protocol
# router still has a learnable spectral footprint) over a common burst length.
#
# Burst duration is what makes mobility learnable. num_symbols*sps samples at `sample_rate`:
#   LTE 262144 @ 15.36 MHz = 17.1 ms | WiFi 262144 @ 20 MHz = 13.1 ms | 5G 262144 @ 30.72 MHz = 8.5 ms
# MEASURED Doppler budget across the LTE burst (cycles of phase rotation):
#   static 0 | pedestrian 0.10-1.00 | vehicular 1.59-9.96
# The earlier 4.27 ms burst gave pedestrian only 0.02-0.25 cycles -- indistinguishable from static,
# which pinned 3-way mobility at chance. sps=2 (not 4) is deliberate: it keeps occupancy at
# (1+beta)/2 ~ 68% of the band and lets 128 frames x 4 symbols observe ~512 symbols; sps=4 drops
# occupancy to 34% and halves the observed symbol count, measurably costing modulation
# (macro-F1 0.416 -> 0.298, BPSK/QPSK accuracy 0.802 -> 0.473).
@dataclass(frozen=True)
class SingleCarrierConfig:
    name: str
    sample_rate: float        # Hz
    sps: int                  # N_os, samples per symbol (oversampling factor)
    rolloff: float            # RRC beta; occupied bandwidth = (1+beta)/sps of sample_rate
    span_symbols: int         # RRC span -> filter length N_g = span*sps + 1
    num_symbols: int          # N_s symbols per burst

    @property
    def num_samples(self) -> int:
        return self.num_symbols * self.sps

    @property
    def duration_s(self) -> float:
        return self.num_samples / self.sample_rate

    @property
    def occupied_frac(self) -> float:
        """Fraction of the simulated band the shaped signal occupies."""
        return (1.0 + self.rolloff) / self.sps


SC_CONFIGS = {
    "WiFi": SingleCarrierConfig("WiFi", sample_rate=20.00e6, sps=2, rolloff=0.25,
                                span_symbols=10, num_symbols=131072),
    "LTE": SingleCarrierConfig("LTE", sample_rate=15.36e6, sps=2, rolloff=0.35,
                               span_symbols=10, num_symbols=131072),
    "5G": SingleCarrierConfig("5G", sample_rate=30.72e6, sps=2, rolloff=0.15,
                              span_symbols=10, num_symbols=131072),
}


def doppler_hz(mobility: str) -> float:
    speed = MOBILITY_SPEED_MS[mobility]
    return speed * CARRIER_FREQUENCY_HZ / SPEED_OF_LIGHT
