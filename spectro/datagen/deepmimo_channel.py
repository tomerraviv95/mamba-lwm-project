"""Site-specific channel for the spectrogram pipeline: 3GPP-TDL-style fading whose tap
delays/powers come from DeepMIMO ray-tracing (matching the LWM-Spectro authors' description).

Instead of a generic TDL-A..E power-delay profile, we use each city user's ray-traced
(delay, power, phase, AoA) paths as the taps, with per-ray Doppler from the UE speed and the
ray's azimuth AoA. The resulting CIR plugs into the SAME Sionna apply path
(``cir_to_time_channel`` -> ``ApplyTimeChannel``) the synthetic pipeline already uses.

Two pieces:
- ``extract_city_pdp``  : pull padded per-user (delay, power, phase, aoa) tables from DeepMIMO.
- ``deepmimo_tdl_cir``  : turn a batch of those tables into Sionna ``(a, tau)`` CIR tensors.
"""
from __future__ import annotations

import numpy as np
import torch

C = 3e8

# The 20 LWM city scenarios (path delays/powers/angles are geometric -> antenna config irrelevant).
CITY_SCENARIOS = [
    "city_0_newyork_3p5_lwm", "city_1_losangeles_3p5_lwm", "city_2_chicago_3p5_lwm",
    "city_3_houston_3p5_lwm", "city_4_phoenix_3p5_lwm", "city_5_philadelphia_3p5_lwm",
    "city_6_miami_3p5_lwm", "city_7_sandiego_3p5_lwm", "city_8_dallas_3p5_lwm",
    "city_9_sanfrancisco_3p5_lwm", "city_10_austin_3p5_lwm", "city_11_santaclara_3p5_lwm",
    "city_12_fortworth_3p5_lwm", "city_13_columbus_3p5_lwm", "city_14_charlotte_3p5_lwm",
    "city_15_indianapolis_3p5_lwm", "city_16_sanfrancisco_3p5_lwm", "city_17_seattle_3p5_lwm",
    "city_18_denver_3p5_lwm", "city_19_oklahoma_3p5_lwm",
]


def extract_city_pdp(scenario, bs_idx=1, grid_idx=0, max_paths=None):
    """Return padded per-(valid-)user ray tables for a DeepMIMO city.

    Returns a dict of float32 arrays, each (U, K): ``delay`` [s], ``power_linear``,
    ``phase`` [rad], ``aoa_az`` [rad], plus ``num_paths`` (U,). Invalid (no-link) users dropped.
    """
    import deepmimo as dm
    d = dm.load(scenario, tx_sets=[bs_idx], rx_sets=[grid_idx])
    valid = np.where(np.asarray(d.los) != -1)[0]

    delay = np.asarray(d.delay, dtype=np.float32)[valid]          # (U, Kmax) seconds
    power = np.asarray(getattr(d, 'power_linear', d.power), dtype=np.float32)[valid]
    phase = np.deg2rad(np.asarray(d.phase, dtype=np.float32)[valid])   # phase stored in degrees
    aoa = np.deg2rad(np.asarray(d.aoa_az, dtype=np.float32)[valid])    # azimuth AoA in degrees
    npaths = np.asarray(d.num_paths, dtype=np.int64)[valid]

    # NaNs mark empty path slots -> zero power / zero delay (won't contribute).
    power = np.nan_to_num(power, nan=0.0)
    delay = np.nan_to_num(delay, nan=0.0)
    phase = np.nan_to_num(phase, nan=0.0)
    aoa = np.nan_to_num(aoa, nan=0.0)
    if max_paths is not None:
        delay, power, phase, aoa = (x[:, :max_paths] for x in (delay, power, phase, aoa))
    return {'delay': delay, 'power_linear': power, 'phase': phase, 'aoa_az': aoa,
            'num_paths': npaths}


def deepmimo_tdl_cir(delay, power_linear, phase, aoa_az, speed, num_time_steps, sample_rate,
                     fc=3.5e9, device='cpu'):
    """Build Sionna CIR ``(a, tau)`` from ray-traced taps + per-ray Doppler.

    Args (all batched (B, K) numpy/torch): path delay [s], linear power, phase [rad], AoA [rad].
    ``speed`` [m/s] sets Doppler ``f_d,k = (speed/c)*fc*cos(AoA_k)`` per ray (0 -> time-invariant).

    Returns ``a`` (B,1,1,1,1,K,T) complex64 and ``tau`` (B,1,1,K) float32, matching Sionna's
    TDL output so ``cir_to_time_channel`` can consume it unchanged.
    """
    to = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    delay, power_linear, phase, aoa_az = map(to, (delay, power_linear, phase, aoa_az))
    B, K = delay.shape
    t = torch.arange(num_time_steps, device=device, dtype=torch.float32) / sample_rate   # (T,)

    # speed: scalar or per-sample (B,) [m/s] -> (B,1) for broadcasting against (B,K)
    speed_t = torch.as_tensor(np.asarray(speed), dtype=torch.float32, device=device).reshape(-1, 1)
    g = torch.sqrt(torch.clamp(power_linear, min=0.0)) * torch.exp(1j * phase.to(torch.complex64))
    fd = (speed_t / C) * fc * torch.cos(aoa_az)                    # (B,K) per-ray Doppler [Hz]
    ramp = torch.exp(1j * (2 * np.pi * fd[..., None] * t[None, None, :]).to(torch.complex64))  # (B,K,T)
    a = (g[..., None] * ramp).reshape(B, 1, 1, 1, 1, K, num_time_steps)
    tau = delay.reshape(B, 1, 1, K)
    return a.to(torch.complex64), tau.to(torch.float32)


def sample_user_pdp(pdp, idx):
    """Slice one user's (1, K) tables from an ``extract_city_pdp`` dict (batchable to many idxs)."""
    idx = np.atleast_1d(idx)
    return {k: pdp[k][idx] for k in ('delay', 'power_linear', 'phase', 'aoa_az')}
