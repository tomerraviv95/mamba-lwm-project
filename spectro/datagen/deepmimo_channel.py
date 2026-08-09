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
    ``phase`` [rad], ``aoa_az`` [rad], plus ``num_paths`` (U,) and ``user_idx`` (U,) = the RAW grid
    index of each kept user. ``user_idx`` is what makes a BS-independent train/eval user split
    possible: the valid-user SET differs per BS, so splitting each BS's filtered list separately
    would put the same physical location in train for one BS and in eval for another.
    Invalid (no-link) users dropped.
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
            'num_paths': npaths, 'user_idx': valid.astype(np.int64)}


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


def sc_channel_apply(x, delay, power_linear, phase, aoa_az, speed, sample_rate,
                     fc=3.5e9, n_subrays=12, angle_spread_rad=0.52, rng=None):
    """Apply the ray-traced channel to a single-carrier waveform: y[n] = sum_l h_l[n] x[n-d_l].

    This is the single-carrier counterpart of ``deepmimo_tdl_cir`` + Sionna's
    ``cir_to_time_channel``/``ApplyTimeChannel``. It exists for two reasons:

    * **Memory.** Sionna's path materializes a ``(B, T, l_tot)`` time-channel tensor. The
      single-carrier burst is ~262k samples, which makes that tensor tens of GB. The explicit tap
      loop here is O(B*T).
    * **Correct fading statistics.** ``deepmimo_tdl_cir`` gives each ray ONE deterministic Doppler
      shift ``f_d cos(AoA)``, which leaves ``|h_l[n]|`` CONSTANT -- Doppler becomes a pure phase
      rotation that a magnitude spectrogram discards, so LOS-dominated users (the majority: median
      2 significant paths in DeepMIMO) carry no mobility signature at all. Measured effect: 3-way
      mobility sat at chance (0.34 macro-F1). The paper's eq. (6) instead specifies a *temporally
      correlated fading process generated according to the 3GPP Doppler spectrum*. Modelling each
      tap as a sum of sub-rays spread about the path AoA reproduces that, and mobility separability
      jumps to ~0.69 (moment probe) / ~0.54 (from-scratch CNN).

    Args:
        x: (B, T) complex transmit waveform.
        delay/power_linear/phase/aoa_az: (B, K) ray tables from ``extract_city_pdp``.
        speed: (B,) UE speed [m/s]. sample_rate: Hz. rng: numpy RandomState for the sub-ray draw.

    Returns:
        (B, T) complex received waveform, before AWGN.
    """
    dev = x.device
    B, T = x.shape
    rng = rng if rng is not None else np.random
    t = torch.arange(T, device=dev, dtype=torch.float32) / sample_rate
    d = torch.as_tensor(np.asarray(delay), dtype=torch.float32, device=dev)
    p = torch.as_tensor(np.asarray(power_linear), dtype=torch.float32, device=dev)
    ph = torch.as_tensor(np.asarray(phase), dtype=torch.float32, device=dev)
    ao = torch.as_tensor(np.asarray(aoa_az), dtype=torch.float32, device=dev)
    sp = torch.as_tensor(np.asarray(speed), dtype=torch.float32, device=dev).reshape(-1, 1)

    p = p / torch.clamp(p.sum(dim=1, keepdim=True), min=1e-20)      # unit total channel power
    lag = torch.round(d * sample_rate).long().clamp_(0, T - 1)
    g = torch.sqrt(torch.clamp(p, min=0.0)) * torch.exp(1j * ph.to(torch.complex64))
    fd_max = (sp / C) * fc                                          # (B,1) max Doppler [Hz]

    arange = torch.arange(T, device=dev)
    y = torch.zeros_like(x)
    for k in range(d.shape[1]):
        gk = g[:, k:k + 1]
        if torch.all(gk.abs() == 0):
            continue
        alpha = torch.zeros(B, T, dtype=torch.complex64, device=dev)
        dth = torch.as_tensor(rng.uniform(-angle_spread_rad, angle_spread_rad, (B, n_subrays)),
                              dtype=torch.float32, device=dev)
        phi = torch.as_tensor(rng.uniform(0, 2 * np.pi, (B, n_subrays)),
                              dtype=torch.float32, device=dev)
        for m in range(n_subrays):
            fdm = fd_max * torch.cos(ao[:, k:k + 1] + dth[:, m:m + 1])
            arg = 2 * np.pi * fdm * t[None, :] + phi[:, m:m + 1]
            alpha = alpha + torch.exp(1j * arg.to(torch.complex64))
        alpha = alpha / np.sqrt(n_subrays)
        idx = (arange[None, :] - lag[:, k:k + 1]).clamp_(min=0)
        xs = torch.gather(x, 1, idx) * (arange[None, :] >= lag[:, k:k + 1])
        y = y + gk * alpha * xs
    return y


def sample_user_pdp(pdp, idx):
    """Slice one user's (1, K) tables from an ``extract_city_pdp`` dict (batchable to many idxs)."""
    idx = np.atleast_1d(idx)
    return {k: pdp[k][idx] for k in ('delay', 'power_linear', 'phase', 'aoa_az')}
