"""Single-carrier pulse-shaped waveform synthesis (LWM-Spectro paper eq. 4).

The reference paper (arXiv:2601.08780, sec. II-A) generates its transmit waveform as

    x[n] = sum_i s[i] * g[n - i*N_os],   n = 0 .. N_x-1                        (eq. 4)

i.e. a **single-carrier**, pulse-shaped stream: one symbol sequence upsampled by ``N_os`` and
convolved with a shaping filter ``g`` of length ``N_g``. There is no IFFT, no subcarrier mapping
and no cyclic prefix anywhere in the paper.

Why this matters (and why the OFDM generator this replaces could not work):
OFDM sums 52-624 independent subcarriers, so by the CLT its time-domain samples are ~complex
Gaussian *regardless of the constellation* — modulation is erased from the magnitude spectrogram.
A single-carrier pulse-shaped signal is the opposite: the constellation survives in the envelope.

  - **BPSK** symbols are real, so ``x[n]`` is a real sequence times a real filter -> the waveform
    stays on the real axis, its envelope crosses ~zero at every sign transition, and its spectrum
    is conjugate-symmetric. Very high envelope variance / PAPR.
  - **QPSK** is constant-modulus but complex; I and Q flip independently so the envelope nulls are
    far shallower than BPSK's. This is the pair the OFDM pipeline could never separate.
  - **QAM16/64/256** have progressively more amplitude rings, so symbol energy itself varies and
    the envelope variance grows monotonically with the order.

All of these are visible in a *magnitude* spectrogram given a short enough STFT window, which is
what lets the paper use single-channel (C=1) power spectrograms.
"""
from __future__ import annotations

import functools
import math

import torch


def rrc_taps(beta: float, sps: int, span: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Root-raised-cosine impulse response, unit-energy.

    Args:
        beta: roll-off factor in [0, 1). Occupied bandwidth is ``(1+beta)/sps`` of the sample rate.
        sps: samples per symbol (the oversampling factor ``N_os`` in eq. 4).
        span: filter span in symbols; the filter length is ``span*sps + 1`` (``N_g``).

    The closed form has removable singularities at ``t = 0`` and ``t = +/- T/(4*beta)``; both are
    substituted with their analytic limits rather than an epsilon fudge, so the taps are exact.
    """
    n = torch.arange(-span * sps / 2, span * sps / 2 + 1, device=device, dtype=torch.float64)
    t = n / sps                                     # time in symbol periods
    pi = math.pi
    h = torch.empty_like(t)

    # generic branch
    denom = pi * t * (1.0 - (4.0 * beta * t) ** 2)
    num = torch.sin(pi * t * (1.0 - beta)) + 4.0 * beta * t * torch.cos(pi * t * (1.0 + beta))
    safe = denom.abs() > 1e-12
    h[safe] = num[safe] / denom[safe]

    # t = 0  ->  1 + beta*(4/pi - 1)
    h[t == 0] = 1.0 + beta * (4.0 / pi - 1.0)

    # t = +/- 1/(4*beta)  ->  (beta/sqrt(2)) * [(1+2/pi)sin(pi/4beta) + (1-2/pi)cos(pi/4beta)]
    if beta > 0:
        sing = (~safe) & (t != 0)
        if sing.any():
            h[sing] = (beta / math.sqrt(2.0)) * (
                (1.0 + 2.0 / pi) * math.sin(pi / (4.0 * beta))
                + (1.0 - 2.0 / pi) * math.cos(pi / (4.0 * beta)))

    h = h / torch.sqrt(torch.sum(h ** 2))           # unit energy
    return h.to(dtype)


@functools.lru_cache(maxsize=32)
def _cached_taps(beta: float, sps: int, span: int, device_str: str) -> torch.Tensor:
    return rrc_taps(beta, sps, span, device=torch.device(device_str))


def pulse_shape(symbols: torch.Tensor, sps: int, beta: float, span: int) -> torch.Tensor:
    """Upsample by ``sps`` and RRC-filter a batch of symbol streams -> eq. (4).

    Args:
        symbols: (B, N_s) complex symbol streams.
        sps/beta/span: oversampling factor, roll-off, filter span in symbols.

    Returns:
        (B, N_s*sps) complex waveform. 'same'-length convolution (the filter's group delay is
        trimmed symmetrically) so the output is exactly ``N_s*sps`` samples.
    """
    assert symbols.dim() == 2, f"expected (B, N_s), got {tuple(symbols.shape)}"
    b, ns = symbols.shape
    dev = symbols.device
    taps = _cached_taps(float(beta), int(sps), int(span), str(dev))
    ntaps = taps.numel()

    up = torch.zeros(b, ns * sps, dtype=symbols.dtype, device=dev)
    up[:, ::sps] = symbols                                   # zero-stuff: s[i] at n = i*N_os

    # complex conv via two real convs (conv1d has no complex kernel support)
    k = taps.reshape(1, 1, ntaps)
    pad = ntaps // 2
    def _c(v):
        return torch.nn.functional.conv1d(v.reshape(b, 1, -1), k, padding=pad).reshape(b, -1)[:, :ns * sps]
    return torch.complex(_c(up.real.contiguous()), _c(up.imag.contiguous()))


def matched_filter(rx: torch.Tensor, sps: int, beta: float, span: int) -> torch.Tensor:
    """Receive-side RRC matched filter (RRC * RRC = raised cosine, i.e. Nyquist).

    Applied before the STFT so the stored spectrogram is the *matched-filtered* received signal —
    this suppresses out-of-band noise and is what a real receiver front-end does.
    """
    return pulse_shape_filter_only(rx, sps, beta, span)


def pulse_shape_filter_only(x: torch.Tensor, sps: int, beta: float, span: int) -> torch.Tensor:
    """Filter (no upsampling) a batch of complex waveforms with the RRC taps."""
    b = x.shape[0]
    taps = _cached_taps(float(beta), int(sps), int(span), str(x.device))
    ntaps = taps.numel()
    k = taps.reshape(1, 1, ntaps)
    pad = ntaps // 2
    n = x.shape[-1]
    def _c(v):
        return torch.nn.functional.conv1d(v.reshape(b, 1, -1), k, padding=pad).reshape(b, -1)[:, :n]
    return torch.complex(_c(x.real.contiguous()), _c(x.imag.contiguous()))


def apply_freq_offset(x: torch.Tensor, f_norm: torch.Tensor) -> torch.Tensor:
    """Multiply each row by ``exp(j*2*pi*f_norm*n)``; ``f_norm`` is (B,) cycles/sample.

    Used to place each burst at a random offset within the simulated band. Without this the
    signal always occupies the same centre bins and the out-of-band region sits at a fixed index,
    which turns SNR into a trivial "compare occupied-band brightness to guard-band brightness"
    shortcut that any feature extractor solves (measured at 27-60 sigma on the OFDM corpus).
    """
    n = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32)
    ph = 2.0 * math.pi * f_norm.reshape(-1, 1).to(torch.float32) * n[None, :]
    return x * torch.exp(1j * ph.to(torch.complex64))
