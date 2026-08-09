"""Baseband I/Q -> 128x128 single-channel dB spectrogram (matches the demo contract).

Uses ``torch.stft`` (no extra deps). Output is a per-sample z-scored log-magnitude spectrogram,
the same representation the demo data ships and that ``spectro_patchify`` expects.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

OUT_SIZE = 128
N_FFT = 512


def iq_to_spectrogram(iq: torch.Tensor, n_fft: int = N_FFT, out_size: int = OUT_SIZE,
                      normalize: bool = True) -> torch.Tensor:
    """Convert a 1-D complex baseband signal to a (1, out_size, out_size) dB spectrogram.

    Args:
        iq: complex (or 2-channel real/imag) 1-D tensor of time-domain samples.
        n_fft: STFT FFT size (512 -> 257 freq bins, resized to out_size).
        out_size: target square size (128).
        normalize: per-sample z-score (matches demo normalization).
    """
    if not torch.is_complex(iq):
        iq = iq.to(torch.complex64)
    iq = iq.reshape(-1)
    # hop so that the number of frames is comfortably >= out_size
    n_frames_target = out_size + 4
    hop = max(1, (iq.shape[0] - n_fft) // n_frames_target)

    window = torch.hann_window(n_fft, device=iq.device)
    spec = torch.stft(iq, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window,
                      center=True, return_complex=True)            # (freq, frames)
    mag = spec.abs()
    db = 20.0 * torch.log10(mag + 1e-8)                            # log-magnitude in dB

    # resize (freq, frames) -> (out_size, out_size) via bilinear interpolation
    db = db.unsqueeze(0).unsqueeze(0)                              # (1,1,F,T)
    db = F.interpolate(db, size=(out_size, out_size), mode="bilinear", align_corners=False)
    db = db.squeeze(0)                                            # (1, out, out)

    if normalize:
        mean = db.mean()
        std = torch.clamp(db.std(), min=1e-6)
        db = (db - mean) / std
    return db.to(torch.float16)


def iq_batch_to_spectrogram(iq: torch.Tensor, n_fft: int = N_FFT, out_size: int = OUT_SIZE,
                            normalize: bool = True) -> torch.Tensor:
    """Batched version: (n, T) complex -> (n, 1, out_size, out_size) float16 dB spectrograms."""
    if not torch.is_complex(iq):
        iq = iq.to(torch.complex64)
    n = iq.shape[0]
    n_frames_target = out_size + 4
    hop = max(1, (iq.shape[-1] - n_fft) // n_frames_target)
    window = torch.hann_window(n_fft, device=iq.device)
    spec = torch.stft(iq, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window,
                      center=True, return_complex=True)            # (n, freq, frames)
    db = 20.0 * torch.log10(spec.abs() + 1e-8)
    db = db.unsqueeze(1)                                            # (n,1,F,T)
    db = F.interpolate(db, size=(out_size, out_size), mode="bilinear", align_corners=False)
    if normalize:
        mean = db.mean(dim=(1, 2, 3), keepdim=True)
        std = torch.clamp(db.std(dim=(1, 2, 3), keepdim=True), min=1e-6)
        db = (db - mean) / std
    return db.to(torch.float16)                                    # (n,1,out,out)


# Corpus-level dB normalization constants for ``norm='global'``.
# Per-sample z-scoring divides each spectrogram by the std of its own dB values -- but envelope
# variance IS the statistic that separates modulation orders (BPSK 0.44 / QPSK 0.24 / QAM16 0.51 /
# QAM64 0.58 / QAM256 0.59 pre-channel), so per-sample normalization deletes the modulation cue and
# couples what remains to SNR. The paper normalizes "with pretrained statistics", i.e. corpus-level.
# Measured over the single-carrier corpus (all 3 protocols x 5 mods x 7 SNRs x 3 mobilities,
# DeepMIMO channels): mean -4.49 dB, std 13.34 dB, 1st/99th pct -39.2/+16.4 dB. Recalibrate with
# `--sc-norm none` + a few hundred samples if the numerology or SNR grid changes.
SC_DB_MEAN = -4.5
SC_DB_STD = 13.3


def sc_power_spectrogram(iq: torch.Tensor, n_fft: int = OUT_SIZE, win_length: int = 8,
                         out_size: int = OUT_SIZE, norm: str = 'global',
                         db_mean: float = SC_DB_MEAN, db_std: float = SC_DB_STD,
                         eps: float = 1e-12) -> torch.Tensor:
    """Single-carrier power spectrogram, matching LWM-Spectro eq. (7)-(10).

    ``P[t,k] = |Y[t,k]|^2`` -> ``10*log10`` -> per-sample normalize -> ``(n, 1, 128, 128)``.

    Two deliberate differences from every other function in this module:

    1. **No resize.** The hop is chosen so the STFT produces exactly ``out_size`` frames and
       ``n_fft == out_size`` gives exactly ``out_size`` frequency bins, so the spectrogram is
       natively 128x128. The nearest/bilinear ``F.interpolate`` used by the OFDM paths was
       measured to duplicate 25% of LTE rows and 50% of WiFi columns (injecting a spurious +0.24
       lag-1 autocorrelation into *static* samples, contaminating the Doppler cue) while
       discarding 7/8 of LTE's resource elements.
    2. **Short analysis window, zero-padded to ``n_fft``.** ``win_length`` spans only a few symbol
       periods, so each frame resolves the *instantaneous* envelope rather than converging to the
       average RRC power spectrum (which is identical for every constellation). This is what keeps
       modulation visible in a magnitude representation. Zero-padding to ``n_fft`` still yields the
       full ``out_size`` frequency bins.

    ``win_length`` therefore trades modulation visibility (short) against frequency resolution
    (long) — see ``sweep_sc_config.py`` for the measurement that picked the default.

    ``norm``: ``'global'`` (default) subtracts fixed corpus-level dB statistics, preserving both
    absolute power (the SNR cue) and per-sample dB variance (the modulation cue). ``'sample'`` is
    the legacy per-sample z-score, which deletes both. ``'none'`` returns raw dB.
    """
    if not torch.is_complex(iq):
        iq = iq.to(torch.complex64)
    if iq.dim() == 1:
        iq = iq[None]
    n_samples = iq.shape[-1]
    hop = max(1, (n_samples - n_fft) // (out_size - 1))
    window = torch.hann_window(win_length, device=iq.device)
    spec = torch.stft(iq, n_fft=n_fft, hop_length=hop, win_length=win_length, window=window,
                      center=False, onesided=False, return_complex=True)   # (n, n_fft, frames)
    spec = spec[..., :out_size]
    if spec.shape[-1] < out_size:                       # pad-repeat only if the burst was short
        spec = torch.cat([spec, spec[..., -1:].expand(-1, -1, out_size - spec.shape[-1])], dim=-1)
    spec = torch.fft.fftshift(spec, dim=1)              # DC to the centre bin
    p_db = 10.0 * torch.log10(spec.abs().pow(2) + eps)  # eq. (9) + log scaling
    p_db = p_db.unsqueeze(1)                            # (n, 1, K, T)
    if norm == 'global':
        p_db = (p_db - db_mean) / db_std
    elif norm == 'sample':
        mean = p_db.mean(dim=(1, 2, 3), keepdim=True)
        std = torch.clamp(p_db.std(dim=(1, 2, 3), keepdim=True), min=1e-6)
        p_db = (p_db - mean) / std
    elif norm != 'none':
        raise ValueError(f"norm must be one of global|sample|none, got {norm!r}")
    return p_db.to(torch.float16)                       # (n, 1, out, out)


def grid_mag_to_spectrogram(grid_mag: torch.Tensor, out_size: int = OUT_SIZE,
                            normalize: bool = True) -> torch.Tensor:
    """Received resource-grid magnitude -> (n,1,out,out) float16 dB z-scored spectrogram.

    ``grid_mag``: (n, num_ofdm_symbols, fft_size) magnitude of the DEMODULATED received grid |Y[k,n]|.
    Unlike ``iq_batch_to_spectrogram`` (|STFT| of the time-domain OFDM waveform, which averages the
    constellation away by CLT so modulation is invisible), the resource grid preserves each
    subcarrier's symbol amplitude -> modulation order IS visible (BPSK ~constant |.|, high-QAM many
    levels), while SNR (per-cell noise), fading (across subcarriers) and Doppler (across symbols)
    remain. Same dB + per-sample z-score contract as the STFT path."""
    g = grid_mag if torch.is_tensor(grid_mag) else torch.as_tensor(grid_mag)
    g = g.float()
    if g.dim() == 2:
        g = g[None]
    db = 20.0 * torch.log10(g + 1e-8).unsqueeze(1)                  # (n,1,n_sym,fft)
    # NEAREST (not bilinear): bilinear averages neighbouring resource elements, which washes out the
    # per-RE constellation amplitude that carries modulation. Nearest samples native REs (no averaging).
    db = F.interpolate(db, size=(out_size, out_size), mode="nearest")
    if normalize:
        mean = db.mean(dim=(1, 2, 3), keepdim=True)
        std = torch.clamp(db.std(dim=(1, 2, 3), keepdim=True), min=1e-6)
        db = (db - mean) / std
    return db.to(torch.float16)                                    # (n,1,out,out)


def grid_complex_to_spectrogram(grid_c: torch.Tensor, out_size: int = OUT_SIZE,
                                normalize: bool = True) -> torch.Tensor:
    """COMPLEX received resource grid Y[k,n] -> (n, 2, out, out) float16 [real, imag].

    Like ``grid_mag_to_spectrogram`` but KEEPS PHASE: the two channels are Re(Y) and Im(Y) of the
    demodulated received grid. Magnitude alone cannot separate BPSK from QPSK (both constant |.|); the
    full complex grid exposes the constellation (I/Q per resource element) so ALL modulation orders are
    distinguishable. NEAREST resize (no averaging of the per-RE constellation). Per-sample z-score is
    applied JOINTLY across both channels (preserves the I/Q relationship / phase)."""
    g = grid_c if torch.is_tensor(grid_c) else torch.as_tensor(grid_c)
    if not torch.is_complex(g):
        g = g.to(torch.complex64)
    if g.dim() == 2:
        g = g[None]
    ri = torch.stack([g.real, g.imag], dim=1).float()              # (n, 2, n_sym, fft)
    ri = F.interpolate(ri, size=(out_size, out_size), mode="nearest")
    if normalize:
        mean = ri.mean(dim=(1, 2, 3), keepdim=True)
        std = torch.clamp(ri.std(dim=(1, 2, 3), keepdim=True), min=1e-6)
        ri = (ri - mean) / std
    return ri.to(torch.float16)                                    # (n, 2, out, out)


def iq_batch_to_complex_spectrogram(iq: torch.Tensor, n_fft: int = N_FFT, out_size: int = OUT_SIZE,
                                    normalize: bool = True) -> torch.Tensor:
    """Batched COMPLEX spectrogram: (n, T) complex -> (n, 2, out, out) float16 [real, imag].

    The LWM-Spectro authors pretrain on complex spectrograms (real+imag interleaved,
    element_length=32) rather than the magnitude/dB representation. Here we keep the real and
    imaginary STFT components as two channels (resized + per-sample z-scored jointly across both
    channels). ``spectro_patchify`` turns a (2,128,128) sample into 4x4x2 = 32-dim patch tokens.
    """
    if not torch.is_complex(iq):
        iq = iq.to(torch.complex64)
    n_frames_target = out_size + 4
    hop = max(1, (iq.shape[-1] - n_fft) // n_frames_target)
    window = torch.hann_window(n_fft, device=iq.device)
    spec = torch.stft(iq, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window,
                      center=True, return_complex=True)            # (n, freq, frames)
    ri = torch.stack([spec.real, spec.imag], dim=1)                # (n, 2, F, T)
    ri = F.interpolate(ri, size=(out_size, out_size), mode="bilinear", align_corners=False)
    if normalize:
        mean = ri.mean(dim=(1, 2, 3), keepdim=True)
        std = torch.clamp(ri.std(dim=(1, 2, 3), keepdim=True), min=1e-6)
        ri = (ri - mean) / std
    return ri.to(torch.float16)                                    # (n,2,out,out)
