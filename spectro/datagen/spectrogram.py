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
