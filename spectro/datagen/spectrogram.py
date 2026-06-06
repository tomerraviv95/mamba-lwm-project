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
