"""Single-channel spectrogram patchify + masked-token sample builder.

Mirrors the LWM-Spectro tokenizer (``spectro/hf_cache/utils.py``: ``patch_maker`` with
``interleaved=False`` and ``make_sample``) but specialised for the real-valued (128,128)
demo spectrograms we have. Patches are 4x4 -> ``element_length = 16`` (no real/imag x2),
with per-sample z-score normalisation and a CLS token of ``0.2*ones(16)``.
"""
from __future__ import annotations

import numpy as np
import torch

PATCH = 4
ELEMENT_LENGTH = PATCH * PATCH          # 16, single real channel
CLS_TOKEN = np.full(ELEMENT_LENGTH, 0.2, dtype=np.float32)
MASK_TOKEN = np.full(ELEMENT_LENGTH, 0.1, dtype=np.float32)


def _as_2d(spec: np.ndarray) -> np.ndarray:
    """Squeeze (1,H,W)/(1,1,H,W) leading singleton dims down to (H,W)."""
    spec = np.asarray(spec, dtype=np.float32)
    while spec.ndim > 2 and spec.shape[0] == 1:
        spec = spec[0]
    if spec.ndim != 2:
        raise ValueError(f"Expected a 2-D spectrogram, got shape {spec.shape}")
    return spec


def normalize_per_sample(spec: np.ndarray) -> np.ndarray:
    """Z-score a single spectrogram (matches HF per-sample normalisation)."""
    mean = float(spec.mean())
    std = float(spec.std())
    denom = std if abs(std) > 1e-6 else 1e-6
    return (spec - mean) / denom


def _patchify_2d(spec: np.ndarray, patch: int) -> np.ndarray:
    """(H,W) -> (n_patches, patch*patch) row-major 4x4 patch flatten."""
    n_rows, n_cols = spec.shape
    n_pr, n_pc = n_rows // patch, n_cols // patch
    cropped = spec[: n_pr * patch, : n_pc * patch]
    reshaped = cropped.reshape(n_pr, patch, n_pc, patch)
    return reshaped.transpose(0, 2, 1, 3).reshape(-1, patch * patch)


def spectrogram_patchify(specs, patch: int = PATCH, normalize: bool = True) -> np.ndarray:
    """Turn spectrograms into 4x4 patch tokens (single- or multi-channel).

    Args:
        specs: array/tensor of shape (N,128,128), (N,1,128,128), a single (128,128), or
            (N,2,128,128) for COMPLEX [real, imag] spectrograms.
        patch: patch side length (default 4).
        normalize: per-sample z-score (over all channels jointly) before patchifying.

    Returns:
        ``np.ndarray`` (N, n_patches, patch*patch*C): (N,1024,16) for magnitude, (N,1024,32) for
        complex (real patch flat ++ imag patch flat).
    """
    if torch.is_tensor(specs):
        specs = specs.detach().cpu().numpy()
    specs = np.asarray(specs, dtype=np.float32)
    if specs.ndim == 2:                 # (H,W) -> (1,1,H,W)
        specs = specs[None, None, ...]
    elif specs.ndim == 3:               # (N,H,W) single channel -> (N,1,H,W)
        specs = specs[:, None, ...]
    # now (N, C, H, W) with C in {1, 2}
    out = []
    for spec in specs:                  # spec: (C,H,W)
        if normalize:
            mean = float(spec.mean()); std = float(spec.std())
            spec = (spec - mean) / (std if abs(std) > 1e-6 else 1e-6)
        chans = [_patchify_2d(ch, patch) for ch in spec]          # each (n_patches, patch*patch)
        result = np.concatenate(chans, axis=-1)                   # (n_patches, patch*patch*C)
        out.append(result.astype(np.float32, copy=False))
    return np.stack(out, axis=0)


def make_sample_spectro(patches: np.ndarray, n_masks: int, mask: bool = True,
                        rng: np.random.Generator | None = None):
    """Prepend CLS and (optionally) apply 80/10/10 BERT masking to one sample's patches.

    Args:
        patches: (n_patches, element_length) tokens for a single spectrogram.
        n_masks: number of patch positions to mask.
        mask: if False, just prepend CLS and return the token tensor.
        rng: numpy Generator for reproducibility.

    Returns:
        If ``mask`` is False: ``np.ndarray`` (n_patches+1, element_length).
        Else ``[input_ids, masked_tokens (n_masks, E), masked_pos (n_masks,)]``.
    """
    rng = rng or np.random.default_rng()
    E = patches.shape[1]                                  # element_length (16 magnitude / 32 complex)
    cls_tok = np.full(E, 0.2, dtype=np.float32)
    mask_tok = np.full(E, 0.1, dtype=np.float32)
    input_ids = np.vstack((cls_tok, patches)).astype(np.float32)
    if not mask:
        return input_ids

    n_patches = patches.shape[0]
    if n_masks <= 0 or n_patches == 0:
        masked_pos = np.empty(0, dtype=np.int64)
    else:
        n_masks = min(n_masks, n_patches)
        masked_pos = rng.choice(np.arange(1, n_patches + 1), size=n_masks, replace=False)

    masked_tokens = []
    for pos in masked_pos:
        masked_tokens.append(input_ids[pos].astype(np.float32, copy=True))
        rnd = rng.random()
        if rnd < 0.1:
            input_ids[pos] = rng.random(E).astype(np.float32)
        elif rnd < 0.9:
            input_ids[pos] = mask_tok
    masked_tokens = (np.stack(masked_tokens).astype(np.float32)
                     if masked_tokens else np.empty((0, E), dtype=np.float32))
    return [input_ids, masked_tokens, masked_pos.astype(np.int64)]


def build_masked_tensors(specs, mask_percent: float = 0.6, seed: int = 42, patch: int = PATCH):
    """Build stacked (input_ids, masked_tokens, masked_pos) tensors for MLM pretraining.

    All 128x128 spectrograms share n_patches=1024 and the same n_masks, so the per-sample
    outputs stack cleanly into TensorDataset-ready tensors.

    Returns:
        ``(input_ids, masked_tokens, masked_pos)`` torch tensors of shapes
        (N, 1025, 16), (N, n_masks, 16), (N, n_masks).
    """
    patches = spectrogram_patchify(specs, patch=patch, normalize=True)
    n_patches = patches.shape[1]
    n_masks = max(1, int(mask_percent * n_patches))
    rng = np.random.default_rng(seed)

    ids, toks, pos = [], [], []
    for p in patches:
        input_ids, masked_tokens, masked_pos = make_sample_spectro(p, n_masks, mask=True, rng=rng)
        ids.append(input_ids)
        toks.append(masked_tokens)
        pos.append(masked_pos)
    return (
        torch.tensor(np.stack(ids), dtype=torch.float32),
        torch.tensor(np.stack(toks), dtype=torch.float32),
        torch.tensor(np.stack(pos), dtype=torch.long),
    )
