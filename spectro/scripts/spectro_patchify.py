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


def patch_geometry(patch: int = PATCH, channels: int = 1, img: int = 128) -> dict:
    """Token geometry for a patch size: element_length, n_patches, max_len (+CLS), grid side.

    img=128, patch 4 -> 32x32=1024 patches, element 16, max_len 1025; patch 6 -> 21x21=441/36/442;
    patch 8 -> 16x16=256/64/257. (channels=2 for complex doubles element_length.)
    """
    side = img // patch
    n_patches = side * side
    return {"element_length": patch * patch * channels, "n_patches": n_patches,
            "max_len": n_patches + 1, "side": side}


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
                        rng: np.random.Generator | None = None,
                        masked_pos: np.ndarray | None = None):
    """Prepend CLS and (optionally) apply 80/10/10 BERT masking to one sample's patches.

    Args:
        patches: (n_patches, element_length) tokens for a single spectrogram.
        n_masks: number of patch positions to mask (ignored if ``masked_pos`` is given).
        mask: if False, just prepend CLS and return the token tensor.
        rng: numpy Generator for reproducibility.
        masked_pos: optional explicit 1-based positions to mask (e.g. time-column masking computed by
            the caller). When provided it overrides the random ``n_masks`` selection.

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
    if masked_pos is not None:
        masked_pos = np.asarray(masked_pos, dtype=np.int64)
    elif n_masks <= 0 or n_patches == 0:
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


def time_column_positions(side: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """1-based patch positions covering a random subset of TIME columns of a side x side patch grid.

    Patches are row-major (freq_block, time_block): patch_idx = f*side + t (0-based), so a time
    column t = {f*side + t : f in 0..side-1}. Masking whole time columns (instead of random patches)
    forces MLM to reconstruct a missing time slice from its temporal neighbours -> the model must
    model how the channel evolves in time (Doppler), which random-patch masking does not require.
    +1 converts to 1-based positions (CLS is at 0)."""
    n_cols = max(1, int(round(frac * side)))
    cols = rng.choice(side, size=min(n_cols, side), replace=False)
    f = np.arange(side)
    pos = (f[:, None] * side + cols[None, :]).reshape(-1)        # (side * n_cols,)
    return np.sort(pos.astype(np.int64)) + 1


def build_masked_tensors(specs, mask_percent: float = 0.6, seed: int = 42, patch: int = PATCH,
                         mask_mode: str = "random", half: bool = False):
    """Build stacked (input_ids, masked_tokens, masked_pos) tensors for MLM pretraining.

    All 128x128 spectrograms share the same n_patches and a constant n_masks (random mode) or a
    constant masked-column count (time_col mode), so the per-sample outputs stack cleanly into
    TensorDataset-ready tensors.

    Args:
        mask_mode: 'random' (BERT 4x4-patch masking, default) or 'time_col' (mask whole time columns
            of the patch grid -> temporal/Doppler pretext, see ``time_column_positions``).
        half: store ``input_ids``/``masked_tokens`` as float16 (cast to float32 at use). Halves the
            RAM footprint — needed for large corpora (~40k samples/expert would otherwise peak >25 GB).
            The random-mode path also preallocates and uses ``torch.from_numpy`` (zero-copy) to avoid
            the list->stack->tensor 3x transient peak. Cast back to float() before the model.

    Returns:
        ``(input_ids, masked_tokens, masked_pos)`` torch tensors; for patch 4: (N,1025,16),
        (N,n_masks,16), (N,n_masks).
    """
    patches = spectrogram_patchify(specs, patch=patch, normalize=True)
    n_patches = patches.shape[1]
    n_masks = max(1, int(mask_percent * n_patches))
    rng = np.random.default_rng(seed)
    side = int(round(n_patches ** 0.5))
    if mask_mode == "time_col" and side * side != n_patches:
        raise ValueError(f"time_col masking needs a square patch grid; got n_patches={n_patches}")

    store = np.float16 if half else np.float32
    if mask_mode == "random":                        # common path: preallocate + zero-copy (low peak)
        n, elem = patches.shape[0], patches.shape[2]
        ids = np.empty((n, n_patches + 1, elem), dtype=store)
        toks = np.empty((n, n_masks, elem), dtype=store)
        pos = np.empty((n, n_masks), dtype=np.int64)
        for i in range(n):
            a, b, c = make_sample_spectro(patches[i], n_masks, mask=True, rng=rng, masked_pos=None)
            ids[i], toks[i], pos[i] = a, b, c
        return torch.from_numpy(ids), torch.from_numpy(toks), torch.from_numpy(pos)

    ids, toks, pos = [], [], []                      # time_col path (rare pretext): keep list build
    for p in patches:
        mp = time_column_positions(side, mask_percent, rng)
        input_ids, masked_tokens, masked_pos = make_sample_spectro(p, n_masks, mask=True, rng=rng,
                                                                   masked_pos=mp)
        ids.append(input_ids)
        toks.append(masked_tokens)
        pos.append(masked_pos)
    tdt = torch.float16 if half else torch.float32
    return (
        torch.tensor(np.stack(ids), dtype=tdt),
        torch.tensor(np.stack(toks), dtype=tdt),
        torch.tensor(np.stack(pos), dtype=torch.long),
    )
