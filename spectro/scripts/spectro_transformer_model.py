"""Transformer (LWM-Spectro MoE) baseline feature provider.

The LWM-Spectro baseline is a 3-expert Transformer MoE + router pretrained on the full
(unavailable) corpus. Rather than reconstruct that inference stack (which needs the
``task1``/``task2`` module graph and expert checkpoints), we use the **precomputed
embeddings shipped in ``demo_data.pt``**:

- ``moe_embedding`` (128-d): the router top-1 MoE output — the faithful baseline features.
- ``tech_embedding`` (128-d): oracle (true-protocol) single-expert features.

This keeps the baseline robust and exactly the representation the model authors published.
``get_baseline_features`` returns the chosen (N, 128) matrix for the downstream sweep.
"""
from __future__ import annotations

import torch

from spectro_data import SpectroData


def get_baseline_features(data: SpectroData, which: str = "moe") -> torch.Tensor:
    """Return precomputed Transformer-MoE baseline features.

    Args:
        data: loaded ``SpectroData``.
        which: 'moe' (router-selected, default) or 'tech' (oracle-routed).
    """
    if which == "moe":
        return data.moe_embedding
    if which == "tech":
        return data.tech_embedding
    raise ValueError(f"Unknown baseline feature set: {which!r}")
