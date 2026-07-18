"""Single-channel bidirectional Mamba backbone for spectrogram patches.

Adapted from ``scripts/mamba_model.py:lwm_mamba`` (single-resolution path only) and built
on the shared ``MambaLayer`` primitive. Input tokens are 4x4 real spectrogram patches
(``element_length=16``); a CLS token sits at position 0, so ``max_len=1025`` for 128x128.

Forward contract matches the HF Transformer ``LWM``:
- ``forward(input_ids, masked_pos)`` -> ``(logits_lm, output)`` for masked-token pretraining,
- ``forward(input_ids)`` -> ``output`` of shape (B, T, d_model).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from shared.mamba_layers import MambaLayer  # noqa: E402


class lwm_mamba_spectro(nn.Module):
    """Bidirectional Mamba LWM over single-channel spectrogram patches.

    No positional encoding: Mamba's selective state-space recurrence captures order.

    Args:
        element_length (int): Input token width (4*4 = 16 for single-channel 4x4 patches).
        d_model (int): Embedding dimension.
        n_layers (int): Number of Mamba layers.
        max_len (int): Maximum sequence length (1024 patches + 1 CLS = 1025).
        d_state, d_conv, expand: Mamba SSM hyperparameters.
        dropout (float): Dropout probability.
        bidirectional (bool): Bidirectional Mamba (default True for MLM).
    """

    def __init__(self, element_length=16, d_model=128, n_layers=12, max_len=1025,
                 d_state=16, d_conv=4, expand=2, dropout=0.1, bidirectional=True, use_fast_path=True):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_len = max_len
        self.bidirectional = bidirectional

        self.proj = nn.Linear(element_length, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # use_fast_path=False (downstream LoRA finetuning) forces the SSM slow path so weight-
        # parametrization LoRA on the SSM projections is applied; pretraining keeps True for speed.
        self.layers = nn.ModuleList([
            MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                       d_ff=d_model * 4, dropout=dropout, bidirectional=bidirectional,
                       use_fast_path=use_fast_path)
            for _ in range(n_layers)
        ])

        # Masked-token prediction head
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, element_length, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(element_length))

    def forward(self, input_ids, masked_pos=None):
        output = self.proj(input_ids.float())
        output = self.input_norm(output)
        for layer in self.layers:
            output = layer(output)

        if masked_pos is not None:
            masked_pos = masked_pos.long()[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(F.gelu(self.linear(h_masked)))
            logits_lm = self.decoder(h_masked) + self.decoder_bias
            return logits_lm, output
        return output

    @torch.no_grad()
    def embed(self, input_ids, pool: str = "mean") -> torch.Tensor:
        """Extract an embedding. ``pool`` in {'mean','cls','meanstd_t'} (meanstd_t -> 2*d_model)."""
        from spectro_backbones import pool_tokens   # lazy import to avoid the build_expert cycle
        return pool_tokens(self.forward(input_ids), pool)
