"""Backbone factory for the spectro MoE: a Mamba or a Transformer expert behind one interface.

Both expert types expose the same contract the MoE/pretraining/extraction code relies on:
    forward(input_ids, masked_pos=None) -> (logits_lm, output) | output   # (B,T,16) -> (B,T,d_model)
    embed(input_ids, pool="mean"|"cls") -> (B, d_model)

The Transformer expert wraps the HF ``LWM`` class shipped in ``spectro/hf_cache`` (downloaded from
wi-lab/lwm-spectro) so the synthetic-pretrained Transformer is the *same architecture* as the
baseline — letting us pretrain it exactly like the Mamba and compare backbones fairly.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectro_mamba_model import lwm_mamba_spectro  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_HF_LWM_PATH = os.path.join(_REPO_ROOT, 'spectro', 'hf_cache', 'pretraining', 'pretrained_model.py')

ARCHS = ('mamba', 'transformer')
_hf_lwm_cls = None


def _load_hf_lwm():
    """Import the HF Transformer ``LWM`` class from the downloaded source (cached)."""
    global _hf_lwm_cls
    if _hf_lwm_cls is None:
        if not os.path.exists(_HF_LWM_PATH):
            raise FileNotFoundError(
                f"HF LWM source not found at {_HF_LWM_PATH}. Run "
                "`python spectro/scripts/download_spectro_hf.py --skip-weights` first.")
        spec = importlib.util.spec_from_file_location('hf_spectro_pretrained_model', _HF_LWM_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _hf_lwm_cls = mod.LWM
    return _hf_lwm_cls


class TransformerExpert(nn.Module):
    """Single-channel spectrogram Transformer expert (wraps the HF ``LWM``), mirroring
    ``lwm_mamba_spectro``'s interface so the MoE/pretraining code is architecture-agnostic."""

    def __init__(self, element_length=16, d_model=128, n_layers=12, max_len=1025,
                 n_heads=8, dropout=0.1):
        super().__init__()
        LWM = _load_hf_lwm()
        self.net = LWM(element_length=element_length, d_model=d_model, n_layers=n_layers,
                       max_len=max_len, n_heads=n_heads, dropout=dropout)

    def forward(self, input_ids, masked_pos=None):
        return self.net(input_ids, masked_pos)

    @torch.no_grad()
    def embed(self, input_ids, pool: str = "mean") -> torch.Tensor:
        output = self.net(input_ids)
        return output[:, 0] if pool == "cls" else output.mean(dim=1)


def build_expert(arch: str, *, d_model=128, n_layers=12, element_length=16, max_len=1025,
                 n_heads=8, dropout=0.1):
    """Construct one per-protocol expert of the requested architecture.

    ``n_heads`` is used only by the Transformer; the Mamba ignores it.
    """
    if arch == 'mamba':
        return lwm_mamba_spectro(element_length=element_length, d_model=d_model,
                                 n_layers=n_layers, max_len=max_len, dropout=dropout)
    if arch == 'transformer':
        return TransformerExpert(element_length=element_length, d_model=d_model,
                                 n_layers=n_layers, max_len=max_len, n_heads=n_heads,
                                 dropout=dropout)
    raise ValueError(f"Unknown arch {arch!r}; expected one of {ARCHS}")
