"""Spectrogram Mixture-of-Experts: per-protocol experts (Mamba or Transformer) + a CNN router.

Mirrors LWM-Spectro's MoE structure (``spectro/hf_cache/mixture/train_embedding_router.py``):
- one expert per protocol (LTE/WiFi/5G), of a chosen architecture (``arch`` in {mamba, transformer}),
- a lightweight CNN ``RouterNet`` that selects the expert from the raw spectrogram (top-1).

``extract_embeddings`` produces a (N, d_model) routed embedding matrix for downstream probing,
mirroring the precomputed ``moe_embedding`` the Transformer baseline ships with.
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from spectro_backbones import build_expert  # noqa: E402
from spectro_patchify import spectrogram_patchify  # noqa: E402


class RouterNet(nn.Module):
    """Lightweight CNN router over (B,1,128,128) spectrograms (copied from LWM-Spectro)."""

    def __init__(self, num_experts: int, in_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96), nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        head: List[nn.Module] = [nn.Flatten()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(128, num_experts))
        self.classifier = nn.Sequential(*head)

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        x = specs
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected specs rank 3 or 4, got {tuple(specs.shape)}")
        return self.classifier(self.features(x))


def _normalize_per_sample(specs: torch.Tensor) -> torch.Tensor:
    mean = specs.mean(dim=(-2, -1), keepdim=True)
    std = torch.clamp(specs.std(dim=(-2, -1), keepdim=True, unbiased=False), min=1e-6)
    return (specs - mean) / std


class SpectroMoE(nn.Module):
    """Per-protocol experts (``arch``) + a router; produces routed spectrogram embeddings."""

    def __init__(self, protocols: List[str], d_model: int = 128, pool: str = "mean",
                 arch: str = "mamba", patch: int = 4, in_channels: int = 1, **expert_kwargs):
        super().__init__()
        self.protocols = list(protocols)
        self.d_model = d_model
        self.pool = pool
        self.arch = arch
        self.patch = patch                      # patchify granularity (must match the experts' element_length)
        self.in_channels = in_channels          # raw-spectrogram channels for the router (1 mag / 2 grid_stft|complex)
        self.experts = nn.ModuleDict({
            p: build_expert(arch, d_model=d_model, **expert_kwargs) for p in self.protocols
        })
        self.router = RouterNet(num_experts=len(self.protocols), in_channels=in_channels)

    # --- weight (de)serialization helpers -------------------------------------------------
    def load_expert(self, protocol: str, state_dict):
        self.experts[protocol].load_state_dict(state_dict)

    # --- embedding extraction -------------------------------------------------------------
    @torch.no_grad()
    def _expert_embed(self, protocol: str, specs: torch.Tensor) -> torch.Tensor:
        """Patchify + run a single expert on a (b,128,128) spectrogram batch -> (b,d_model)."""
        patches = spectrogram_patchify(specs, patch=self.patch, normalize=True)  # (b,n_patches,E)
        # prepend CLS token (0.2*ones, sized to the element_length) to match pretraining tokenization
        cls = np.full((patches.shape[0], 1, patches.shape[2]), 0.2, dtype=np.float32)
        input_ids = torch.tensor(np.concatenate([cls, patches], axis=1), dtype=torch.float32,
                                 device=specs.device)
        return self.experts[protocol].embed(input_ids, pool=self.pool)

    @torch.no_grad()
    def extract_embeddings(self, specs: torch.Tensor, *, routing: str = "router",
                           protocol_idx: np.ndarray | None = None, batch_size: int = 64,
                           device: str = "cuda") -> torch.Tensor:
        """Routed (N, d_model) embeddings for a stack of spectrograms.

        Args:
            specs: (N,128,128) spectrograms.
            routing: 'router' (use trained router argmax) or 'oracle' (use ``protocol_idx``).
            protocol_idx: (N,) true protocol indices, required for routing='oracle'.
            batch_size, device: batching controls.
        """
        device = device if torch.cuda.is_available() else "cpu"
        self.to(device).eval()
        n = specs.shape[0]
        emb_dim = self.d_model * (2 if self.pool == "meanstd_t" else 1)  # meanstd_t concats mean++temporal-std
        out = torch.empty(n, emb_dim, dtype=torch.float32)
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            batch = specs[sl].to(device).float()
            if routing == "oracle":
                assert protocol_idx is not None, "oracle routing needs protocol_idx"
                expert_ids = torch.as_tensor(protocol_idx[sl], device=device)
            else:
                logits = self.router(_normalize_per_sample(batch))
                expert_ids = logits.argmax(dim=1)
            # group by expert to batch each expert's forward pass
            for e_idx, proto in enumerate(self.protocols):
                mask = expert_ids == e_idx
                if not torch.any(mask):
                    continue
                emb = self._expert_embed(proto, batch[mask])
                out[torch.arange(start, start + batch.shape[0])[mask.cpu()]] = emb.cpu().float()
        return out


# Backward-compatible alias (the MoE now holds Mamba *or* Transformer experts).
MambaMoE = SpectroMoE
