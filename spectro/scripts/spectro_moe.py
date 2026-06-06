"""Mamba Mixture-of-Experts: per-protocol Mamba experts + a CNN router.

Mirrors LWM-Spectro's MoE structure (``spectro/hf_cache/mixture/train_embedding_router.py``):
- one bidirectional Mamba expert per protocol (LTE/WiFi/5G),
- a lightweight CNN ``RouterNet`` that selects the expert from the raw spectrogram (top-1).

``extract_embeddings`` produces a (N, d_model) routed embedding matrix for downstream probing,
mirroring the precomputed ``moe_embedding`` the Transformer baseline ships with.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from spectro_mamba_model import lwm_mamba_spectro  # noqa: E402
from spectro_patchify import spectrogram_patchify  # noqa: E402


class RouterNet(nn.Module):
    """Lightweight CNN router over (B,1,128,128) spectrograms (copied from LWM-Spectro)."""

    def __init__(self, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
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


class MambaMoE(nn.Module):
    """Per-protocol Mamba experts + a router; produces routed spectrogram embeddings."""

    def __init__(self, protocols: List[str], d_model: int = 128, pool: str = "mean",
                 **expert_kwargs):
        super().__init__()
        self.protocols = list(protocols)
        self.d_model = d_model
        self.pool = pool
        self.experts = nn.ModuleDict({
            p: lwm_mamba_spectro(d_model=d_model, **expert_kwargs) for p in self.protocols
        })
        self.router = RouterNet(num_experts=len(self.protocols))

    # --- weight (de)serialization helpers -------------------------------------------------
    def load_expert(self, protocol: str, state_dict):
        self.experts[protocol].load_state_dict(state_dict)

    # --- embedding extraction -------------------------------------------------------------
    @torch.no_grad()
    def _expert_embed(self, protocol: str, specs: torch.Tensor) -> torch.Tensor:
        """Patchify + run a single expert on a (b,128,128) spectrogram batch -> (b,d_model)."""
        patches = spectrogram_patchify(specs, normalize=True)        # (b,1024,16)
        # prepend CLS token (0.2*ones) to match pretraining tokenization
        from spectro_patchify import CLS_TOKEN
        cls = np.broadcast_to(CLS_TOKEN, (patches.shape[0], 1, patches.shape[2]))
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
        out = torch.empty(n, self.d_model, dtype=torch.float32)
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
