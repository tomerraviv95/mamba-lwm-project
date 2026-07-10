"""Downstream classification heads + per-task training configs for the spectro tasks.

Two heads:
- ``ClassificationHead`` — MLP probe over a pooled feature VECTOR (N, d). Used by the vector-feature
  arms (raw, ImageNet backbones) and as a lightweight option for the MoE arms.
- ``Conv1dHead`` — the LWM-Spectro PAPER downstream head: a residual 1-D CNN over the encoder TOKEN
  SEQUENCE (N, T, d) followed by global average pooling and a linear classifier. Used by the MoE arms
  when features are extracted as token sequences.

Features are frozen (Transformer-MoE / Mamba-MoE / raw / ImageNet), so the head is the only thing
trained during the sample sweep (except the end-to-end Deep CNN baseline, which trains its backbone).
"""
from __future__ import annotations

import torch.nn as nn


class ClassificationHead(nn.Module):
    """MLP probe: Linear -> BN -> ReLU -> Dropout -> Linear over a feature vector (N, d)."""

    def __init__(self, input_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class _ResidualConv1dBlock(nn.Module):
    """Conv1d -> BN -> GELU -> Conv1d -> BN, with a residual add then GELU (channels preserved)."""

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.c1 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.b1 = nn.BatchNorm1d(channels)
        self.c2 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.b2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        r = x
        x = self.act(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return self.act(x + r)


class Conv1dHead(nn.Module):
    """Paper downstream head: residual 1-D CNN over the token sequence (N, T, d) -> GAP -> linear.

    Input is the encoder output token sequence with ``d`` treated as channels and ``T`` as the 1-D
    length, so the CNN models local dependencies along the token axis before pooling — unlike a plain
    mean/meanstd pool + MLP, which collapses the sequence first.
    """

    def __init__(self, d_model: int, n_classes: int, hidden: int = 128, n_blocks: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(d_model, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[_ResidualConv1dBlock(hidden) for _ in range(n_blocks)])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.float().transpose(1, 2)  # (N, T, d) -> (N, d, T); cast (sequences stored fp16 to save RAM)
        x = self.blocks(self.proj(x))
        x = x.mean(dim=2)              # global average pool over tokens -> (N, hidden)
        return self.fc(self.drop(x))


# Per-task training configuration for the frozen-feature probe (same schedule for every task).
_BASE_CFG = {'epochs': 150, 'lr': 1e-3, 'batch_size': 128, 'patience': 25,
             'scheduler_step': 50, 'scheduler_gamma': 0.5, 'weight_decay': 1e-4}
_TASK_NAMES = ('modulation', 'snr_doppler', 'protocol', 'snr', 'mobility')
TASK_CONFIGS = {t: dict(_BASE_CFG) for t in _TASK_NAMES}
