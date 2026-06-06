"""Downstream classification head + per-task training configs for the spectro tasks.

A single parameterized ``ClassificationHead`` (modeled on the channel pipeline's
``LosNlosClassificationHead``) serves all three tasks; only the class count and a few
training hyperparameters differ. Features are frozen embeddings (Transformer-MoE,
Mamba-MoE, or raw), so the head is the only thing trained during the sample sweep.
"""
from __future__ import annotations

import torch.nn as nn


class ClassificationHead(nn.Module):
    """MLP probe: Linear -> BN -> ReLU -> Dropout -> Linear over a feature vector."""

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


# Per-task training configuration for the frozen-feature probe.
TASK_CONFIGS = {
    'modulation': {'epochs': 150, 'lr': 1e-3, 'batch_size': 128, 'patience': 25,
                   'scheduler_step': 50, 'scheduler_gamma': 0.5, 'weight_decay': 1e-4},
    'snr':        {'epochs': 150, 'lr': 1e-3, 'batch_size': 128, 'patience': 25,
                   'scheduler_step': 50, 'scheduler_gamma': 0.5, 'weight_decay': 1e-4},
    'mobility':   {'epochs': 150, 'lr': 1e-3, 'batch_size': 128, 'patience': 25,
                   'scheduler_step': 50, 'scheduler_gamma': 0.5, 'weight_decay': 1e-4},
}
