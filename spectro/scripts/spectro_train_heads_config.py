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

import torch
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

    ``second_order`` fixes a structural blind spot in the paper head. As published it is
    ``Conv1d(d_model, hidden, kernel_size=1)`` -- a LINEAR map over each token -- followed by a
    global average pool, i.e. a MEAN of features. But the statistic that separates modulation
    orders is an envelope VARIANCE (pre-channel: BPSK 0.44 / QPSK 0.24 / QAM16 0.51 / QAM64 0.58 /
    QAM256 0.59), and the statistic that separates mobility is a temporal autocorrelation. Neither
    is a mean, and a linear map of many resource elements is Gaussian regardless of constellation.
    Measured consequence on the old corpus: the ``raw`` arm held the full input losslessly and
    still read chance (0.19) on modulation, while a from-scratch CNN on the identical tensor
    reached 0.48 -- the information was present and the head could not form it. Convolutional
    baselines (ResNet's conv1+ReLU, DeepCNN) compute these moments for free, which is a large part
    of why generic vision models rivalled the domain model.

    Two changes, applied IDENTICALLY to every arm so the comparison stays fair:
      * ``[x, x^2]`` concatenated on the channel axis before ``proj`` -> a per-token nonlinearity,
        so a squared term exists before any pooling;
      * ``mean ++ std`` over tokens instead of mean alone -> the pooled readout can express a
        variance.

    Set ``second_order=False`` to recover the exact published head for an ablation.
    """

    def __init__(self, d_model: int, n_classes: int, hidden: int = 128, n_blocks: int = 2,
                 dropout: float = 0.1, second_order: bool = True):
        super().__init__()
        self.second_order = second_order
        in_ch = d_model * 2 if second_order else d_model
        self.proj = nn.Conv1d(in_ch, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[_ResidualConv1dBlock(hidden) for _ in range(n_blocks)])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2 if second_order else hidden, n_classes)

    def forward(self, x):
        x = x.float().transpose(1, 2)  # (N, T, d) -> (N, d, T); cast (sequences stored fp16 to save RAM)
        if self.second_order:
            x = torch.cat([x, x * x], dim=1)                    # per-token nonlinearity
        x = self.blocks(self.proj(x))
        if self.second_order:
            x = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)  # mean ++ std over tokens
        else:
            x = x.mean(dim=2)          # published head: global average pool -> (N, hidden)
        return self.fc(self.drop(x))


# Module-level switch for Conv1dHead.second_order, set once from the CLI (--head-variant) so every
# arm in a run gets the SAME head. Kept as a module global rather than threaded through _build_head's
# callers because the head is constructed deep inside run_sweep for each arm/task/restart.
HEAD_SECOND_ORDER = True


def set_head_second_order(flag: bool) -> None:
    global HEAD_SECOND_ORDER
    if HEAD_SECOND_ORDER != flag:
        print(f"[head] Conv1dHead second_order={flag} "
              f"({'[x,x^2] + mean/std pool' if flag else 'published head: linear proj + GAP'})")
    HEAD_SECOND_ORDER = flag


# Per-task training configuration for the frozen-feature probe (same schedule for every task).
_BASE_CFG = {'epochs': 150, 'lr': 1e-3, 'batch_size': 128, 'patience': 25,
             'scheduler_step': 50, 'scheduler_gamma': 0.5, 'weight_decay': 1e-4}
_TASK_NAMES = ('modulation', 'modulation3', 'snr_doppler', 'protocol', 'snr', 'mobility')
TASK_CONFIGS = {t: dict(_BASE_CFG) for t in _TASK_NAMES}
