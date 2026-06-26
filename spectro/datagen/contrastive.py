"""Contrastive objective for spectro pretraining, lifted from the LWM-Spectro authors' code.

``ProjectionHead`` and ``supervised_contrastive_loss`` are copied verbatim from
``spectro/hf_cache/pretraining/train_lwm_spectro_contrastive.py`` (wi-lab/lwm-spectro). The
authors pretrain with MLM + supervised contrastive (SupCon, Khosla et al. 2020) on modulation
and mobility labels; the SupCon term is what prevents the representation collapse we observed
with MLM-only pretraining.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """SimCLR-style projection head: avg-pool over sequence -> MLP -> L2-normalized embedding.
    (Verbatim from the LWM-Spectro authors' train_lwm_spectro_contrastive.py.)"""

    def __init__(self, d_model: int, projection_dim: int = 128, pool: str = "mean"):
        super().__init__()
        self.pool = pool      # 'mean'/'cls' (d_model) or 'meanstd_t' (2*d_model: keeps temporal/Doppler)
        in_dim = 2 * d_model if pool == "meanstd_t" else d_model
        self.projection = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim),
        )

    def forward(self, x):
        if self.pool == "meanstd_t":
            from spectro_backbones import pool_tokens   # lazy: avoids datagen<->scripts import cycle
            pooled = pool_tokens(x, "meanstd_t")        # (batch, 2*d_model)
        else:
            pooled = x[:, 0] if self.pool == "cls" else x.mean(dim=1)   # (batch, d_model)
        z = self.projection(pooled)
        return F.normalize(z, dim=1)


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.07, base_temperature: float = 0.07):
    """Supervised Contrastive Loss (SupCon, Khosla et al. 2020).
    (Verbatim from the LWM-Spectro authors' train_lwm_spectro_contrastive.py.)"""
    batch_size = embeddings.size(0)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    labels = labels.contiguous().view(-1, 1)
    mask_pos = torch.eq(labels, labels.T).float().to(embeddings.device)
    logits_mask = torch.scatter(
        torch.ones_like(mask_pos), 1,
        torch.arange(batch_size).view(-1, 1).to(embeddings.device), 0)
    mask_pos = mask_pos * logits_mask
    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    return loss.mean()
