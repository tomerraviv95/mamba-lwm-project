"""Generic frozen-backbone probing / fine-tuning harness.

Extracted and generalized from ``scripts/train_heads.py`` so it works for any backbone
that maps an input batch to features plus any task head. The channel pipeline keeps its
own (tightly task-coupled) copy for now; the spectro pipeline uses this generic version.

Design:
- ``FineTuningWrapper`` wraps an optional ``backbone`` + a ``head``. An ``embed_fn``
  turns ``(backbone, batch)`` into a feature tensor the head consumes. For frozen-feature
  probing the backbone can be ``None`` and ``embed_fn`` simply returns the batch.
- ``finetune`` runs a standard train/val/early-stop loop and evaluates a test split via a
  caller-supplied ``score_fn`` (so scoring stays domain-specific: accuracy, NMSE, etc.).
- ``subsample_training_data`` deterministically selects a fraction of a training set,
  matching the channel pipeline's behaviour (seeded permutation).
"""
from __future__ import annotations

import copy
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def subsample_training_data(n_total: int, pct: float, seed: int = 42):
    """Deterministically pick ``round(pct * n_total)`` indices (>=1).

    Returns a sorted ``np.ndarray`` of indices, mirroring the seeded-permutation
    subsampling used by the channel pipeline so runs are reproducible across arms.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    k = max(1, int(round(pct * n_total)))
    return np.sort(perm[:k])


class FineTuningWrapper(nn.Module):
    """Optional (frozen) backbone + task head, joined by an ``embed_fn``.

    Args:
        head: the task-specific head module (consumes features, emits logits/predictions).
        backbone: optional feature extractor. If provided it is frozen unless
            ``fine_tune_layers`` matches parameter-name substrings.
        embed_fn: ``(backbone, batch) -> features``. Defaults to identity on the batch
            (frozen precomputed-feature probing).
        fine_tune_layers: ``"full"`` to unfreeze the whole backbone, or a list of name
            substrings to unfreeze matching params, or ``None`` to keep it fully frozen.
    """

    def __init__(self, head: nn.Module, backbone: Optional[nn.Module] = None,
                 embed_fn: Optional[Callable] = None, fine_tune_layers=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.embed_fn = embed_fn if embed_fn is not None else (lambda bb, x: x)

        if backbone is not None:
            for p in backbone.parameters():
                p.requires_grad = False
            if fine_tune_layers == "full":
                for p in backbone.parameters():
                    p.requires_grad = True
            elif isinstance(fine_tune_layers, (list, tuple)):
                for name, p in backbone.named_parameters():
                    if any(key in name for key in fine_tune_layers):
                        p.requires_grad = True

    def forward(self, batch):
        feats = self.embed_fn(self.backbone, batch)
        return self.head(feats)


def _to_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool):
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=shuffle)


def finetune(
    head: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    score_fn: Callable[[torch.Tensor, torch.Tensor], float],
    criterion: Optional[nn.Module] = None,
    backbone: Optional[nn.Module] = None,
    embed_fn: Optional[Callable] = None,
    fine_tune_layers=None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    patience: int = 20,
    scheduler_step: Optional[int] = None,
    scheduler_gamma: float = 0.5,
    device: str = "cuda",
    verbose: bool = False,
):
    """Train ``head`` (optionally fine-tuning ``backbone``) and evaluate on the test split.

    ``score_fn(preds, labels) -> float`` computes the domain metric (higher is better);
    the best model by validation score is restored before the final test evaluation.

    Returns ``(wrapper, history, test_score, test_ground_truth, test_predictions)``.
    """
    device = device if torch.cuda.is_available() else "cpu"
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    wrapper = FineTuningWrapper(head, backbone=backbone, embed_fn=embed_fn,
                                fine_tune_layers=fine_tune_layers).to(device)

    params = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = None
    if scheduler_step:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step,
                                                    gamma=scheduler_gamma)

    train_loader = _to_loader(train_features, train_labels, batch_size, shuffle=True)
    history = {"train_loss": [], "val_score": []}
    best_val = -float("inf")
    best_state = copy.deepcopy(wrapper.state_dict())
    patience_ctr = 0

    val_features_d = val_features.to(device)
    for epoch in range(epochs):
        wrapper.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = wrapper(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        if scheduler:
            scheduler.step()
        history["train_loss"].append(running / max(len(train_loader.dataset), 1))

        wrapper.eval()
        with torch.no_grad():
            val_out = wrapper(val_features_d).cpu()
        val_score = score_fn(val_out, val_labels)
        history["val_score"].append(val_score)

        if val_score > best_val + 1e-6:
            best_val = val_score
            best_state = copy.deepcopy(wrapper.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f"  early stop @ epoch {epoch+1}, best val {best_val:.4f}")
                break

    wrapper.load_state_dict(best_state)
    wrapper.eval()
    with torch.no_grad():
        test_out = wrapper(test_features.to(device)).cpu()
    test_score = score_fn(test_out, test_labels)
    return wrapper, history, test_score, test_labels, test_out
