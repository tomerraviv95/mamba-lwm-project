"""Minimal weight-parametrization LoRA for the custom spectro backbones.

Hand-rolled (no ``peft``) because these are custom, non-HuggingFace modules and — critically — the
Mamba SSM reads its projection ``.weight`` tensors DIRECTLY (``in_proj``/``dt_proj`` even in the slow
path, and all four SSM linears in the fused fast path). A ``.forward()``-wrapping LoRA would be a
SILENT no-op on those. So we adapt at the **weight** level via ``torch.nn.utils.parametrize``:

    W_adapted = W0 + (alpha / r) * (B @ A)

registered as a parametrization on each target Linear's ``weight``. Any read of ``.weight`` — module
call OR raw-tensor access by a fused kernel — sees the adapted weight, and gradients flow to A/B.

Budget convention: the LoRA ADAPTERS are capped at a fraction of the backbone params; the task head is
trainable on top and NOT counted (standard LoRA finetuning).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class _LoRADelta(nn.Module):
    """Parametrization module: maps the original weight W0 -> W0 + scaling * (B @ A).

    A is kaiming-init, B is zero -> the adapter starts as an exact no-op (W_adapted == W0), so a freshly
    wrapped model reproduces the pretrained backbone until training moves B off zero.
    """

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float | None = None):
        super().__init__()
        assert rank > 0
        out_f, in_f = weight.shape
        dev, dt = weight.device, weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scaling = (alpha if alpha is not None else rank) / rank

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W + self.scaling * (self.lora_B @ self.lora_A)


# parent-class-name -> child attr names to adapt. Keyed by ``type(module).__name__`` so we don't have to
# import the dynamically-loaded HF ``LWM`` module. Note the parent-class guard deliberately EXCLUDES the
# top-level MLM head ``.linear`` (a child of LWM / lwm_mamba_spectro, not of an attention/FFN block).
TRANSFORMER_TARGETS = {
    'MultiHeadAttention': {'W_Q', 'W_K', 'W_V', 'linear'},   # Q,K,V + attention output projection
    'PoswiseFeedForwardNet': {'fc1', 'fc2'},                 # FFN
}
# Mamba: custom bidirectional block I/O projections, the FFN Sequential's two Linears (the expert's only
# nn.Sequential), and the four SSM projections inside each mamba_ssm ``Mamba`` (class name 'Mamba').
MAMBA_TARGETS = {
    'MambaBlock': {'input_proj', 'output_proj'},
    'Mamba': {'in_proj', 'x_proj', 'dt_proj', 'out_proj'},
    'Sequential': '*',                                       # MambaLayer.ffn -> its Linear children
}
TARGETS = {'transformer': TRANSFORMER_TARGETS, 'mamba': MAMBA_TARGETS}


def _iter_targets(root: nn.Module, rules: dict):
    """Yield (parent_module, child_name, child_linear) for every nn.Linear matched by ``rules``."""
    for module in root.modules():
        want = rules.get(type(module).__name__)
        if want is None:
            continue
        for cname, child in module.named_children():
            if isinstance(child, nn.Linear) and (want == '*' or cname in want):
                yield module, cname, child


def backbone_param_count(root: nn.Module) -> int:
    """Total parameter count of the (un-wrapped) backbone — the denominator for the LoRA budget."""
    return sum(p.numel() for p in root.parameters())


def lora_params_at_rank(root: nn.Module, rules: dict, rank: int) -> int:
    """Number of trainable LoRA params (A + B over every target linear) at a given rank."""
    return sum(rank * (c.in_features + c.out_features) for _, _, c in _iter_targets(root, rules))


def pick_rank(root: nn.Module, rules: dict, budget_frac: float = 0.05, max_rank: int = 16) -> int:
    """Largest rank in [1, max_rank] whose adapter params <= budget_frac * backbone params (>= 1)."""
    per_rank = lora_params_at_rank(root, rules, 1)
    if per_rank == 0:
        return 0
    r = int((budget_frac * backbone_param_count(root)) // per_rank)
    return max(1, min(max_rank, r))


def apply_lora(root: nn.Module, rules: dict, rank: int, alpha: float | None = None) -> list[str]:
    """Freeze ``root`` and register a LoRA weight-parametrization on every matched Linear (in place).

    Returns the list of adapted "ParentClass.child" names. After this call, the ONLY trainable params in
    ``root`` are the LoRA A/B matrices (originals + biases are frozen).
    """
    for p in root.parameters():
        p.requires_grad_(False)                 # freeze the whole backbone first
    adapted = []
    for parent, cname, child in list(_iter_targets(root, rules)):
        parametrize.register_parametrization(child, 'weight', _LoRADelta(child.weight, rank, alpha))
        adapted.append(f"{type(parent).__name__}.{cname}")
    return adapted


def trainable_param_count(root: nn.Module) -> int:
    return sum(p.numel() for p in root.parameters() if p.requires_grad)
