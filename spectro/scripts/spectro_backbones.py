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


class ConvStem(nn.Module):
    """Per-patch conv+GELU stem replacing the bare ``Linear(element_length, d_model)`` tokenizer.

    Measured motivation. Modulation order is a fine-scale amplitude statistic, and the arms' scores
    track the granularity at which each model first applies a NONLINEARITY, not the number of tokens:

        MobileNetV3-S   1.7 px (3x3 conv @ stride 1.14 on the 224-upsampled image)  -> 0.341
        LWM patch 4     4 px, LINEAR, random init                                   -> 0.363
        ResNet-18       4.0 px (7x7 conv)                                           -> 0.328
        LWM patch 8     8 px, LINEAR                                                -> 0.302
        raw patch 8     8 px, LINEAR                                                -> 0.242

    Note the CV models have COARSER final grids than us (49 tokens vs 256/1024) -- their advantage
    is that BN+ReLU follows their first conv at 1.7-4 px, while our first op is a linear map over a
    whole 8x8 patch with no nonlinearity until after token mixing. This stem puts a 3x3 conv + GELU
    inside each patch, so a nonlinear local statistic exists BEFORE aggregation, at ~3 px
    granularity, without changing the token count or sequence length.

    Applied PER PATCH rather than over the whole image on purpose: a full-image conv would pull
    visible pixels across patch borders into masked patches during pretraining, leaking the
    reconstruction target through the 3x3 kernel.
    """

    def __init__(self, element_length: int, d_model: int, patch: int, hidden: int = 16):
        super().__init__()
        self.patch = patch
        self.in_ch = max(1, element_length // (patch * patch))
        self.element_length = element_length
        self.net = nn.Sequential(
            nn.Conv2d(self.in_ch, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.GELU(),
        )
        self.proj = nn.Linear(hidden * patch * patch, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, e = x.shape
        # tokens are row-major (row, col) within the patch, channels concatenated -> (B*T, C, p, p)
        z = x.reshape(b * t, self.in_ch, self.patch, self.patch)
        z = self.net(z).flatten(1)
        return self.proj(z).reshape(b, t, -1)

# Positional-embedding init std for the Transformer expert (see _patch_hf_embedding_second_order).
_POS_EMBED_STD = float(os.environ.get('SPECTRO_POS_EMBED_STD', '0.02'))
_hf_lwm_cls = None


def _patch_hf_attention_sdpa(mod):
    """Swap the HF LWM's manual softmax attention for memory-efficient SDPA (in place, at import time).

    The vendored ``spectro/hf_cache`` is gitignored and re-downloaded per environment as the ORIGINAL
    source, whose ``ScaledDotProductAttention`` materializes the full (B, heads, seq, seq) scores tensor
    (~4 GB/layer at batch 128, seq 1025 -> OOMs a 24 GB GPU). SDPA is numerically equivalent (default
    scale 1/sqrt(d_k)) but O(seq) memory, which is what lets both backbones pretrain at batch 128. We
    patch here — in tracked code — so the optimization travels with the repo instead of a gitignored edit.
    The returned attention weights are unused downstream, so we return None.
    """
    sdpa = getattr(nn.functional, 'scaled_dot_product_attention', None)
    if sdpa is None or not hasattr(mod, 'ScaledDotProductAttention'):
        return
    def forward(self, Q, K, V):
        return sdpa(Q, K, V), None
    mod.ScaledDotProductAttention.forward = forward


def _patch_hf_embedding_second_order(mod):
    """Give the vendored HF ``Embedding`` an optional [x, x^2] token projection.

    Measured motivation: modulation order is a WITHIN-token variance. The published embedding is
    ``nn.Linear(element_length, d_model)`` -- a linear map of the flattened patch -- and a linear
    function of 64 resource elements is Gaussian regardless of constellation, so the statistic is
    destroyed before the first attention layer. A per-layer probe of the pretrained transformer
    found modulation at CHANCE at every depth (best 0.273 vs its random-init control at 0.263,
    i.e. no pretraining lift anywhere), while mamba -- whose blocks contain a token-axis conv1d and
    multiplicative SiLU gating, both second-order ops -- reached 0.370 vs 0.284 by layer 7.
    Feeding raw tokens plus their squares to a linear probe lifts modulation 0.206 -> 0.298 at
    patch 8 and 0.191 -> 0.324 at patch 4, confirming the squared term is the missing ingredient.

    Patched here (tracked code) rather than in spectro/hf_cache, which is gitignored and
    re-downloaded per environment -- same rationale as the SDPA patch above.
    """
    if not hasattr(mod, 'Embedding'):
        return
    E = mod.Embedding
    if getattr(E, '_second_order_patched', False):
        return
    orig_init = E.__init__

    def __init__(self, element_length, d_model, max_len=None, second_order_embed=False,
                 pos_embed_std=_POS_EMBED_STD):
        orig_init(self, element_length, d_model, max_len)
        self.second_order_embed = second_order_embed
        if second_order_embed:
            self.proj = nn.Linear(element_length * 2, d_model)
        # nn.Embedding defaults to N(0,1). Measured: that makes the POSITIONAL signal ~75% of the
        # input variance (pos std 1.00 vs token std 0.578), and since LayerNorm normalizes the SUM,
        # attention has almost no content to key on -> softmax(QK^T) stays near-uniform and every
        # output token becomes ~the mean of all tokens. Modulation is a PER-TOKEN statistic, so it
        # is averaged away in layer 1. Random-init modulation probe vs pos-embed std:
        #     std 1.00 -> 0.253 | std 0.10 -> 0.295 | std 0.02 -> 0.316  (mamba random init: 0.309)
        # std 0.02 is the standard ViT/BERT initialization.
        if pos_embed_std is not None:
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=pos_embed_std)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}.")
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        xf = x.float()
        if getattr(self, 'second_order_embed', False):
            xf = torch.cat([xf, xf * xf], dim=-1)
        return self.norm(self.proj(xf) + self.pos_embed(pos))

    E.__init__ = __init__
    E.forward = forward
    E._second_order_patched = True


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
        _patch_hf_attention_sdpa(mod)         # memory-efficient attention so batch-128 fits on 24GB
        _patch_hf_embedding_second_order(mod)  # optional [x, x^2] token projection (see docstring)
        _hf_lwm_cls = mod.LWM
    return _hf_lwm_cls


class TransformerExpert(nn.Module):
    """Single-channel spectrogram Transformer expert (wraps the HF ``LWM``), mirroring
    ``lwm_mamba_spectro``'s interface so the MoE/pretraining code is architecture-agnostic."""

    def __init__(self, element_length=16, d_model=128, n_layers=12, max_len=1025,
                 n_heads=8, dropout=0.1, second_order_embed=False, conv_stem=False, patch=None):
        super().__init__()
        LWM = _load_hf_lwm()
        self.net = LWM(element_length=element_length, d_model=d_model, n_layers=n_layers,
                       max_len=max_len, n_heads=n_heads, dropout=dropout)
        if second_order_embed:
            # widen the token projection in place to consume [x, x^2]
            emb = self.net.embedding
            emb.second_order_embed = True
            emb.proj = nn.Linear(element_length * 2, d_model)
        if conv_stem:
            # swap the linear tokenizer for the per-patch conv stem; Embedding.forward calls
            # self.proj(x), so a module taking (B,T,E) -> (B,T,d) drops straight in
            self.net.embedding.proj = ConvStem(element_length, d_model, patch)

    def forward(self, input_ids, masked_pos=None):
        return self.net(input_ids, masked_pos)

    @torch.no_grad()
    def embed(self, input_ids, pool: str = "mean") -> torch.Tensor:
        return pool_tokens(self.net(input_ids), pool)


def pool_tokens(output: 'torch.Tensor', pool: str = "mean") -> 'torch.Tensor':
    """Pool encoder output (B, T, d) -> embedding. T = 1 CLS + n_patches (patches on a freq x time grid).

    - 'mean' / 'cls': (B, d).
    - 'meanstd_t': (B, 2d) = [mean over patches] ++ [std over TIME-blocks per freq, meaned over freq].
      The temporal-std term keeps the per-frequency time-variation that encodes Doppler/mobility,
      which plain mean-pooling discards.
    - 'seq': the full token sequence (B, T, d) unchanged (for the paper's 1-D CNN downstream head).
    """
    if pool == "seq":
        return output
    if pool == "cls":
        return output[:, 0]
    if pool == "mean":
        return output.mean(dim=1)
    if pool == "meanstd_t":
        enc = output[:, 1:]                                   # drop CLS -> (B, P, d)
        B, P, d = enc.shape
        mean = enc.mean(dim=1)
        side = int(round(P ** 0.5))
        if side * side == P:
            g = enc.reshape(B, side, side, d)                 # (B, freq, time, d)
            tstd = g.std(dim=2).mean(dim=1)                   # std over time per freq -> mean over freq
        else:
            tstd = enc.std(dim=1)
        return torch.cat([mean, tstd], dim=-1)                # (B, 2d)
    return output.mean(dim=1)


def build_expert(arch: str, *, d_model=128, n_layers=12, element_length=16, max_len=1025,
                 n_heads=8, dropout=0.1, use_fast_path=True, second_order_embed=None,
                 conv_stem=None, patch=None):
    """Construct one per-protocol expert of the requested architecture.

    ``n_heads`` is used only by the Transformer; ``use_fast_path`` only by the Mamba (set False for
    downstream LoRA finetuning so the SSM slow path exposes its projection weights to LoRA).
    """
    if second_order_embed is None:
        second_order_embed = os.environ.get('SPECTRO_SO_EMBED', '1') not in ('0', 'false', 'False')
    if conv_stem is None:
        conv_stem = os.environ.get('SPECTRO_CONV_STEM', '0') not in ('0', 'false', 'False')
    if conv_stem and patch is None:                       # infer the patch side from the geometry
        for c in (1, 2):
            side = int(round((element_length / c) ** 0.5))
            if side * side * c == element_length:
                patch = side
                break
    if conv_stem and second_order_embed:
        # the stem already supplies a per-patch nonlinearity; [x, x^2] on top is redundant and
        # would double the stem's input width for no measured benefit
        second_order_embed = False
    if arch == 'mamba':
        return lwm_mamba_spectro(element_length=element_length, d_model=d_model,
                                 n_layers=n_layers, max_len=max_len, dropout=dropout,
                                 use_fast_path=use_fast_path,
                                 second_order_embed=second_order_embed,
                                 conv_stem=conv_stem, patch=patch)
    if arch == 'transformer':
        return TransformerExpert(element_length=element_length, d_model=d_model,
                                 n_layers=n_layers, max_len=max_len, n_heads=n_heads,
                                 dropout=dropout, second_order_embed=second_order_embed,
                                 conv_stem=conv_stem, patch=patch)
    raise ValueError(f"Unknown arch {arch!r}; expected one of {ARCHS}")
