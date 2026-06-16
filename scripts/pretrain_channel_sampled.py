"""Pretrain the channel-domain LWM (Transformer or Mamba) on a token-balanced DeepMIMO sample.

Samples ~58k channel realizations across the 20 LWM cities so that each city contributes the
same number of 4x4 patch-tokens (default 500k/city -> ~10M tokens total). Because cities differ
in antenna x subcarrier size, patches-per-realization vary 32x, so we sample *inversely* to a
city's patch count (``realizations = TOKENS_PER_CITY / patches_per_realization``). This gives
every propagation environment equal weight while hitting a fixed token budget.

Reuses the original channel machinery in ``scripts/utils.py`` (DeepMIMO generation, tokenizer,
masked-modeling ``train_lwm``) and the model classes in ``pretrained_model.py`` / ``mamba_model.py``.

Usage::

    python scripts/pretrain_channel_sampled.py --arch transformer
    python scripts/pretrain_channel_sampled.py --arch mamba
    python scripts/pretrain_channel_sampled.py --smoke        # tiny sanity run (no DeepMIMO if --fake)
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mamba_model  # noqa: E402
import pretrained_model  # noqa: E402
from utils import generate_channels_and_labels, tokenizer_train, train_lwm  # noqa: E402

PATCH = 4
ELEMENT_LENGTH = PATCH * PATCH * 2          # 32: real/imag interleaved 4x4 patches
MAX_LEN = 1100                              # > max seq len (1024 patches + CLS = 1025)
MASK_PERCENT = 0.40

# The 20 LWM cities: scenario name + base-station antenna count + subcarriers (grid_idx=1).
# (Values mirror scripts/train_lwm.py's scenario table.)
CITY_CONFIG = [
    ("city_0_newyork_3p5_lwm", 8, 32),    ("city_1_losangeles_3p5_lwm", 8, 64),
    ("city_2_chicago_3p5_lwm", 8, 128),   ("city_3_houston_3p5_lwm", 8, 256),
    ("city_4_phoenix_3p5_lwm", 8, 512),   ("city_5_philadelphia_3p5_lwm", 8, 1024),
    ("city_6_miami_3p5_lwm", 16, 32),     ("city_7_sandiego_3p5_lwm", 16, 64),
    ("city_8_dallas_3p5_lwm", 16, 128),   ("city_9_sanfrancisco_3p5_lwm", 16, 256),
    ("city_10_austin_3p5_lwm", 16, 512),  ("city_11_santaclara_3p5_lwm", 32, 32),
    ("city_12_fortworth_3p5_lwm", 32, 64),("city_13_columbus_3p5_lwm", 32, 128),
    ("city_14_charlotte_3p5_lwm", 32, 256),("city_15_indianapolis_3p5_lwm", 64, 32),
    ("city_16_sanfrancisco_3p5_lwm", 64, 64),("city_17_seattle_3p5_lwm", 64, 128),
    ("city_18_denver_3p5_lwm", 128, 32),  ("city_19_oklahoma_3p5_lwm", 128, 64),
]
GRID_IDX = 1


def patches_per_realization(n_ant: int, n_sub: int, patch: int = PATCH) -> int:
    """Number of 4x4 patches in one channel.

    ``patch_maker`` tiles 4x4 patches over the original (n_ant x n_subcarrier) grid; the real/imag
    interleaving lives *inside* each patch (4*4*2 = element_length 32), so the patch GRID is not
    doubled. Hence patches = ceil(n_ant/p) * ceil(n_sub/p) (verified against tokenizer output)."""
    return math.ceil(n_ant / patch) * math.ceil(n_sub / patch)


def plan_sampling(tokens_per_city: int):
    """Return [(scenario, n_ant, n_sub, n_realizations, patches, tokens), ...] for the token budget."""
    plan = []
    for scn, n_ant, n_sub in CITY_CONFIG:
        p = patches_per_realization(n_ant, n_sub)
        n = max(1, round(tokens_per_city / p))
        plan.append((scn, n_ant, n_sub, n, p, n * p))
    return plan


def subsample(channels: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Random subset of ``n`` channel realizations (or all, if fewer)."""
    m = channels.shape[0]
    if n >= m:
        return channels
    idx = np.random.RandomState(seed).permutation(m)[:n]
    return channels[torch.as_tensor(np.sort(idx))]


def gather_city_channels(scn, n_ant, n_sub, n, bs_idx, seed):
    """Generate a city's channels via DeepMIMO and return a random ``n``-realization subset."""
    channels, _ = generate_channels_and_labels(
        n_ant_bs=n_ant, n_subcarriers=n_sub, bs_idx=bs_idx, grid_idx=GRID_IDX,
        scenario_name=scn, rows=None, task=None)
    return subsample(channels, n, seed)


def tokenize_cities(channel_sets, seed):
    """Tokenize each city's channels (masked) and merge into one {seq_len: [samples]} dict."""
    from collections import defaultdict
    merged = defaultdict(list)
    np.random.seed(seed)
    for ch in channel_sets:
        grouped = tokenizer_train([ch], masking_percent=MASK_PERCENT, mask=True, seed=seed)
        for seq_len, samples in grouped.items():
            merged[seq_len].extend(samples)
    return merged


def build_loaders(merged, batch_size, val_frac=0.2, seed=42):
    """Per-seq-length in-memory train/val DataLoaders of (input_ids, masked_tokens, masked_pos)."""
    train_loaders, val_loaders = {}, {}
    for seq_len, samples in merged.items():
        ids = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32)
        toks = torch.tensor(np.stack([np.stack(s[1]) for s in samples]), dtype=torch.float32)
        pos = torch.tensor(np.stack([s[2] for s in samples]), dtype=torch.long)
        n = ids.shape[0]
        perm = np.random.RandomState(seed).permutation(n)
        n_val = max(1, int(val_frac * n))
        vi, ti = perm[:n_val], perm[n_val:]
        train_loaders[seq_len] = DataLoader(TensorDataset(ids[ti], toks[ti], pos[ti]),
                                            batch_size=batch_size, shuffle=True)
        val_loaders[seq_len] = DataLoader(TensorDataset(ids[vi], toks[vi], pos[vi]),
                                          batch_size=batch_size, shuffle=False)
    return train_loaders, val_loaders


def build_model(arch, d_model, n_layers, device):
    """Single-resolution channel LWM (element_length=32) of the requested architecture."""
    if arch == "mamba":
        model = mamba_model.lwm_mamba(element_length=ELEMENT_LENGTH, d_model=d_model,
                                      n_layers=n_layers, max_len=MAX_LEN, patch_sizes=None)
    elif arch == "transformer":
        model = pretrained_model.lwm(element_length=ELEMENT_LENGTH, d_model=d_model,
                                     n_layers=n_layers, max_len=MAX_LEN, patch_sizes=None)
    else:
        raise ValueError(arch)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['transformer', 'mamba'], default='transformer')
    ap.add_argument('--tokens-per-city', type=int, default=500_000, help='~10M total over 20 cities')
    ap.add_argument('--bs-idx', type=int, default=1)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-layers', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--warmup-epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--max-batches-per-epoch', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.smoke:
        args.tokens_per_city, args.n_layers, args.epochs, args.warmup_epochs = 4000, 2, 1, 0

    plan = plan_sampling(args.tokens_per_city)
    total_real = sum(p[3] for p in plan)
    total_tok = sum(p[5] for p in plan)
    print(f"Sampling plan ({args.tokens_per_city} tokens/city): "
          f"{total_real} realizations -> ~{total_tok/1e6:.2f}M tokens")
    for scn, a, s, n, pp, tok in plan:
        print(f"  {scn:32s} {a:3d}x{s:<4d} patches={pp:5d}  n={n:6d}  tokens={tok}")

    print("\nGenerating + tokenizing channels (DeepMIMO) ...")
    channel_sets = []
    for scn, n_ant, n_sub, n, _, _ in plan:
        ch = gather_city_channels(scn, n_ant, n_sub, n, args.bs_idx, args.seed)
        print(f"  {scn}: kept {ch.shape[0]} realizations of shape {tuple(ch.shape[1:])}")
        channel_sets.append(ch)

    merged = tokenize_cities(channel_sets, args.seed)
    print(f"\nSeq-length groups: {{ {', '.join(f'{k}:{len(v)}' for k, v in sorted(merged.items()))} }}")
    train_loaders, val_loaders = build_loaders(merged, args.batch_size, seed=args.seed)

    model = build_model(args.arch, args.d_model, args.n_layers, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{args.arch} channel LWM: {n_params:,} params on {device}")

    steps = sum(len(l) for l in train_loaders.values())
    total_steps = max(1, steps * args.epochs)
    warmup_steps = max(1, steps * args.warmup_epochs)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    save_dir = os.path.join(_REPO_ROOT, 'outputs', 'pretrained_models', 'channel', f'{args.arch}_sampled')
    print(f"\nPretraining -> {save_dir}")
    train_lwm(model, train_loaders, val_loaders, optimizer, scheduler, args.epochs,
              device=device, save_dir=save_dir, log_file='training_log.csv',
              max_batches_per_epoch=args.max_batches_per_epoch)
    print(f"\nDone. Channel {args.arch} checkpoints in {save_dir}")


if __name__ == '__main__':
    main()
