"""Validate that synthetic protocols (LTE/WiFi/5G) are separable.

Trains the same CNN ``RouterNet`` used by the Mamba MoE on a synthetic corpus with a held-out
split and reports protocol-classification accuracy + a confusion matrix. High accuracy confirms
the per-protocol numerology makes the techs distinguishable (the router's whole job).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'spectro', 'scripts'))
from spectro_data import PROTOCOLS, load_synthetic_data  # noqa: E402
from spectro_moe import RouterNet, _normalize_per_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'synthetic'))
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_synthetic_data(args.dir, seed=args.seed)
    specs, proto = data.spectrograms, torch.as_tensor(data.protocol, dtype=torch.long)
    tr, te = data.train_idx, data.test_idx
    print(f"Loaded {len(specs)} synthetic samples; train={len(tr)} test={len(te)}")

    router = RouterNet(num_experts=len(PROTOCOLS)).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(specs[tr], proto[tr]), batch_size=args.batch_size, shuffle=True)
    for ep in range(args.epochs):
        router.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(router(_normalize_per_sample(xb)), yb).backward()
            opt.step()

    router.eval()
    with torch.no_grad():
        preds = router(_normalize_per_sample(specs[te].to(device))).argmax(1).cpu()
    y = proto[te]
    acc = float((preds == y).float().mean())
    conf = np.zeros((len(PROTOCOLS), len(PROTOCOLS)), dtype=int)
    for t, p in zip(y.tolist(), preds.tolist()):
        conf[t, p] += 1
    print(f"\nProtocol separability accuracy (test): {acc:.3f}  (chance = {1/len(PROTOCOLS):.3f})")
    print("confusion matrix (rows=true, cols=pred), order=" + ",".join(PROTOCOLS))
    for i, row in enumerate(conf):
        print(f"  {PROTOCOLS[i]:4s} {row}")


if __name__ == '__main__':
    main()
