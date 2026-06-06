"""Pretrain the Mamba MoE on the LWM-Spectro demo spectrograms.

Two stages, mirroring LWM-Spectro:
1. Per-protocol expert pretraining via masked-spectrogram modeling (MSE on masked patches),
   each expert trained only on its protocol subset of the (train split of the) demo data.
2. Router training: a CNN that classifies the protocol from the raw spectrogram (the experts
   stay frozen; routing is learned as protocol prediction, matching the HF router objective).

Checkpoints are saved to ``spectro/outputs/pretrained_models/spectro_mamba_weights/``.

NOTE: only the ~10.5k demo spectrograms are available (no full corpus), so each expert sees
~2.4k training spectrograms. This is a small-data foundation model; see spectro/README.md.

Usage::

    python spectro/scripts/spectro_pretrain.py                 # full run
    python spectro/scripts/spectro_pretrain.py --smoke         # tiny/fast sanity run
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectro_data import PROTOCOLS, load_spectro_data, load_synthetic_data  # noqa: E402
from spectro_mamba_model import lwm_mamba_spectro  # noqa: E402
from spectro_moe import RouterNet, _normalize_per_sample  # noqa: E402
from spectro_patchify import build_masked_tensors  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_WEIGHTS_DIR = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'pretrained_models', 'spectro_mamba_weights')


def pretrain_expert(specs: torch.Tensor, *, d_model, n_layers, mask_percent, epochs, lr,
                    batch_size, device, seed, val_frac=0.1, patience=4):
    """Masked-spectrogram-modeling pretraining of one Mamba expert. Returns best state_dict."""
    ids, toks, pos = build_masked_tensors(specs, mask_percent=mask_percent, seed=seed)
    n = ids.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    tr_loader = DataLoader(TensorDataset(ids[tr_i], toks[tr_i], pos[tr_i]),
                           batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(ids[val_i], toks[val_i], pos[val_i]),
                            batch_size=batch_size, shuffle=False)

    model = lwm_mamba_spectro(d_model=d_model, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    criterion = nn.MSELoss(reduction='sum')

    best_val, best_state, ctr = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for b_ids, b_toks, b_pos in tr_loader:
            b_ids, b_toks, b_pos = b_ids.to(device), b_toks.to(device), b_pos.to(device)
            opt.zero_grad()
            logits = model(b_ids, b_pos)[0]
            loss = criterion(b_toks, logits)
            loss.backward()
            opt.step()
            tr_loss += loss.item(); tr_n += b_ids.shape[0]
        sched.step()

        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for b_ids, b_toks, b_pos in val_loader:
                b_ids, b_toks, b_pos = b_ids.to(device), b_toks.to(device), b_pos.to(device)
                logits = model(b_ids, b_pos)[0]
                v_loss += criterion(b_toks, logits).item(); v_n += b_ids.shape[0]
        v_loss /= max(v_n, 1)
        print(f"    epoch {ep+1}/{epochs}  train_mse={tr_loss/max(tr_n,1):.4f}  val_mse={v_loss:.4f}")
        if v_loss < best_val - 1e-6:
            best_val, ctr = v_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            ctr += 1
            if ctr >= patience:
                print(f"    early stop @ epoch {ep+1} (best val_mse={best_val:.4f})")
                break
    return best_state, best_val


def train_router(specs: torch.Tensor, protocol: np.ndarray, *, epochs, lr, batch_size,
                 device, seed, val_frac=0.1):
    """Train the CNN router to classify protocol from the raw spectrogram."""
    n = specs.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    y = torch.as_tensor(protocol, dtype=torch.long)

    tr_loader = DataLoader(TensorDataset(specs[tr_i], y[tr_i]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(specs[val_i], y[val_i]), batch_size=batch_size, shuffle=False)

    router = RouterNet(num_experts=len(PROTOCOLS)).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        router.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(router(_normalize_per_sample(xb)), yb)
            loss.backward(); opt.step()
        router.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = router(_normalize_per_sample(xb)).argmax(1).cpu()
                correct += (pred == yb).sum().item(); total += yb.size(0)
        acc = correct / max(total, 1)
        print(f"    router epoch {ep+1}/{epochs}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
    return best_state, best_acc


def _build_pool(args):
    """Return (spectrograms, protocol) for pretraining, per --data.

    - demo:      demo_data train split only (val/test stay unseen for the downstream sweep).
    - synthetic: the full generated synthetic corpus.
    - mixed:     demo train split + synthetic corpus concatenated.
    """
    parts_specs, parts_proto = [], []
    if args.data in ('demo', 'mixed'):
        data = load_spectro_data(seed=args.seed)
        tr = data.train_idx
        parts_specs.append(data.spectrograms[torch.as_tensor(tr)])
        parts_proto.append(data.protocol[tr])
    if args.data in ('synthetic', 'mixed'):
        syn = load_synthetic_data(args.synthetic_dir, seed=args.seed)
        parts_specs.append(syn.spectrograms)
        parts_proto.append(syn.protocol)
    specs = torch.cat(parts_specs, dim=0)
    proto = np.concatenate(parts_proto, axis=0)
    return specs, proto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-layers', type=int, default=12)
    ap.add_argument('--mask-percent', type=float, default=0.6)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--router-epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true', help='tiny fast run for sanity checks')
    ap.add_argument('--data', choices=['demo', 'synthetic', 'mixed'], default='demo',
                    help="pretraining corpus: demo_data, the synthetic corpus, or both.")
    ap.add_argument('--synthetic-dir', default=os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'synthetic'),
                    help="directory of a generated synthetic corpus (manifest.json + shards).")
    args = ap.parse_args()

    if args.smoke:
        args.n_layers, args.epochs, args.router_epochs = 2, 2, 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(_WEIGHTS_DIR, exist_ok=True)

    # Build the pretraining pool (spectrograms + protocol) per the chosen data source.
    pool_specs, pool_proto = _build_pool(args)

    print(f"Pretraining Mamba experts on device={device} (data={args.data}, "
          f"n_layers={args.n_layers}, mask={args.mask_percent}, epochs={args.epochs})")
    for p_idx, proto in enumerate(PROTOCOLS):
        sel = pool_proto == p_idx
        specs = pool_specs[torch.as_tensor(sel)]
        print(f"\n[Expert {proto}] {specs.shape[0]} training spectrograms")
        if args.smoke:
            specs = specs[:64]
        state, val = pretrain_expert(
            specs, d_model=args.d_model, n_layers=args.n_layers,
            mask_percent=args.mask_percent, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=device, seed=args.seed)
        path = os.path.join(_WEIGHTS_DIR, f"{proto}_expert.pth")
        torch.save({'state_dict': state, 'val_mse': val, 'd_model': args.d_model,
                    'n_layers': args.n_layers}, path)
        print(f"[Expert {proto}] saved -> {path} (val_mse={val:.4f})")

    print("\n[Router] training protocol router")
    specs = pool_specs
    proto = pool_proto
    if args.smoke:
        specs, proto = specs[:192], proto[:192]
    r_state, r_acc = train_router(specs, proto, epochs=args.router_epochs, lr=1e-3,
                                  batch_size=args.batch_size, device=device, seed=args.seed)
    r_path = os.path.join(_WEIGHTS_DIR, 'router.pth')
    torch.save({'state_dict': r_state, 'val_acc': r_acc, 'protocols': PROTOCOLS}, r_path)
    print(f"[Router] saved -> {r_path} (val_acc={r_acc:.3f})")
    print("\nDone. Pretrained Mamba MoE weights in", _WEIGHTS_DIR)


if __name__ == '__main__':
    main()
