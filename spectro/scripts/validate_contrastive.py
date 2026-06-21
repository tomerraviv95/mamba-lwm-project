"""Short fitting experiment: does contrastive streaming pretraining behave + beat random-init?

For one architecture, trains the 3 experts for a small number of steps with the contrastive
objective (MLM + SupCon mod/mobility, fewer-mods-per-batch so SupCon has positives), printing the
loss trajectory, then compares downstream demo-task accuracy (oracle routing, to isolate expert
quality) of the trained experts vs a random-init backbone. Use this to confirm readiness before the
long cluster runs.

Usage::

    CUDA_VISIBLE_DEVICES=1 python spectro/scripts/validate_contrastive.py --arch transformer --steps 1000 --batch 8
    CUDA_VISIBLE_DEVICES=1 python spectro/scripts/validate_contrastive.py --arch mamba --steps 1000 --batch 32
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datagen'))
from spectro_data import PROTOCOLS, TASKS, load_spectro_data  # noqa: E402
from spectro_moe import SpectroMoE  # noqa: E402
from spectro_train_heads_config import ClassificationHead  # noqa: E402
from shared.finetune import finetune  # noqa: E402
import stream as S  # noqa: E402
from spectro_pretrain_stream import pretrain_expert_stream  # noqa: E402


def _acc(logits, y):
    return float((logits.argmax(1) == y).float().mean())


def downstream(moe, data, device, extract_batch):
    F = moe.extract_embeddings(data.spectrograms, routing='oracle', protocol_idx=data.protocol,
                               device=device, batch_size=extract_batch)
    out = {}
    for t in TASKS:
        y = torch.as_tensor(data.labels[t])
        _, _, s, _, _ = finetune(ClassificationHead(128, data.n_classes(t)),
                                 F[data.train_idx], y[data.train_idx], F[data.val_idx], y[data.val_idx],
                                 F[data.test_idx], y[data.test_idx], score_fn=_acc, epochs=100, lr=1e-3,
                                 batch_size=128, patience=20, device=device)
        out[t] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['transformer', 'mamba'], required=True)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=None, help='default 8 (transformer) / 32 (mamba)')
    ap.add_argument('--n-layers', type=int, default=12)
    ap.add_argument('--mod-classes-per-batch', type=int, default=3)
    ap.add_argument('--pdp-pool', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       '..', 'outputs', 'pdp_pool'))
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    batch = args.batch or (8 if args.arch == 'transformer' else 32)
    extract_batch = 16 if args.arch == 'transformer' else 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pool = S.load_pool(args.pdp_pool)
    data = load_spectro_data(seed=args.seed)
    print(f"arch={args.arch} steps={args.steps} batch={batch} mod_classes/batch={args.mod_classes_per_batch}")

    print("\n[baseline] random-init experts -> downstream ...")
    moe0 = SpectroMoE(PROTOCOLS, d_model=128, arch=args.arch, n_layers=args.n_layers)
    base = downstream(moe0, data, device, extract_batch)
    del moe0; torch.cuda.empty_cache()

    print("\n[train] contrastive streaming per expert ...")
    moe1 = SpectroMoE(PROTOCOLS, d_model=128, arch=args.arch, n_layers=args.n_layers)
    for proto in PROTOCOLS:
        print(f"  -- {proto} --")
        m = pretrain_expert_stream(pool, proto, arch=args.arch, d_model=128, n_layers=args.n_layers,
                                   mask_percent=0.6, steps=args.steps, lr=1e-3, batch=batch,
                                   device=device, seed=args.seed, objective='contrastive',
                                   mod_classes_per_batch=args.mod_classes_per_batch,
                                   log_every=max(100, args.steps // 5), out_path=None)
        moe1.experts[proto].load_state_dict(m.state_dict())
        del m; torch.cuda.empty_cache()

    print("\n[trained] contrastive experts -> downstream ...")
    aft = downstream(moe1, data, device, extract_batch)

    print(f"\n===== {args.arch}: downstream accuracy (oracle routing) =====")
    print(f"{'task':12s} {'random':>8s} {'contrastive':>12s} {'delta':>8s}")
    for t in TASKS:
        print(f"{t:12s} {base[t]:8.3f} {aft[t]:12.3f} {aft[t]-base[t]:+8.3f}")
    wins = sum(aft[t] > base[t] for t in TASKS)
    print(f"\nverdict: contrastive beats random-init on {wins}/{len(TASKS)} tasks "
          f"-> {'READY for cluster' if wins >= 2 else 'NOT yet (tune weights/batch/steps)'}")


if __name__ == '__main__':
    main()
