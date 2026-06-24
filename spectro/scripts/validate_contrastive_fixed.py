"""Fast FIXED-DATA contrastive validation: do the losses behave, and does pretraining beat
random-init on one downstream task?

Pretrains the 3 per-protocol experts on a (subset of a) fixed corpus with MLM + supervised
contrastive on two CONFIGURABLE labels (mod/snr/mob), printing every loss component per epoch,
then compares downstream accuracy on ONE task (oracle routing, frozen backbone) of the trained
experts vs a random-init backbone.

Use it to iterate on the contrastive recipe until (a) all loss terms descend and (b) trained > random.

Usage::

    CUDA_VISIBLE_DEVICES=0 python spectro/scripts/validate_contrastive_fixed.py --arch transformer \
        --pretrain-dir spectro/outputs/spectro_deepmimo_big --subset 1500 \
        --contrast snr mob --task snr --epochs 10 --eval-subset 3000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (shared.*)
from spectro_data import PROTOCOLS, load_spectro_data, load_synthetic_data  # noqa: E402
from spectro_moe import SpectroMoE  # noqa: E402
from spectro_train_heads_config import ClassificationHead  # noqa: E402
from shared.finetune import finetune  # noqa: E402
from spectro_pretrain import pretrain_expert  # noqa: E402

_LABEL_FIELD = {'mod': 'modulation', 'snr': 'snr', 'mob': 'mobility'}


def _acc(logits, y):
    return float((logits.argmax(1) == y).float().mean())


def _subset_idx(data, idx, n, seed):
    """Take up to n indices from idx (a 1-D array), reproducibly."""
    if n <= 0 or n >= len(idx):
        return idx
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(idx, size=n, replace=False))


def downstream_one(moe, data, task, device, extract_batch, eval_subset, seed):
    """Frozen-backbone accuracy on a single task (oracle routing)."""
    # Optionally subsample the eval set (stratified-ish via the existing splits) for speed.
    tr = _subset_idx(data, data.train_idx, int(eval_subset * 0.8), seed)
    va = _subset_idx(data, data.val_idx, int(eval_subset * 0.1), seed)
    te = _subset_idx(data, data.test_idx, int(eval_subset * 0.1), seed)
    keep = np.concatenate([tr, va, te])
    specs = data.spectrograms[torch.as_tensor(keep)]
    proto = data.protocol[keep]
    y = torch.as_tensor(data.labels[task][keep])
    F = moe.extract_embeddings(specs, routing='oracle', protocol_idx=proto,
                               device=device, batch_size=extract_batch)
    # remap kept indices to 0..len(keep)-1 for the splits
    pos = {g: i for i, g in enumerate(keep)}
    tri = torch.as_tensor([pos[g] for g in tr]); vai = torch.as_tensor([pos[g] for g in va])
    tei = torch.as_tensor([pos[g] for g in te])
    _, _, s, _, _ = finetune(ClassificationHead(128, data.n_classes(task)),
                             F[tri], y[tri], F[vai], y[vai], F[tei], y[tei], score_fn=_acc,
                             epochs=100, lr=1e-3, batch_size=128, patience=20, device=device)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['transformer', 'mamba'], default='transformer')
    ap.add_argument('--pretrain-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           '..', 'outputs', 'spectro_deepmimo_big'),
                    help='synthetic corpus dir to pretrain on (manifest.json + shards)')
    ap.add_argument('--subset', type=int, default=1500, help='cap training spectrograms per expert')
    ap.add_argument('--contrast', nargs=2, default=['snr', 'mob'], choices=['mod', 'snr', 'mob'],
                    help='two labels to contrast on (magnitude: snr separable, mod weak)')
    ap.add_argument('--task', default='snr', choices=['modulation', 'snr', 'mobility'],
                    help='downstream task for the before/after check')
    ap.add_argument('--w-mlm', type=float, default=1.0)
    ap.add_argument('--w-a', type=float, default=50.0, help='weight for contrast[0]')
    ap.add_argument('--w-b', type=float, default=30.0, help='weight for contrast[1]')
    ap.add_argument('--proj-pool', choices=['mean', 'cls'], default='mean',
                    help="projection-head pooling: 'mean' (authors) or 'cls' (keeps local structure)")
    ap.add_argument('--temperature', type=float, default=0.2, help='SupCon temperature (paper: 0.2)')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--mask-percent', type=float, default=0.6)
    ap.add_argument('--n-layers', type=int, default=12)
    ap.add_argument('--warmup-frac', type=float, default=0.25)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--eval-subset', type=int, default=3000, help='downstream eval set size (speed)')
    ap.add_argument('--pretrain-on', choices=['corpus', 'demo'], default='corpus',
                    help="'corpus' = --pretrain-dir; 'demo' = pretrain on the demo TRAIN split and "
                         "eval on the demo TEST split (demo actually encodes mod/snr/mob, unlike the "
                         "synthetic corpus). Use 'demo' to test whether contrastive helps given real signal.")
    ap.add_argument('--eval-on', choices=['demo', 'self'], default='demo',
                    help="downstream eval set: 'demo' (real magnitude) or 'self' (held-out split of "
                         "the pretrain corpus — REQUIRED for complex, since demo is magnitude)")
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    batch = args.batch_size or (8 if args.arch == 'transformer' else 32)
    extract_batch = 16 if args.arch == 'transformer' else 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)  # reproducible random-init baseline across runs

    if args.pretrain_on == 'demo':
        pre = load_spectro_data(seed=args.seed)                    # demo HAS mod/snr/mob signal
        pretrain_mask = np.zeros(len(pre.protocol), dtype=bool); pretrain_mask[pre.train_idx] = True
        eval_data = pre                                            # eval on the demo TEST split
        args.eval_on = 'self'
    else:
        pre = load_synthetic_data(args.pretrain_dir, seed=args.seed)   # synthetic DeepMIMO corpus
        pretrain_mask = np.ones(len(pre.protocol), dtype=bool)
        eval_data = pre if args.eval_on == 'self' else load_spectro_data(seed=args.seed)
    # element_length: 16 for (N,1,H,W)/(N,H,W) magnitude, 32 for (N,2,H,W) complex.
    sp = pre.spectrograms
    n_chan = sp.shape[1] if sp.ndim == 4 else 1
    element_length = 16 * n_chan
    if n_chan == 2 and args.eval_on == 'demo':
        raise SystemExit("complex corpus (2ch) is incompatible with the magnitude demo; use --eval-on self")
    la, lb = args.contrast
    fa, fb = _LABEL_FIELD[la], _LABEL_FIELD[lb]
    print(f"arch={args.arch} element_length={element_length} contrast=({la},{lb}) w=({args.w_a},{args.w_b}) "
          f"task={args.task} eval_on={args.eval_on} epochs={args.epochs} batch={batch} subset={args.subset}/expert")

    print("\n[baseline] random-init experts -> downstream ...")
    moe0 = SpectroMoE(PROTOCOLS, d_model=128, arch=args.arch, n_layers=args.n_layers,
                      element_length=element_length)
    base = downstream_one(moe0, eval_data, args.task, device, extract_batch, args.eval_subset, args.seed)
    print(f"  random-init  {args.task} acc = {base:.4f}")
    del moe0; torch.cuda.empty_cache()

    print("\n[train] fixed-data contrastive per expert ...")
    moe1 = SpectroMoE(PROTOCOLS, d_model=128, arch=args.arch, n_layers=args.n_layers,
                      element_length=element_length)
    for p_idx, proto in enumerate(PROTOCOLS):
        sel = (pre.protocol == p_idx) & pretrain_mask
        specs = pre.spectrograms[torch.as_tensor(sel)]
        idx = np.arange(specs.shape[0])
        if args.subset and args.subset < len(idx):
            rng = np.random.RandomState(args.seed); idx = np.sort(rng.choice(idx, args.subset, replace=False))
        specs = specs[torch.as_tensor(idx)]
        a = pre.labels[fa][sel][idx]; b = pre.labels[fb][sel][idx]
        print(f"  -- {proto}: {specs.shape[0]} specs --")
        state, _, _ = pretrain_expert(
            specs, arch=args.arch, d_model=128, n_layers=args.n_layers, mask_percent=args.mask_percent,
            epochs=args.epochs, lr=5e-4, min_lr=1e-8, batch_size=batch, device=device, seed=args.seed,
            warmup_frac=args.warmup_frac, weight_decay=args.weight_decay, patience=10**9,
            contrastive=True, mod=a, mob=b, w_mlm=args.w_mlm, w_mod=args.w_a, w_mob=args.w_b,
            element_length=element_length, proj_pool=args.proj_pool, temperature=args.temperature,
            tag=proto)
        moe1.experts[proto].load_state_dict(state)
        del state; torch.cuda.empty_cache()

    print("\n[trained] contrastive experts -> downstream ...")
    aft = downstream_one(moe1, eval_data, args.task, device, extract_batch, args.eval_subset, args.seed)

    print(f"\n===== {args.arch}: {args.task} accuracy (oracle routing) =====")
    print(f"  random-init : {base:.4f}")
    print(f"  contrastive : {aft:.4f}")
    print(f"  delta       : {aft-base:+.4f}   -> {'BETTER' if aft > base else 'not better'}")


if __name__ == '__main__':
    main()
