"""Lightweight FINE-TUNE of a pretrained spectro MoE on one downstream task, vs the frozen probe.

Unlike spectro_train_heads.py (frozen embedding -> head), this fine-tunes the routed expert backbones
end-to-end together with a small classification head, on the downstream TRAIN split (oracle routing),
early-stops on VAL, and reports TEST accuracy. Tests whether fine-tuning lifts a task the frozen probe
caps (e.g. modulation ~0.50). Loads spectro_{arch}_p{patch}_{suffix}_weights.

Example (grid modulation, mamba, patch 4):
    CUDA_VISIBLE_DEVICES=0 python spectro/scripts/spectro_finetune.py --arch mamba --patch 4 \
        --weights-suffix grid --synth-dir spectro/outputs/spectro_eval_heldout_cities_grid --task modulation
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectro_data import PROTOCOLS, load_synthetic_data, load_spectro_data  # noqa: E402
from spectro_moe import SpectroMoE  # noqa: E402
from spectro_patchify import spectrogram_patchify, patch_geometry  # noqa: E402
from spectro_train_heads_config import ClassificationHead  # noqa: E402
from spectro_pretrain import weights_dir  # noqa: E402


def _load_moe(arch, patch, suffix, pool, device):
    wdir = weights_dir(arch, patch, suffix)
    router_ckpt = torch.load(os.path.join(wdir, 'router.pth'), map_location='cpu', weights_only=False)
    sample = torch.load(os.path.join(wdir, f'{PROTOCOLS[0]}_expert.pth'), map_location='cpu', weights_only=False)
    d_model, n_layers = sample.get('d_model', 128), sample.get('n_layers', 12)
    element_length = sample.get('element_length', patch * patch)
    max_len = sample.get('max_len', patch_geometry(patch)['max_len'])
    in_ch = max(1, element_length // (patch * patch))
    moe = SpectroMoE(PROTOCOLS, d_model=d_model, arch=arch, n_layers=n_layers, pool=pool, patch=patch,
                     element_length=element_length, max_len=max_len, in_channels=in_ch)
    for p in PROTOCOLS:
        moe.load_expert(p, torch.load(os.path.join(wdir, f'{p}_expert.pth'),
                                      map_location='cpu', weights_only=False)['state_dict'])
    moe.router.load_state_dict(router_ckpt['state_dict'])
    return moe.to(device), d_model


def _input_ids(specs, patch, device):
    """(N,[C,]128,128) -> (N, n_patches+1, element_length) tokens with CLS, on device."""
    P = spectrogram_patchify(specs, patch=patch, normalize=True)          # (N, n_patches, E)
    cls = np.full((P.shape[0], 1, P.shape[2]), 0.2, dtype=np.float32)
    return torch.tensor(np.concatenate([cls, P], axis=1), dtype=torch.float32, device=device)


def _embed_routed(moe, ids, proto, pool, device):
    """Grad-enabled routed embedding: run each protocol's expert on its samples -> (B, pooled_dim)."""
    out = None
    for e_idx, p in enumerate(moe.protocols):
        mask = torch.as_tensor(proto == e_idx, device=device)
        if not mask.any():
            continue
        emb = moe.experts[p].embed(ids[mask], pool=pool)                  # (m, pooled_dim) WITH grad
        if out is None:
            out = torch.zeros(ids.shape[0], emb.shape[1], device=device, dtype=emb.dtype)
        out[mask] = emb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['mamba', 'transformer'], default='mamba')
    ap.add_argument('--patch', type=int, default=4, choices=[4, 6, 8])
    ap.add_argument('--weights-suffix', default='grid')
    ap.add_argument('--synth-dir', default='spectro/outputs/spectro_eval_heldout_cities_grid',
                    help="in-domain eval corpus (train/val/test split); omit to use the demo set")
    ap.add_argument('--task', default='modulation', choices=['modulation', 'snr', 'mobility'])
    ap.add_argument('--pool', choices=['mean', 'cls', 'meanstd_t'], default='meanstd_t')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--lr-backbone', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data = load_synthetic_data(args.synth_dir, seed=args.seed) if args.synth_dir else load_spectro_data(seed=args.seed)
    moe, d_model = _load_moe(args.arch, args.patch, args.weights_suffix, args.pool, device)
    y = torch.as_tensor(data.labels[args.task], dtype=torch.long)
    proto = data.protocol
    ids_all = _input_ids(data.spectrograms, args.patch, device)           # precompute tokens once
    pooled_dim = d_model * (2 if args.pool == 'meanstd_t' else 1)
    head = ClassificationHead(pooled_dim, data.n_classes(args.task)).to(device)

    tr, va, te = data.train_idx, data.val_idx, data.test_idx
    print(f"finetune {args.arch} p{args.patch} [{args.weights_suffix}] task={args.task} pool={args.pool} "
          f"train/val/test={len(tr)}/{len(va)}/{len(te)} classes={data.n_classes(args.task)}")

    opt = torch.optim.AdamW([
        {'params': [p for e in moe.experts.values() for p in e.parameters()], 'lr': args.lr_backbone},
        {'params': head.parameters(), 'lr': args.lr_head}], weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(idx):
        moe.eval(); head.eval(); correct = 0
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            logits = head(_embed_routed(moe, ids_all[torch.as_tensor(b)], proto[b], args.pool, device))
            correct += (logits.argmax(1).cpu() == y[torch.as_tensor(b)]).sum().item()
        return correct / len(idx)

    best_va, best, ctr = -1.0, None, 0
    rng = np.random.RandomState(args.seed)
    for ep in range(args.epochs):
        moe.train(); head.train(); order = rng.permutation(tr)
        for s in range(0, len(order), args.batch_size):
            b = order[s:s + args.batch_size]
            opt.zero_grad()
            logits = head(_embed_routed(moe, ids_all[torch.as_tensor(b)], proto[b], args.pool, device))
            loss = crit(logits, y[torch.as_tensor(b)].to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g['params']], 1.0); opt.step()
        va_acc = evaluate(va)
        print(f"  epoch {ep+1}/{args.epochs}  val={va_acc:.4f}", flush=True)
        if va_acc > best_va:
            best_va, ctr = va_acc, 0
            best = ({k: v.detach().cpu().clone() for k, v in moe.state_dict().items()},
                    {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        else:
            ctr += 1
            if ctr >= args.patience:
                print(f"  early stop @ {ep+1}"); break
    if best is not None:
        moe.load_state_dict(best[0]); head.load_state_dict(best[1])
    test_acc = evaluate(te)
    print(f"\n===== FINETUNE {args.arch} p{args.patch} {args.task}: TEST acc = {test_acc:.4f} "
          f"(best val {best_va:.4f}) =====")


if __name__ == '__main__':
    main()
