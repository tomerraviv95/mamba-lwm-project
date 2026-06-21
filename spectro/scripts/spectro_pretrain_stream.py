"""Streaming MoE pretraining: ~10M DeepMIMO-channel spectrograms generated on the fly.

Instead of a stored dataset, this draws PDPs from the compact pool (build_pdp_pool.py) and
synthesizes masked spectrograms per batch (spectro/datagen/stream.py) to pretrain the per-protocol
experts + router. Realizes the authors' ~10M scale without materializing ~327 GB.

Usage::

    CUDA_VISIBLE_DEVICES=1 python spectro/scripts/spectro_pretrain_stream.py \
        --arch mamba --pdp-pool spectro/outputs/pdp_pool --target-samples 10000000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datagen'))
from spectro_backbones import build_expert  # noqa: E402
from spectro_data import PROTOCOLS  # noqa: E402
from spectro_moe import RouterNet, _normalize_per_sample  # noqa: E402
import stream as S  # noqa: E402
from contrastive import ProjectionHead, supervised_contrastive_loss  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


def weights_dir(arch):
    return os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'pretrained_models', f'spectro_{arch}_stream_weights')


def pretrain_expert_stream(pool, tech, *, arch, d_model, n_layers, mask_percent, steps, lr,
                           batch, device, seed, grad_clip=1.0, log_every=200, ckpt_every=2000,
                           out_path=None, objective='contrastive',
                           w_mlm=1.0, w_a=1.0, w_b=1.0, contrast=('snr', 'mob'),
                           mod_classes_per_batch=3):
    """Stream-pretrain one expert. objective='contrastive' = MLM(mean) + SupCon(mod)+SupCon(mobility)
    (the authors' recipe; prevents the collapse seen with MLM-only). 'mlm' = masked-MSE only."""
    model = build_expert(arch, d_model=d_model, n_layers=n_layers).to(device)
    params = list(model.parameters())
    mod_proj = mob_proj = None
    if objective == 'contrastive':
        mod_proj = ProjectionHead(d_model).to(device)
        mob_proj = ProjectionHead(d_model).to(device)
        params += list(mod_proj.parameters()) + list(mob_proj.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    mse_mean = nn.MSELoss(reduction='mean')
    mse_sum = nn.MSELoss(reduction='sum')
    rng = np.random.RandomState(seed)
    model.train()
    ca, cb = contrast   # which two labels to contrast on (e.g. 'snr','mob')
    agg = {'mlm': 0.0, 'a': 0.0, 'b': 0.0, 'n': 0}
    for step in range(steps):
        opt.zero_grad()
        if objective == 'contrastive':
            ids, toks, pos, labels = S.gen_contrastive_batch(
                pool, tech, batch, mask_percent, rng, device=device,
                mod_classes_per_batch=mod_classes_per_batch)
            logits, enc = model(ids, pos)          # expert returns (masked_logits, encoder_out)
            l_mlm = mse_mean(toks, logits)
            l_a = supervised_contrastive_loss(mod_proj(enc), labels[ca])
            l_b = supervised_contrastive_loss(mob_proj(enc), labels[cb])
            loss = w_mlm * l_mlm + w_a * l_a + w_b * l_b
            agg['mlm'] += l_mlm.item(); agg['a'] += l_a.item(); agg['b'] += l_b.item()
        else:
            ids, toks, pos = S.gen_masked_batch(pool, tech, batch, mask_percent, rng, device=device)
            loss = mse_sum(toks, model(ids, pos)[0]); agg['mlm'] += loss.item()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step(); sched.step()
        agg['n'] += 1
        if (step + 1) % log_every == 0:
            n = max(agg['n'], 1)
            print(f"    [{tech}] step {step+1}/{steps}  mlm={agg['mlm']/n:.4f} "
                  f"supcon_{ca}={agg['a']/n:.4f} supcon_{cb}={agg['b']/n:.4f}", flush=True)
            agg = {'mlm': 0.0, 'a': 0.0, 'b': 0.0, 'n': 0}
        if out_path and (step + 1) % ckpt_every == 0:
            _save(model, out_path, arch, d_model, n_layers)
    if out_path:
        _save(model, out_path, arch, d_model, n_layers)
    return model


def _save(model, path, arch, d_model, n_layers):
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'arch': arch, 'd_model': d_model, 'n_layers': n_layers}, path)


def train_router_stream(pool, *, steps, lr, batch, device, seed, log_every=200):
    router = RouterNet(num_experts=len(PROTOCOLS)).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed + 99)
    router.train()
    correct = total = 0
    for step in range(steps):
        specs, labels = S.gen_router_batch(pool, batch, rng, device=device)
        specs, labels = specs.to(device), labels.to(device)
        opt.zero_grad()
        logits = router(_normalize_per_sample(specs))
        loss = crit(logits, labels)
        loss.backward(); opt.step()
        correct += (logits.argmax(1) == labels).sum().item(); total += batch
        if (step + 1) % log_every == 0:
            print(f"    [router] step {step+1}/{steps}  acc={correct/max(total,1):.3f}", flush=True)
            correct = total = 0
    return router


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['mamba', 'transformer'], default='mamba')
    ap.add_argument('--pdp-pool', default=os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'pdp_pool'))
    ap.add_argument('--target-samples', type=int, default=10_000_000,
                    help='~total spectrograms streamed across the 3 experts (router adds ~10%).')
    ap.add_argument('--steps-per-expert', type=int, default=None, help='override target-samples.')
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-layers', type=int, default=12)
    ap.add_argument('--mask-percent', type=float, default=0.6)
    ap.add_argument('--objective', choices=['contrastive', 'mlm'], default='contrastive',
                    help="contrastive = MLM + SupCon(mod)+SupCon(mobility) (authors' recipe); mlm = masked-MSE only.")
    ap.add_argument('--w-mlm', type=float, default=1.0)
    ap.add_argument('--w-a', type=float, default=1.0)
    ap.add_argument('--w-b', type=float, default=1.0)
    ap.add_argument('--contrast', nargs=2, default=['snr', 'mob'], choices=['mod', 'snr', 'mob'],
                    help='two labels to contrast on. magnitude spectrograms -> snr/mob (mod barely separable).')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=32,
                    help='keep <=32 on a 24GB GPU; the time-varying channel tensor is the limiter.')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    if args.smoke:
        args.n_layers, args.steps_per_expert, args.batch = 2, 3, 8
    steps = args.steps_per_expert or max(1, args.target_samples // (len(PROTOCOLS) * args.batch))
    router_steps = max(1, steps // 4) if not args.smoke else 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = weights_dir(args.arch)
    os.makedirs(out, exist_ok=True)
    pool = S.load_pool(args.pdp_pool)
    print(f"Streaming pretrain: arch={args.arch}, pool={pool['delay'].shape[0]} users, "
          f"{steps} steps/expert × {args.batch} batch ≈ {steps*args.batch*len(PROTOCOLS)/1e6:.1f}M spectrograms")

    for proto in PROTOCOLS:
        print(f"\n[Expert {proto}] streaming {steps} steps ...")
        pretrain_expert_stream(pool, proto, arch=args.arch, d_model=args.d_model,
                               n_layers=args.n_layers, mask_percent=args.mask_percent, steps=steps,
                               lr=args.lr, batch=args.batch, device=device, seed=args.seed,
                               grad_clip=args.grad_clip, out_path=os.path.join(out, f'{proto}_expert.pth'),
                               objective=args.objective, w_mlm=args.w_mlm, w_a=args.w_a, w_b=args.w_b,
                               contrast=tuple(args.contrast))
        print(f"[Expert {proto}] saved -> {out}/{proto}_expert.pth")

    print(f"\n[Router] streaming {router_steps} steps ...")
    router = train_router_stream(pool, steps=router_steps, lr=1e-3, batch=args.batch,
                                 device=device, seed=args.seed)
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in router.state_dict().items()},
                'protocols': PROTOCOLS}, os.path.join(out, 'router.pth'))
    print(f"[Router] saved -> {out}/router.pth\n\nDone. Streaming MoE weights in {out}")


if __name__ == '__main__':
    main()
