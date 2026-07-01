"""Real STEP-BASED spectro MoE pretraining with periodic demo validation.

Trains each per-protocol expert for a fixed number of optimizer STEPS (cycling the corpus) with the
LWM-Spectro PAPER recipe (MLM + supervised contrastive on mod & mobility; lambda_recon=1.0,
lambda_cont=0.3; tau=0.2; mask 70%; AdamW wd=0.05; lr 5e-4 with linear warmup -> cosine). Every
``--eval-every`` steps it probes the expert on its protocol's slice of the REAL demo set for one
downstream task (a quick logistic-regression head on the frozen mean-pooled embedding) and logs the
accuracy alongside the train losses (stdout + CSV + optional W&B). Checkpoints -> spectro_{arch}_weights/.

Usage::

    CUDA_VISIBLE_DEVICES=0 python spectro/scripts/spectro_pretrain_real.py --arch transformer \
        --pretrain-dir spectro/outputs/spectro_deepmimo_150k --steps 30000 --eval-every 2000 \
        --eval-task modulation --wandb
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datagen'))
from spectro_backbones import build_expert  # noqa: E402
from spectro_data import PROTOCOLS, load_spectro_data, load_synthetic_data  # noqa: E402
from spectro_moe import RouterNet, _normalize_per_sample  # noqa: E402
from spectro_patchify import spectrogram_patchify, build_masked_tensors, patch_geometry  # noqa: E402
from contrastive import ProjectionHead, supervised_contrastive_loss  # noqa: E402
from spectro_pretrain import weights_dir, train_router  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


@torch.no_grad()
def _embed(expert, specs, pool, device, patch, batch=16):
    """Mean/CLS-pooled embeddings for a stack of spectrograms via one expert."""
    expert.eval()
    P = spectrogram_patchify(specs, patch=patch, normalize=True)         # (N,n_patches,E)
    cls = np.full((P.shape[0], 1, P.shape[2]), 0.2, dtype=np.float32)
    ids = torch.tensor(np.concatenate([cls, P], axis=1), dtype=torch.float32)
    out = []
    for s in range(0, ids.shape[0], batch):
        out.append(expert.embed(ids[s:s + batch].to(device), pool=pool).float().cpu())
    expert.train()
    return torch.cat(out).numpy()


def demo_probe(expert, demo, p_idx, task, pool, device, patch, cap=1500, seed=0):
    """Frozen-embedding logistic-regression accuracy on this protocol's demo slice (one task)."""
    sel = np.where(demo.protocol == p_idx)[0]
    rng = np.random.RandomState(seed)
    if len(sel) > cap:
        sel = np.sort(rng.choice(sel, cap, replace=False))
    feats = _embed(expert, demo.spectrograms[torch.as_tensor(sel)], pool, device, patch)
    y = demo.labels[task][sel].astype(int)
    n = len(sel); ntr = int(0.8 * n)
    perm = rng.permutation(n)
    tr, te = perm[:ntr], perm[ntr:]
    if len(set(y[tr].tolist())) < 2:
        return float('nan')
    clf = LogisticRegression(max_iter=300).fit(feats[tr], y[tr])
    return float((clf.predict(feats[te]) == y[te]).mean())


def pretrain_expert_steps(specs, mod, mob, *, arch, proto, demo, eval_task, device, args, wandb_run):
    """Step-based MLM+SupCon pretraining of one expert with periodic demo probing. Returns state, history."""
    ids, toks, pos = build_masked_tensors(specs, mask_percent=args.mask_percent, seed=args.seed,
                                          patch=args.patch)
    mod_t = torch.as_tensor(np.asarray(mod), dtype=torch.long)
    mob_t = torch.as_tensor(np.asarray(mob), dtype=torch.long)
    loader = DataLoader(TensorDataset(ids, toks, pos, mod_t, mob_t),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
    p_idx = PROTOCOLS.index(proto)

    model = build_expert(arch, d_model=args.d_model, n_layers=args.n_layers,
                         element_length=args.element_length, max_len=args.max_len).to(device)
    proj_mod = ProjectionHead(args.d_model, 128, pool=args.proj_pool).to(device)
    proj_mob = ProjectionHead(args.d_model, 128, pool=args.proj_pool).to(device)
    params = list(model.parameters()) + list(proj_mod.parameters()) + list(proj_mob.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.warmup_frac * args.steps))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps - warmup), eta_min=args.min_lr),
    ], milestones=[warmup])
    mse = nn.MSELoss(reduction='mean')

    history, step = [], 0
    accum = max(1, args.accum_steps)            # >1: micro-batch grad accumulation (effective batch = batch*accum)
    base_acc = (demo_probe(model, demo, p_idx, eval_task, args.proj_pool, device, args.patch, seed=args.seed)
                if args.probe_demo else float('nan'))
    print(f"  [{proto}] step 0 (random-init) demo {eval_task} acc = {base_acc:.4f}", flush=True)
    loader_iter = itertools.cycle(loader)
    while step < args.steps:                    # `step` counts OPTIMIZER steps (apples-to-apples w/ batch=accum*micro)
        model.train(); proj_mod.train(); proj_mob.train()
        opt.zero_grad()
        mlm_v = sc_mod_v = sc_mob_v = 0.0
        for _ in range(accum):                  # accumulate grads over `accum` micro-batches, then one opt step
            b_ids, b_toks, b_pos, b_mod, b_mob = next(loader_iter)
            b_ids, b_toks, b_pos = b_ids.to(device), b_toks.to(device), b_pos.to(device)
            b_mod, b_mob = b_mod.to(device), b_mob.to(device)
            logits, output = model(b_ids, b_pos)
            mlm = mse(logits, b_toks)
            sc_mod = supervised_contrastive_loss(proj_mod(output), b_mod, temperature=args.temperature,
                                                 base_temperature=args.temperature)
            sc_mob = supervised_contrastive_loss(proj_mob(output), b_mob, temperature=args.temperature,
                                                 base_temperature=args.temperature)
            loss = (args.w_mlm * mlm + args.w_cont * sc_mod + args.w_cont * sc_mob) / accum
            loss.backward()
            mlm_v += mlm.item() / accum; sc_mod_v += sc_mod.item() / accum; sc_mob_v += sc_mob.item() / accum
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step(); sched.step()
        step += 1

        if step % args.eval_every == 0 or step == args.steps:
            acc = (demo_probe(model, demo, p_idx, eval_task, args.proj_pool, device, args.patch, seed=args.seed)
                   if args.probe_demo else float('nan'))
            row = {'step': step, 'mlm': mlm_v, 'sc_mod': sc_mod_v, 'sc_mob': sc_mob_v,
                   'demo_acc': acc, 'lr': opt.param_groups[0]['lr']}
            history.append(row)
            print(f"  [{proto}] step {step}/{args.steps}  mlm={mlm_v:.4f} sc_mod={sc_mod_v:.4f} "
                  f"sc_mob={sc_mob_v:.4f}  demo_{eval_task}={acc:.4f}", flush=True)
            if wandb_run is not None:
                wandb_run.log({f'{proto}/{k}': v for k, v in row.items() if k != 'step'} | {f'{proto}/step': step})
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, history, base_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['mamba', 'transformer'], default='transformer')
    ap.add_argument('--pretrain-dir', default=os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'spectro_deepmimo_150k'))
    ap.add_argument('--steps', type=int, default=30000, help='optimizer steps per expert')
    ap.add_argument('--eval-every', type=int, default=2000, help='demo-probe every N steps')
    ap.add_argument('--eval-task', default='modulation', choices=['modulation', 'snr', 'mobility'])
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--weights-suffix', default='',
                    help="suffix for the weights dir -> spectro_{arch}_p{patch}_{suffix}_weights (e.g. "
                         "'grid' so grid-representation checkpoints don't clobber the STFT ones).")
    ap.add_argument('--accum-steps', type=int, default=1,
                    help='gradient accumulation: micro-batches per optimizer step. Effective batch = '
                         'batch-size * accum-steps. Use e.g. --batch-size 8 --accum-steps 4 (=eff 32) so a '
                         'memory-heavy transformer matches mamba\'s batch 32 without OOM. Optimizer-step '
                         'count (--steps) is unchanged, so runs stay apples-to-apples.')
    ap.add_argument('--patch', type=int, default=4, choices=[4, 6, 8],
                    help='patch side: 4->1025 tokens/elem16, 6->442/36, 8->257/64 (baked into the weights dir name)')
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-layers', type=int, default=12)
    ap.add_argument('--router-epochs', type=int, default=8)
    # paper Table I
    ap.add_argument('--mask-percent', type=float, default=0.7)
    ap.add_argument('--w-mlm', type=float, default=1.0)
    ap.add_argument('--w-cont', type=float, default=0.3)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--min-lr', type=float, default=1e-8)
    ap.add_argument('--warmup-frac', type=float, default=0.1)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--proj-pool', choices=['mean', 'cls'], default='mean')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--wandb-project', default='lwm-spectro')
    ap.add_argument('--run-name', default=None)
    args = ap.parse_args()
    args.batch_size = args.batch_size or 32   # SAME default for both arches (apples-to-apples)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    pre = load_synthetic_data(args.pretrain_dir, seed=args.seed)
    demo = load_spectro_data(seed=args.seed)
    sp = pre.spectrograms
    channels = sp.shape[1] if sp.ndim == 4 else 1
    geom = patch_geometry(args.patch, channels=channels)
    args.element_length, args.max_len = geom['element_length'], geom['max_len']
    # the in-training demo probe embeds the 1-channel magnitude demo set; skip it when the corpus is
    # multi-channel (grid_stft/complex) — the shapes don't match and it's cross-representation anyway.
    args.probe_demo = (channels == 1)
    if not args.probe_demo:
        print(f"[note] corpus has {channels} channels -> skipping the 1-channel demo probe during training")
    out_dir = weights_dir(args.arch, args.patch, args.weights_suffix); os.makedirs(out_dir, exist_ok=True)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project,
                                   name=args.run_name or f"real-{args.arch}-p{args.patch}{('-'+args.weights_suffix) if args.weights_suffix else ''}",
                                   config=vars(args))
        except Exception as e:
            print(f"WARNING: wandb init failed ({e}); continuing without it.")

    print(f"REAL pretrain {args.arch} patch={args.patch} (elem={args.element_length}, max_len={args.max_len}): "
          f"steps={args.steps}/expert eval_every={args.eval_every} "
          f"task={args.eval_task} batch={args.batch_size} corpus={pre.spectrograms.shape[0]} "
          f"(w_mlm={args.w_mlm}, w_cont={args.w_cont}, tau={args.temperature}, mask={args.mask_percent})")
    for proto in PROTOCOLS:
        p_idx = PROTOCOLS.index(proto)
        sel = pre.protocol == p_idx
        specs = pre.spectrograms[torch.as_tensor(sel)]
        mod, mob = pre.labels['modulation'][sel], pre.labels['mobility'][sel]
        print(f"\n[Expert {proto}] {specs.shape[0]} corpus spectrograms")
        state, history, base = pretrain_expert_steps(specs, mod, mob, arch=args.arch, proto=proto,
                                                     demo=demo, eval_task=args.eval_task, device=device,
                                                     args=args, wandb_run=wandb_run)
        torch.save({'state_dict': state, 'arch': args.arch, 'd_model': args.d_model,
                    'n_layers': args.n_layers, 'patch': args.patch,
                    'element_length': args.element_length, 'max_len': args.max_len},
                   os.path.join(out_dir, f"{proto}_expert.pth"))
        with open(os.path.join(out_dir, f"{proto}_steps.csv"), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
        final = history[-1]['demo_acc'] if history else float('nan')
        print(f"[Expert {proto}] saved. demo {args.eval_task}: {base:.4f} (init) -> {final:.4f} (final)")

    print("\n[Router] training protocol router")
    ridx = np.random.RandomState(args.seed).permutation(len(pre.protocol))[:40000]  # cap for speed
    r_state, r_acc = train_router(pre.spectrograms[torch.as_tensor(ridx)], pre.protocol[ridx],
                                  epochs=args.router_epochs, lr=1e-3,
                                  batch_size=max(64, args.batch_size), device=device, seed=args.seed)
    torch.save({'state_dict': r_state, 'val_acc': r_acc, 'protocols': PROTOCOLS},
               os.path.join(out_dir, 'router.pth'))
    print(f"[Router] saved (val_acc={r_acc:.3f})")
    print(f"\nDone. {args.arch} MoE weights + {{proto}}_steps.csv in {out_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
