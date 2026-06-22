"""Pretrain a spectrogram MoE (Mamba or Transformer experts) on LWM-Spectro spectrograms.

Two stages, mirroring LWM-Spectro:
1. Per-protocol expert pretraining. Two objectives:
     - MLM only  (masked-spectrogram modeling, MSE on masked patches), or
     - ``--contrastive``: MLM + supervised contrastive on modulation & mobility, the authors'
       flagship recipe (``train_lwm_spectro_contrastive.py``): loss = 1*MLM + 50*SupCon(mod)
       + 30*SupCon(mob), two SimCLR-style projection heads on the encoder output, AdamW
       (wd=0.05), 5-epoch warmup -> cosine to 1e-5.
   Each expert trains only on its protocol subset of the chosen corpus.
2. Router training: a CNN that classifies the protocol from the raw spectrogram (experts frozen).

We keep magnitude spectrograms (element_length=16, 4x4x1) rather than the authors' complex
(element_length=32) representation. To preserve the authors' MLM-vs-contrastive *balance* at this
smaller element scale, the contrastive path measures MLM as a per-element MEAN (scale-comparable to
SupCon ~O(1)) so the 1/50/30 weights have their intended (contrastive-dominant) effect. The MLM-only
path keeps the original per-sample-sum scale.

Checkpoints are saved to ``spectro/outputs/pretrained_models/spectro_{arch}_weights/``.

Usage::

    python spectro/scripts/spectro_pretrain.py --data synthetic --arch mamba --contrastive
    python spectro/scripts/spectro_pretrain.py --data synthetic --arch transformer --contrastive
    python spectro/scripts/spectro_pretrain.py --smoke         # tiny/fast sanity run
"""
from __future__ import annotations

import argparse
import csv
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
from spectro_patchify import build_masked_tensors  # noqa: E402
from contrastive import ProjectionHead, supervised_contrastive_loss  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

# Authors' contrastive loss weights (train_lwm_spectro_contrastive.py): MLM=1, mod=50, mob=30.
W_MLM, W_MOD, W_MOB = 1.0, 50.0, 30.0


def weights_dir(arch: str) -> str:
    return os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'pretrained_models', f'spectro_{arch}_weights')


def _wandb_log(run, data: dict):
    if run is not None:
        run.log(data)


def pretrain_expert(specs: torch.Tensor, *, arch, d_model, n_layers, mask_percent, epochs, lr,
                    batch_size, device, seed, val_frac=0.1, patience=4, grad_clip=1.0,
                    warmup_frac=0.1, weight_decay=0.0, min_lr=1e-5, accum_steps=1,
                    contrastive=False, mod=None, mob=None, w_mlm=W_MLM, w_mod=W_MOD, w_mob=W_MOB,
                    proj_dim=128, wandb_run=None, tag=''):
    """Pretrain one expert (``arch``). MLM, or MLM+SupCon when ``contrastive``. Returns best state."""
    ids, toks, pos = build_masked_tensors(specs, mask_percent=mask_percent, seed=seed)
    n = ids.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    if contrastive:
        mod_t = torch.as_tensor(np.asarray(mod), dtype=torch.long)
        mob_t = torch.as_tensor(np.asarray(mob), dtype=torch.long)
        tr_ds = TensorDataset(ids[tr_i], toks[tr_i], pos[tr_i], mod_t[tr_i], mob_t[tr_i])
        val_ds = TensorDataset(ids[val_i], toks[val_i], pos[val_i], mod_t[val_i], mob_t[val_i])
    else:
        tr_ds = TensorDataset(ids[tr_i], toks[tr_i], pos[tr_i])
        val_ds = TensorDataset(ids[val_i], toks[val_i], pos[val_i])
    # SupCon needs >1 sample per batch for positives; drop the ragged last batch when contrastive.
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=contrastive)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_expert(arch, d_model=d_model, n_layers=n_layers).to(device)
    params = list(model.parameters())
    proj_mod = proj_mob = None
    if contrastive:
        proj_mod = ProjectionHead(d_model, proj_dim).to(device)
        proj_mob = ProjectionHead(d_model, proj_dim).to(device)
        params += list(proj_mod.parameters()) + list(proj_mob.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # Linear warmup -> cosine decay to min_lr (per-epoch). A 12-layer transformer from random init
    # diverges at lr=1e-3 with no warmup (train loss climbs); warmup fixes the early instability.
    warmup_epochs = max(1, int(warmup_frac * epochs))
    if warmup_epochs < epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs),
                                                            eta_min=min_lr)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=min_lr)

    mse_sum = nn.MSELoss(reduction='sum')    # MLM-only path: per-sample-sum (original scale)
    mse_mean = nn.MSELoss(reduction='mean')  # contrastive path: scale-comparable to SupCon

    def _forward(batch):
        """Return (loss_to_backward, components_dict, batch_size)."""
        if contrastive:
            b_ids, b_toks, b_pos, b_mod, b_mob = (t.to(device) for t in batch)
            logits, output = model(b_ids, b_pos)
            mlm = mse_mean(logits, b_toks)
            sc_mod = supervised_contrastive_loss(proj_mod(output), b_mod)
            sc_mob = supervised_contrastive_loss(proj_mob(output), b_mob)
            total = w_mlm * mlm + w_mod * sc_mod + w_mob * sc_mob
            comp = {'mlm': mlm.item(), 'sc_mod': sc_mod.item(), 'sc_mob': sc_mob.item(),
                    'total': total.item()}
            return total, comp, b_ids.shape[0]
        b_ids, b_toks, b_pos = (t.to(device) for t in batch)
        loss = mse_sum(b_toks, model(b_ids, b_pos)[0])
        return loss, {'mse': loss.item()}, b_ids.shape[0]

    def _set_train(flag):
        model.train(flag)
        if contrastive:
            proj_mod.train(flag); proj_mob.train(flag)

    best_val, best_state, ctr = float('inf'), None, 0
    history = []
    for ep in range(epochs):
        _set_train(True)
        agg, seen = {}, 0
        opt.zero_grad()
        for i, batch in enumerate(tr_loader):
            loss, comp, bs = _forward(batch)
            (loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(tr_loader):
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step(); opt.zero_grad()
            for k, v in comp.items():
                agg[k] = agg.get(k, 0.0) + v * bs
            seen += bs
        sched.step()
        tr = {f'train_{k}': agg[k] / max(seen, 1) for k in agg}

        _set_train(False)
        vagg, vseen = {}, 0
        with torch.no_grad():
            for batch in val_loader:
                _, comp, bs = _forward(batch)
                for k, v in comp.items():
                    vagg[k] = vagg.get(k, 0.0) + v * bs
                vseen += bs
        va = {f'val_{k}': vagg[k] / max(vseen, 1) for k in vagg}

        # selection metric: total (contrastive) or mse (MLM-only)
        sel = va['val_total'] if contrastive else va['val_mse']
        row = {'epoch': ep + 1, **tr, **va, 'lr': opt.param_groups[0]['lr']}
        history.append(row)
        if contrastive:
            print(f"    epoch {ep+1}/{epochs}  total={va['val_total']:.4f}  mlm={va['val_mlm']:.4f}  "
                  f"sc_mod={va['val_sc_mod']:.4f}  sc_mob={va['val_sc_mob']:.4f}")
        else:
            print(f"    epoch {ep+1}/{epochs}  train_mse={tr['train_mse']:.4f}  val_mse={va['val_mse']:.4f}")
        _wandb_log(wandb_run, {f'{tag}/{k}': v for k, v in row.items() if k != 'epoch'} | {'epoch': ep + 1})

        if sel < best_val - 1e-6:
            best_val, ctr = sel, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            ctr += 1
            if ctr >= patience:
                print(f"    early stop @ epoch {ep+1} (best={best_val:.4f})")
                break
    return best_state, best_val, history


def train_router(specs: torch.Tensor, protocol: np.ndarray, *, epochs, lr, batch_size,
                 device, seed, val_frac=0.1, wandb_run=None):
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
        _wandb_log(wandb_run, {'router/val_acc': acc, 'router/epoch': ep + 1})
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
    return best_state, best_acc


def _build_pool(args):
    """Return (spectrograms, protocol, mod, mob) for pretraining, per --data.

    - demo:      demo_data train split only (val/test stay unseen for the downstream sweep).
    - synthetic: the full generated synthetic corpus.
    - mixed:     demo train split + synthetic corpus concatenated.
    mod/mob are per-sample integer labels (modulation / mobility) used by the contrastive objective.
    """
    parts_specs, parts_proto, parts_mod, parts_mob = [], [], [], []
    if args.data in ('demo', 'mixed'):
        data = load_spectro_data(seed=args.seed)
        tr = data.train_idx
        parts_specs.append(data.spectrograms[torch.as_tensor(tr)])
        parts_proto.append(data.protocol[tr])
        parts_mod.append(data.labels['modulation'][tr])
        parts_mob.append(data.labels['mobility'][tr])
    if args.data in ('synthetic', 'mixed'):
        syn = load_synthetic_data(args.synthetic_dir, seed=args.seed)
        parts_specs.append(syn.spectrograms)
        parts_proto.append(syn.protocol)
        parts_mod.append(syn.labels['modulation'])
        parts_mob.append(syn.labels['mobility'])
    specs = torch.cat(parts_specs, dim=0)
    proto = np.concatenate(parts_proto, axis=0)
    mod = np.concatenate(parts_mod, axis=0)
    mob = np.concatenate(parts_mob, axis=0)
    return specs, proto, mod, mob


def _init_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: --wandb set but wandb not installed; skipping W&B logging.")
        return None
    run = wandb.init(
        project=args.wandb_project, name=args.run_name or f"spectro-{args.arch}",
        config={'arch': args.arch, 'data': args.data, 'epochs': args.epochs, 'lr': args.lr,
                'batch_size': args.batch_size, 'mask_percent': args.mask_percent,
                'contrastive': args.contrastive, 'n_layers': args.n_layers,
                'w_mlm': args.w_mlm, 'w_mod': args.w_mod, 'w_mob': args.w_mob,
                'weight_decay': args.weight_decay, 'accum_steps': args.accum_steps})
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-layers', type=int, default=12)
    ap.add_argument('--mask-percent', type=float, default=0.6)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--router-epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--min-lr', type=float, default=1e-5)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--accum-steps', type=int, default=1,
                    help='gradient accumulation (effective batch = batch-size * accum-steps)')
    ap.add_argument('--warmup-frac', type=float, default=0.1,
                    help='fraction of epochs for linear LR warmup (authors use 5/20=0.25 w/ contrastive)')
    ap.add_argument('--weight-decay', type=float, default=0.0,
                    help='AdamW weight decay (authors use 0.05 for the contrastive recipe)')
    ap.add_argument('--grad-clip', type=float, default=1.0,
                    help='max grad norm (0 disables); guards against late-training MSE spikes')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--arch', choices=['mamba', 'transformer'], default='mamba',
                    help='expert architecture to pretrain (weights -> spectro_{arch}_weights/)')
    ap.add_argument('--smoke', action='store_true', help='tiny fast run for sanity checks')
    ap.add_argument('--data', choices=['demo', 'synthetic', 'mixed'], default='demo',
                    help="pretraining corpus: demo_data, the synthetic corpus, or both.")
    ap.add_argument('--synthetic-dir', default=os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'synthetic'),
                    help="directory of a generated synthetic corpus (manifest.json + shards).")
    # ---- contrastive (authors' flagship) ----
    ap.add_argument('--contrastive', action='store_true',
                    help='MLM + supervised contrastive on mod & mobility (authors\' flagship recipe)')
    ap.add_argument('--w-mlm', type=float, default=W_MLM)
    ap.add_argument('--w-mod', type=float, default=W_MOD)
    ap.add_argument('--w-mob', type=float, default=W_MOB)
    ap.add_argument('--proj-dim', type=int, default=128)
    # ---- W&B ----
    ap.add_argument('--wandb', action='store_true', help='log metrics/curves to Weights & Biases')
    ap.add_argument('--wandb-project', default='lwm-spectro')
    ap.add_argument('--run-name', default=None)
    args = ap.parse_args()

    if args.smoke:
        args.n_layers, args.epochs, args.router_epochs = 2, 2, 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = weights_dir(args.arch)
    os.makedirs(out_dir, exist_ok=True)

    pool_specs, pool_proto, pool_mod, pool_mob = _build_pool(args)
    wandb_run = _init_wandb(args)

    obj = 'MLM+SupCon(mod,mob)' if args.contrastive else 'MLM'
    print(f"Pretraining {args.arch} experts on device={device} (data={args.data}, objective={obj}, "
          f"n_layers={args.n_layers}, mask={args.mask_percent}, epochs={args.epochs}, "
          f"batch={args.batch_size}x{args.accum_steps}, lr={args.lr}, wd={args.weight_decay})")
    for p_idx, proto in enumerate(PROTOCOLS):
        sel = pool_proto == p_idx
        sel_t = torch.as_tensor(sel)
        specs = pool_specs[sel_t]
        mod, mob = pool_mod[sel], pool_mob[sel]
        print(f"\n[Expert {proto}] {specs.shape[0]} training spectrograms")
        if args.smoke:
            specs, mod, mob = specs[:64], mod[:64], mob[:64]
        state, val, history = pretrain_expert(
            specs, arch=args.arch, d_model=args.d_model, n_layers=args.n_layers,
            mask_percent=args.mask_percent, epochs=args.epochs, lr=args.lr, min_lr=args.min_lr,
            batch_size=args.batch_size, device=device, seed=args.seed, grad_clip=args.grad_clip,
            warmup_frac=args.warmup_frac, weight_decay=args.weight_decay, accum_steps=args.accum_steps,
            contrastive=args.contrastive, mod=mod, mob=mob,
            w_mlm=args.w_mlm, w_mod=args.w_mod, w_mob=args.w_mob, proj_dim=args.proj_dim,
            wandb_run=wandb_run, tag=proto)
        path = os.path.join(out_dir, f"{proto}_expert.pth")
        torch.save({'state_dict': state, 'val': val, 'arch': args.arch, 'contrastive': args.contrastive,
                    'd_model': args.d_model, 'n_layers': args.n_layers}, path)
        # per-epoch train/val loss curve (for choosing the epoch count)
        with open(os.path.join(out_dir, f"{proto}_losses.csv"), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
        print(f"[Expert {proto}] saved -> {path} (val={val:.4f})  curve -> {proto}_losses.csv")

    print("\n[Router] training protocol router")
    specs, proto = pool_specs, pool_proto
    if args.smoke:
        specs, proto = specs[:192], proto[:192]
    r_state, r_acc = train_router(specs, proto, epochs=args.router_epochs, lr=1e-3,
                                  batch_size=args.batch_size, device=device, seed=args.seed,
                                  wandb_run=wandb_run)
    r_path = os.path.join(out_dir, 'router.pth')
    torch.save({'state_dict': r_state, 'val_acc': r_acc, 'protocols': PROTOCOLS}, r_path)
    print(f"[Router] saved -> {r_path} (val_acc={r_acc:.3f})")
    print(f"\nDone. Pretrained {args.arch} MoE weights in", out_dir)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
