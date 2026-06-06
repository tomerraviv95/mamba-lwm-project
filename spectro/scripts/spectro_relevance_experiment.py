"""Is the synthetic data relevant to the demo tasks? Before/after pretraining, per architecture.

For each backbone architecture (HF Transformer-LWM and our bidirectional Mamba), we compare:
  - BEFORE: random-initialized backbone -> mean-pooled embeddings on the REAL demo spectrograms
  - AFTER : same backbone after masked-spectrogram pretraining on the SYNTHETIC corpus -> embeddings

Downstream, we train the same classification heads on (modulation / snr / mobility) over the demo
train/test split and sweep training-sample fractions. If AFTER > BEFORE, pretraining on the
synthetic data taught features useful for the real tasks -> the data is relevant.

This isolates the value of synthetic pretraining (architecture-controlled), unlike the main
``transformer`` arm which uses the HF model's precomputed embeddings.

Usage::

    CUDA_VISIBLE_DEVICES=1 python spectro/scripts/spectro_relevance_experiment.py \
        --n-layers 6 --pretrain-epochs 25
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectro_data import load_spectro_data, load_synthetic_data  # noqa: E402
from spectro_mamba_model import lwm_mamba_spectro  # noqa: E402
from spectro_patchify import CLS_TOKEN, build_masked_tensors, spectrogram_patchify  # noqa: E402
from spectro_sweep import run_sweep  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_SUBMISSIONS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'submissions')
_PLOTS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'plots')
_HF_LWM_PATH = os.path.join(_REPO_ROOT, 'spectro', 'hf_cache', 'pretraining', 'pretrained_model.py')


def _load_hf_lwm_class():
    """Import the HF LWM Transformer class from the downloaded source (avoids name clashes)."""
    spec = importlib.util.spec_from_file_location('hf_spectro_pretrained_model', _HF_LWM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LWM


def build_backbone(arch: str, n_layers: int):
    if arch == 'mamba':
        return lwm_mamba_spectro(element_length=16, d_model=128, n_layers=n_layers, max_len=1025)
    if arch == 'transformer':
        LWM = _load_hf_lwm_class()
        return LWM(element_length=16, d_model=128, n_layers=n_layers, max_len=1025, n_heads=8)
    raise ValueError(arch)


def _tokens_from_specs(specs: torch.Tensor) -> torch.Tensor:
    """Patchify + prepend CLS -> (N, 1025, 16) input_ids."""
    patches = spectrogram_patchify(specs, normalize=True)           # (N,1024,16)
    cls = np.broadcast_to(CLS_TOKEN, (patches.shape[0], 1, patches.shape[2]))
    return torch.tensor(np.concatenate([cls, patches], axis=1), dtype=torch.float32)


@torch.no_grad()
def extract_features(model, specs: torch.Tensor, device: str, batch_size: int = 16) -> torch.Tensor:
    """Mean-pooled (N, d_model) embeddings from a backbone over spectrograms."""
    model.to(device).eval()
    ids = _tokens_from_specs(specs)
    feats = []
    for start in range(0, ids.shape[0], batch_size):
        out = model(ids[start:start + batch_size].to(device))      # (B,1025,d_model)
        feats.append(out.mean(dim=1).cpu().float())
    return torch.cat(feats, dim=0)


def pretrain_backbone(model, specs: torch.Tensor, *, mask_percent, epochs, lr, batch_size,
                      device, seed, val_frac=0.1, patience=4):
    """Masked-spectrogram-modeling pretraining (MSE on masked patches). Mutates ``model`` in place."""
    ids, toks, pos = build_masked_tensors(specs, mask_percent=mask_percent, seed=seed)
    n = ids.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    tr = DataLoader(TensorDataset(ids[tr_i], toks[tr_i], pos[tr_i]), batch_size=batch_size, shuffle=True)
    va = DataLoader(TensorDataset(ids[val_i], toks[val_i], pos[val_i]), batch_size=batch_size)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    crit = nn.MSELoss(reduction='sum')
    best, best_state, ctr = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        for b_ids, b_toks, b_pos in tr:
            b_ids, b_toks, b_pos = b_ids.to(device), b_toks.to(device), b_pos.to(device)
            opt.zero_grad()
            loss = crit(b_toks, model(b_ids, b_pos)[0])
            loss.backward(); opt.step()
        sched.step()
        model.eval(); v, vn = 0.0, 0
        with torch.no_grad():
            for b_ids, b_toks, b_pos in va:
                b_ids, b_toks, b_pos = b_ids.to(device), b_toks.to(device), b_pos.to(device)
                v += crit(b_toks, model(b_ids, b_pos)[0]).item(); vn += b_ids.shape[0]
        v /= max(vn, 1)
        print(f"    pretrain epoch {ep+1}/{epochs} val_mse={v:.4f}")
        if v < best - 1e-6:
            best, ctr = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            ctr += 1
            if ctr >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', nargs='*', default=['transformer', 'mamba'])
    ap.add_argument('--n-layers', type=int, default=6)
    ap.add_argument('--pretrain-epochs', type=int, default=25)
    ap.add_argument('--mask-percent', type=float, default=0.6)
    ap.add_argument('--pretrain-lr', type=float, default=1e-3)
    ap.add_argument('--pretrain-batch', type=int, default=16)
    ap.add_argument('--extract-batch', type=int, default=16,
                    help='feature-extraction batch (small: Transformer attention is O(seq^2)).')
    ap.add_argument('--head-epochs', type=int, default=100)
    ap.add_argument('--synthetic-dir', default=os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'synthetic'))
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    demo = load_spectro_data(seed=args.seed)
    syn = load_synthetic_data(args.synthetic_dir, seed=args.seed)
    print(f"demo: {len(demo.spectrograms)} | synthetic: {len(syn.spectrograms)} | device={device}")

    summary = {}
    for arch in args.archs:
        print(f"\n===== {arch} =====")
        torch.manual_seed(args.seed)

        # BEFORE: random-init backbone features on demo
        model = build_backbone(arch, args.n_layers)
        feats_before = extract_features(model, demo.spectrograms, device, batch_size=args.extract_batch)
        print(f"[{arch}] before features {tuple(feats_before.shape)}; running sweep...")
        agg_before = run_sweep(f"{arch}_before", feats_before, demo,
                               os.path.join(_SUBMISSIONS, f'submission_relevance_{arch}_before'),
                               seed=args.seed, device=device, epochs_override=args.head_epochs)

        # AFTER: pretrain on synthetic, re-extract on demo
        print(f"[{arch}] pretraining on synthetic ({len(syn.spectrograms)} samples)...")
        pretrain_backbone(model, syn.spectrograms, mask_percent=args.mask_percent,
                          epochs=args.pretrain_epochs, lr=args.pretrain_lr,
                          batch_size=args.pretrain_batch, device=device, seed=args.seed)
        feats_after = extract_features(model, demo.spectrograms, device, batch_size=args.extract_batch)
        print(f"[{arch}] after features {tuple(feats_after.shape)}; running sweep...")
        agg_after = run_sweep(f"{arch}_after", feats_after, demo,
                              os.path.join(_SUBMISSIONS, f'submission_relevance_{arch}_after'),
                              seed=args.seed, device=device, epochs_override=args.head_epochs)
        summary[arch] = {'before': agg_before, 'after': agg_after}

    _plot_before_after(summary)


def _plot_before_after(summary):
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spectro_data import TASKS

    os.makedirs(_PLOTS, exist_ok=True)
    archs = list(summary.keys())
    tasks = list(TASKS.keys())
    ref = summary[archs[0]]['before']
    pcts = ref['experiment_config']['sample_percentages']

    fig, axes = plt.subplots(len(archs), len(tasks), figsize=(5 * len(tasks), 4 * len(archs)),
                             squeeze=False)
    for r, arch in enumerate(archs):
        for c, task in enumerate(tasks):
            ax = axes[r][c]
            for stage, color in [('before', '#7f7f7f'), ('after', '#d62728')]:
                res = summary[arch][stage]['results_by_task'][f'task_{task}']['results']
                xs = [res[str(p)]['n_samples'] for p in pcts]
                ys = [res[str(p)]['score'] for p in pcts]
                ax.plot(xs, ys, marker='o', color=color, label=stage, linewidth=2)
            ax.set_title(f'{arch} — {TASKS[task]["name"]}')
            ax.set_xlabel('# demo training samples'); ax.set_ylabel('accuracy')
            ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend()
    fig.suptitle('Synthetic-pretraining relevance: before vs after (demo tasks)', y=1.01)
    fig.tight_layout()
    out = os.path.join(_PLOTS, 'spectro_relevance_before_after.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"\nsaved -> {out}")

    # also print a compact delta table at full data (100%)
    print("\n=== AFTER - BEFORE accuracy delta @100% demo data ===")
    for arch in archs:
        for task in tasks:
            b = summary[arch]['before']['results_by_task'][f'task_{task}']['results'][str(pcts[-1])]['score']
            a = summary[arch]['after']['results_by_task'][f'task_{task}']['results'][str(pcts[-1])]['score']
            print(f"  {arch:11s} {TASKS[task]['name']:10s} before={b:.3f} after={a:.3f} delta={a-b:+.3f}")


if __name__ == '__main__':
    main()
