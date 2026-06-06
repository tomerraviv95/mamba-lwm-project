"""Main entry point: extract per-arm spectrogram features and run the sample-variation sweep.

Arms (``--arm``):
- ``transformer``: LWM-Spectro MoE baseline — precomputed ``moe_embedding`` (128-d).
- ``mamba``: our pretrained Mamba MoE — routed embeddings (128-d). Requires
  ``spectro/outputs/pretrained_models/spectro_mamba_weights/`` (run spectro_pretrain.py first).
- ``raw``: no backbone — mean-pooled raw 4x4 patches (16-d) as a lower bound.

Each arm produces a (N, d) feature matrix once, then ``spectro_sweep.run_sweep`` trains a head
per (task, sample-percentage) and writes
``spectro/outputs/submissions/submission_spectro_{arm}/aggregated_results.json``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectro_data import PROTOCOLS, load_spectro_data  # noqa: E402
from spectro_moe import MambaMoE  # noqa: E402
from spectro_patchify import spectrogram_patchify  # noqa: E402
from spectro_sweep import run_sweep  # noqa: E402
from spectro_transformer_model import get_baseline_features  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_WEIGHTS_DIR = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'pretrained_models', 'spectro_mamba_weights')
_SUBMISSIONS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'submissions')


def _raw_features(data) -> torch.Tensor:
    """Mean-pooled raw 4x4 patches -> (N, 16) lower-bound features."""
    patches = spectrogram_patchify(data.spectrograms, normalize=True)  # (N,1024,16)
    return torch.tensor(patches.mean(axis=1), dtype=torch.float32)


def _mamba_features(data, device, routing) -> torch.Tensor:
    """Load the pretrained Mamba MoE and extract routed embeddings -> (N, d_model)."""
    router_ckpt = torch.load(os.path.join(_WEIGHTS_DIR, 'router.pth'), map_location='cpu',
                             weights_only=False)
    sample_expert = torch.load(os.path.join(_WEIGHTS_DIR, f'{PROTOCOLS[0]}_expert.pth'),
                               map_location='cpu', weights_only=False)
    d_model = sample_expert.get('d_model', 128)
    n_layers = sample_expert.get('n_layers', 12)

    moe = MambaMoE(PROTOCOLS, d_model=d_model, n_layers=n_layers)
    for proto in PROTOCOLS:
        ckpt = torch.load(os.path.join(_WEIGHTS_DIR, f'{proto}_expert.pth'),
                          map_location='cpu', weights_only=False)
        moe.load_expert(proto, ckpt['state_dict'])
    moe.router.load_state_dict(router_ckpt['state_dict'])

    return moe.extract_embeddings(data.spectrograms, routing=routing,
                                  protocol_idx=data.protocol, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['transformer', 'mamba', 'raw'], required=True)
    ap.add_argument('--routing', choices=['router', 'oracle'], default='router',
                    help='Mamba-arm routing strategy.')
    ap.add_argument('--baseline', choices=['moe', 'tech'], default='moe',
                    help='Transformer-arm precomputed embedding to use.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=None, help='override head epochs (e.g. for smoke)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_spectro_data(seed=args.seed)

    print(f"Extracting features for arm={args.arm} ...")
    if args.arm == 'transformer':
        features = get_baseline_features(data, which=args.baseline)
    elif args.arm == 'mamba':
        features = _mamba_features(data, device, args.routing)
    else:
        features = _raw_features(data)
    print(f"features: {tuple(features.shape)}  finite={bool(torch.isfinite(features).all())}")

    out_dir = os.path.join(_SUBMISSIONS, f'submission_spectro_{args.arm}')
    run_sweep(args.arm, features, data, out_dir, seed=args.seed, device=device,
              epochs_override=args.epochs)
    print(f"\nDone. Results -> {out_dir}/aggregated_results.json")


if __name__ == '__main__':
    main()
