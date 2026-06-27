"""Main entry point: extract per-arm spectrogram features and run the sample-variation sweep.

Arms (``--arm``):
- ``transformer``: LWM-Spectro MoE *baseline* — precomputed ``moe_embedding`` (real full corpus).
- ``transformer_synth``: our Transformer MoE pretrained on the synthetic corpus (fair vs mamba).
- ``mamba``: our Mamba MoE pretrained on the synthetic corpus — routed embeddings (128-d).
- ``raw``: no backbone — mean-pooled raw 4x4 patches (16-d) as a lower bound.

``transformer_synth`` and ``mamba`` are the *fair* backbone comparison (same data, same
extraction); ``transformer`` is the strong real-corpus reference. Each synth arm needs its
checkpoints from ``spectro_pretrain.py --arch {transformer,mamba}``.

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
from spectro_moe import SpectroMoE  # noqa: E402
from spectro_patchify import spectrogram_patchify  # noqa: E402
from spectro_sweep import run_sweep  # noqa: E402
from spectro_transformer_model import get_baseline_features  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_PRETRAINED = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'pretrained_models')
_SUBMISSIONS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'submissions')

# arm -> (architecture, weights subdir) for the synthetic-pretrained MoE arms
_MOE_ARMS = {
    'mamba': ('mamba', 'spectro_mamba_weights'),
    'transformer_synth': ('transformer', 'spectro_transformer_weights'),
}


def _raw_features(data) -> torch.Tensor:
    """Mean-pooled raw 4x4 patches -> (N, 16) lower-bound features."""
    patches = spectrogram_patchify(data.spectrograms, normalize=True)  # (N,1024,16)
    return torch.tensor(patches.mean(axis=1), dtype=torch.float32)


def _random_init_features(data, device, arch='transformer', pool='mean', seed=42) -> torch.Tensor:
    """Untrained MoE (random weights) embeddings, oracle routing -> isolates the pretraining LIFT
    (random-init backbone is the no-pretraining-but-same-architecture baseline)."""
    torch.manual_seed(seed)
    moe = SpectroMoE(PROTOCOLS, d_model=128, arch=arch, n_layers=12, pool=pool)
    return moe.extract_embeddings(data.spectrograms, routing='oracle',
                                  protocol_idx=data.protocol, device=device)


def _moe_features(data, device, routing, arch, weights_subdir) -> torch.Tensor:
    """Load a pretrained MoE (Mamba or Transformer) and extract routed embeddings -> (N, d_model)."""
    wdir = os.path.join(_PRETRAINED, weights_subdir)
    if not os.path.exists(os.path.join(wdir, 'router.pth')):
        raise FileNotFoundError(
            f"No checkpoints in {wdir}. Run: python spectro/scripts/spectro_pretrain.py "
            f"--data synthetic --arch {arch}")
    router_ckpt = torch.load(os.path.join(wdir, 'router.pth'), map_location='cpu', weights_only=False)
    sample_expert = torch.load(os.path.join(wdir, f'{PROTOCOLS[0]}_expert.pth'),
                               map_location='cpu', weights_only=False)
    d_model = sample_expert.get('d_model', 128)
    n_layers = sample_expert.get('n_layers', 12)

    moe = SpectroMoE(PROTOCOLS, d_model=d_model, arch=arch, n_layers=n_layers)
    for proto in PROTOCOLS:
        ckpt = torch.load(os.path.join(wdir, f'{proto}_expert.pth'),
                          map_location='cpu', weights_only=False)
        moe.load_expert(proto, ckpt['state_dict'])
    moe.router.load_state_dict(router_ckpt['state_dict'])

    return moe.extract_embeddings(data.spectrograms, routing=routing,
                                  protocol_idx=data.protocol, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['transformer', 'transformer_synth', 'mamba', 'raw', 'random_init'],
                    required=True)
    ap.add_argument('--routing', choices=['router', 'oracle'], default='router',
                    help='routing strategy for the synthetic-pretrained MoE arms.')
    ap.add_argument('--baseline', choices=['moe', 'tech'], default='moe',
                    help='Transformer-baseline precomputed embedding to use.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=None, help='override head epochs (e.g. for smoke)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_spectro_data(seed=args.seed)

    print(f"Extracting features for arm={args.arm} ...")
    if args.arm == 'transformer':
        features = get_baseline_features(data, which=args.baseline)
    elif args.arm == 'random_init':
        features = _random_init_features(data, device, arch='transformer', seed=args.seed)
    elif args.arm in _MOE_ARMS:
        arch, weights_subdir = _MOE_ARMS[args.arm]
        features = _moe_features(data, device, args.routing, arch, weights_subdir)
    else:
        features = _raw_features(data)
    print(f"features: {tuple(features.shape)}  finite={bool(torch.isfinite(features).all())}")

    out_dir = os.path.join(_SUBMISSIONS, f'submission_spectro_{args.arm}')
    run_sweep(args.arm, features, data, out_dir, seed=args.seed, device=device,
              epochs_override=args.epochs)
    print(f"\nDone. Results -> {out_dir}/aggregated_results.json")


if __name__ == '__main__':
    main()
