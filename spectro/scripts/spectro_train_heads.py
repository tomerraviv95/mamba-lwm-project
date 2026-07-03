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
from spectro_data import PROTOCOLS, load_spectro_data, load_synthetic_data  # noqa: E402
from spectro_moe import SpectroMoE  # noqa: E402
from spectro_patchify import spectrogram_patchify, patch_geometry  # noqa: E402
from spectro_sweep import run_sweep  # noqa: E402
from spectro_transformer_model import get_baseline_features  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_PRETRAINED = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'pretrained_models')
_SUBMISSIONS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'submissions')

# arm -> architecture for the synthetic-pretrained MoE arms (weights subdir is patch-parameterized
# at call time: spectro_{arch}_p{patch}_weights).
_MOE_ARMS = {'mamba': 'mamba', 'transformer_synth': 'transformer'}


def _raw_features(data, patch=4) -> torch.Tensor:
    """Mean-pooled raw patches -> (N, patch*patch) lower-bound features."""
    patches = spectrogram_patchify(data.spectrograms, patch=patch, normalize=True)  # (N,n_patches,E)
    return torch.tensor(patches.mean(axis=1), dtype=torch.float32)


def _imagenet_features(data, model_name, device, batch=64) -> torch.Tensor:
    """Frozen ImageNet-pretrained vision backbone as a fixed feature extractor -> (N, feat_dim).

    A generic-vision baseline: the spectrogram is turned into a 3-channel 224x224 image (2-channel
    dual [STFT|grid] -> [stft, grid, mean]; 1-channel -> replicated) and run through a frozen
    torchvision model with its classifier removed. Shows how much a domain-agnostic pretrained CNN
    recovers vs. our in-domain LWM MoE."""
    import torch.nn.functional as F
    from torchvision.models import (resnet18, ResNet18_Weights, resnet50, ResNet50_Weights)
    ctor = {'resnet18': (resnet18, ResNet18_Weights.IMAGENET1K_V1),
            'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1)}[model_name]
    model = ctor[0](weights=ctor[1]); model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    specs = data.spectrograms
    if torch.is_tensor(specs):
        specs = specs.float()
    else:
        specs = torch.as_tensor(np.asarray(specs), dtype=torch.float32)
    if specs.ndim == 3:                       # (N,128,128) -> (N,1,128,128)
        specs = specs.unsqueeze(1)
    out = []
    with torch.no_grad():
        for s in range(0, specs.shape[0], batch):
            x = specs[s:s + batch].to(device)                       # (b,C,128,128), C in {1,2}
            if x.shape[1] == 2:
                x = torch.stack([x[:, 0], x[:, 1], x.mean(1)], dim=1)      # 3ch [stft, grid, mean]
            elif x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            out.append(model(x).float().cpu())
    return torch.cat(out)


def _random_init_features(data, device, arch='transformer', pool='mean', seed=42, patch=4) -> torch.Tensor:
    """Untrained MoE (random weights) embeddings, oracle routing -> isolates the pretraining LIFT
    (random-init backbone is the no-pretraining-but-same-architecture baseline)."""
    torch.manual_seed(seed)
    channels = data.spectrograms.shape[1] if data.spectrograms.ndim == 4 else 1
    geom = patch_geometry(patch, channels=channels)
    moe = SpectroMoE(PROTOCOLS, d_model=128, arch=arch, n_layers=12, pool=pool, patch=patch,
                     element_length=geom['element_length'], max_len=geom['max_len'], in_channels=channels)
    return moe.extract_embeddings(data.spectrograms, routing='oracle',
                                  protocol_idx=data.protocol, device=device)


def _moe_features(data, device, routing, arch, patch, pool='mean', weights_suffix='') -> torch.Tensor:
    """Load a pretrained MoE (Mamba or Transformer) at the given patch size and extract routed
    embeddings -> (N, d_model). Geometry (element_length, max_len, patch) is read from the checkpoint."""
    from spectro_pretrain import weights_dir
    wdir = weights_dir(arch, patch, weights_suffix)
    if not os.path.exists(os.path.join(wdir, 'router.pth')):
        # legacy fallback: pre-M3 patch-4 checkpoints live in the un-suffixed dir (spectro_{arch}_weights)
        legacy = os.path.join(_PRETRAINED, f'spectro_{arch}_weights')
        if not weights_suffix and patch == 4 and os.path.exists(os.path.join(legacy, 'router.pth')):
            wdir = legacy
        else:
            raise FileNotFoundError(
                f"No checkpoints in {wdir}. Run: python spectro/scripts/spectro_pretrain_real.py "
                f"--arch {arch} --patch {patch}")
    router_ckpt = torch.load(os.path.join(wdir, 'router.pth'), map_location='cpu', weights_only=False)
    sample_expert = torch.load(os.path.join(wdir, f'{PROTOCOLS[0]}_expert.pth'),
                               map_location='cpu', weights_only=False)
    d_model = sample_expert.get('d_model', 128)
    n_layers = sample_expert.get('n_layers', 12)
    geom = patch_geometry(sample_expert.get('patch', patch))
    element_length = sample_expert.get('element_length', geom['element_length'])
    max_len = sample_expert.get('max_len', geom['max_len'])
    in_channels = max(1, element_length // (patch * patch))     # 2 for grid_stft/complex checkpoints

    moe = SpectroMoE(PROTOCOLS, d_model=d_model, arch=arch, n_layers=n_layers, pool=pool, patch=patch,
                     element_length=element_length, max_len=max_len, in_channels=in_channels)
    for proto in PROTOCOLS:
        ckpt = torch.load(os.path.join(wdir, f'{proto}_expert.pth'),
                          map_location='cpu', weights_only=False)
        moe.load_expert(proto, ckpt['state_dict'])
    moe.router.load_state_dict(router_ckpt['state_dict'])

    return moe.extract_embeddings(data.spectrograms, routing=routing,
                                  protocol_idx=data.protocol, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['transformer', 'transformer_synth', 'mamba', 'raw', 'random_init',
                                      'resnet18', 'resnet50'], required=True)
    ap.add_argument('--routing', choices=['router', 'oracle'], default='router',
                    help='routing strategy for the synthetic-pretrained MoE arms.')
    ap.add_argument('--baseline', choices=['moe', 'tech'], default='moe',
                    help='Transformer-baseline precomputed embedding to use.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--patch', type=int, default=4, choices=[4, 6, 8],
                    help='patch side; must match the pretrained checkpoints. Stamped into the submission dir.')
    ap.add_argument('--pool', choices=['mean', 'cls', 'meanstd_t'], default='meanstd_t',
                    help="MoE-arm embedding pooling. M1 recipe default 'meanstd_t' (mean ++ per-freq temporal std).")
    ap.add_argument('--synth-dir', default=None,
                    help="evaluate IN-DOMAIN on a held-out synthetic corpus dir (manifest.json + shards) "
                         "instead of the demo. Use a corpus DISJOINT from pretraining (e.g. held-out cities) "
                         "to measure honest lift over random-init. The 'transformer' (published) arm is "
                         "unavailable here (no precomputed embeddings for synthetic data).")
    ap.add_argument('--weights-suffix', default='',
                    help="load MoE checkpoints from spectro_{arch}_p{patch}_{suffix}_weights (e.g. 'grid').")
    ap.add_argument('--epochs', type=int, default=None, help='override head epochs (e.g. for smoke)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.synth_dir:
        if args.arm == 'transformer':
            raise SystemExit("--arm transformer (published baseline) needs precomputed demo embeddings; "
                             "it cannot run on a synthetic --synth-dir. Use transformer_synth/mamba/random_init/raw.")
        data = load_synthetic_data(args.synth_dir, seed=args.seed)
    else:
        data = load_spectro_data(seed=args.seed)

    print(f"Extracting features for arm={args.arm} patch={args.patch} pool={args.pool} "
          f"eval={'synth:'+os.path.basename(args.synth_dir.rstrip('/')) if args.synth_dir else 'demo'} ...")
    if args.arm == 'transformer':
        features = get_baseline_features(data, which=args.baseline)
    elif args.arm in ('resnet18', 'resnet50'):
        features = _imagenet_features(data, args.arm, device)
    elif args.arm == 'random_init':
        features = _random_init_features(data, device, arch='transformer', seed=args.seed,
                                         patch=args.patch, pool=args.pool)
    elif args.arm in _MOE_ARMS:
        arch = _MOE_ARMS[args.arm]
        features = _moe_features(data, device, args.routing, arch, args.patch, pool=args.pool,
                                 weights_suffix=args.weights_suffix)
    else:
        features = _raw_features(data, patch=args.patch)
    print(f"features: {tuple(features.shape)}  finite={bool(torch.isfinite(features).all())}")

    # keep in-domain (_heldout) and representation (_grid) results separate from the demo/STFT sweeps
    tag = ('_heldout' if args.synth_dir else '') + (f'_{args.weights_suffix}' if args.weights_suffix else '')
    out_dir = os.path.join(_SUBMISSIONS, f'submission_spectro_{args.arm}_p{args.patch}{tag}')
    run_sweep(args.arm, features, data, out_dir, seed=args.seed, device=device,
              epochs_override=args.epochs)
    print(f"\nDone. Results -> {out_dir}/aggregated_results.json")


if __name__ == '__main__':
    main()
