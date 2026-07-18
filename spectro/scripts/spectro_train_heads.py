"""Main entry point: extract per-arm spectrogram features and run the sample-variation sweep.

Arms (``--arm``):
- ``transformer``: LWM-Spectro MoE *baseline* — precomputed ``moe_embedding`` (real full corpus).
- ``transformer_synth``: our Transformer MoE pretrained on the synthetic corpus (fair vs mamba).
- ``mamba``: our Mamba MoE pretrained on the synthetic corpus.
- ``random_init``: untrained MoE (isolates the pretraining lift).
- ``raw``: mean-pooled raw 4x4 patches (lower bound).
- ``resnet18``/``resnet50``/``efficientnet_b0``/``mobilenet_v3_small``: frozen ImageNet backbones.
- ``deepcnn``: from-scratch supervised CNN trained end-to-end (paper's Deep CNN reference).

``transformer_synth`` and ``mamba`` are the *fair* backbone comparison (same data, same recipe).
The MoE arms use the paper downstream head by default: a residual 1-D CNN over the encoder TOKEN
SEQUENCE (``--head cnn1d``; ``--head mlp`` falls back to a pooled-feature probe). Metric = macro-F1
(primary) + accuracy; the few-shot axis is ``--per-class-counts`` (samples/class) or ``--sample-counts``.
Results -> ``spectro/outputs/submissions/submission_spectro_{arm}_p{patch}{tag}/aggregated_results.json``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

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
_IMAGENET_ARMS = ('resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_small')


class DeepCNN(nn.Module):
    """From-scratch supervised CNN baseline over the raw spectrogram (approximates Wu et al. [7]).

    Unlike the frozen-feature arms, this one is trained END-TO-END on the labelled downstream data
    (no pretraining) — the paper's "Deep CNN" reference that trails the LWM MoE in the few-shot regime.
    ``feat_dim`` is the pooled feature width the sweep's linear head sits on."""
    feat_dim = 256

    def __init__(self, in_channels: int = 1):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(in_channels, 32), block(32, 64), block(64, 128), block(128, self.feat_dim),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.features(x)


def _random_project(features: torch.Tensor, out_dim: int, seed: int = 0) -> torch.Tensor:
    """Fixed seeded Gaussian random projection -> (N, out_dim). Only shrinks features WIDER than
    out_dim; returns the input unchanged otherwise. Scaled by 1/sqrt(out_dim) so it approximately
    preserves pairwise distances (Johnson-Lindenstrauss), giving every arm an equal-width feature
    for a fair head comparison (e.g. ResNet-50 2048-d / ResNet-18 512-d -> 256-d like our MoE arms)."""
    if features.shape[1] <= out_dim:
        return features
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(features.shape[1], out_dim, generator=g) / (out_dim ** 0.5)
    return features.float() @ W


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
    from torchvision.models import (resnet18, ResNet18_Weights, resnet50, ResNet50_Weights,
                                    efficientnet_b0, EfficientNet_B0_Weights,
                                    mobilenet_v3_small, MobileNet_V3_Small_Weights)
    # (ctor, weights, classifier-attribute to replace with Identity to expose the pooled feature)
    reg = {'resnet18': (resnet18, ResNet18_Weights.IMAGENET1K_V1, 'fc'),
           'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1, 'fc'),
           'efficientnet_b0': (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 'classifier'),
           'mobilenet_v3_small': (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1, 'classifier')}
    ctor, weights, clf_attr = reg[model_name]
    model = ctor(weights=weights)
    setattr(model, clf_attr, torch.nn.Identity())           # -> pooled feature vector (no classifier)
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


def _random_init_features(data, device, arch='transformer', pool='mean', seed=42, patch=4,
                          as_sequence=False) -> torch.Tensor:
    """Untrained MoE (random weights) features, oracle routing -> isolates the pretraining LIFT
    (random-init backbone is the no-pretraining-but-same-architecture baseline).
    ``as_sequence`` -> (N, T, d) token sequences for the CNN head; else pooled (N, d)."""
    torch.manual_seed(seed)
    channels = data.spectrograms.shape[1] if data.spectrograms.ndim == 4 else 1
    geom = patch_geometry(patch, channels=channels)
    moe = SpectroMoE(PROTOCOLS, d_model=128, arch=arch, n_layers=12, pool=pool, patch=patch,
                     element_length=geom['element_length'], max_len=geom['max_len'], in_channels=channels)
    fn = moe.extract_sequences if as_sequence else moe.extract_embeddings
    return fn(data.spectrograms, routing='oracle', protocol_idx=data.protocol, device=device)


def _moe_features(data, device, routing, arch, patch, pool='mean', weights_suffix='',
                  as_sequence=False) -> torch.Tensor:
    """Load a pretrained MoE (Mamba or Transformer) at the given patch size and extract routed
    features. ``as_sequence`` -> per-token (N, T, d) for the paper CNN head; else pooled (N, d).
    Geometry (element_length, max_len, patch) is read from the checkpoint."""
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

    fn = moe.extract_sequences if as_sequence else moe.extract_embeddings
    return fn(data.spectrograms, routing=routing, protocol_idx=data.protocol, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['transformer', 'transformer_synth', 'mamba', 'raw', 'random_init',
                                      'resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_small',
                                      'deepcnn'], required=True)
    ap.add_argument('--head', choices=['cnn1d', 'mlp'], default='cnn1d',
                    help="downstream head for the MoE arms: paper residual 1-D CNN over the token "
                         "sequence (cnn1d, default) or a pooled-feature MLP probe (mlp).")
    ap.add_argument('--routing', choices=['router', 'oracle'], default='router',
                    help='routing strategy for the synthetic-pretrained MoE arms.')
    ap.add_argument('--moe-arch', choices=['mamba', 'transformer'], default='mamba',
                    help="architecture for the random_init MoE baseline (match the compared arm so the "
                         "pretraining lift is measured against the SAME untrained architecture).")
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
    ap.add_argument('--run-tag', default='',
                    help="extra suffix on the submission dir to disambiguate eval sets "
                         "(e.g. 'alluser15' vs 'heldoutcities' so two evals don't collide).")
    ap.add_argument('--val-frac', type=float, default=0.10, help='val fraction (70/10/20 split).')
    ap.add_argument('--test-frac', type=float, default=0.20, help='test fraction (70/10/20 split).')
    ap.add_argument('--epochs', type=int, default=None, help='override head epochs (e.g. for smoke)')
    ap.add_argument('--sample-counts', type=int, nargs='+', default=None,
                    help='absolute #training-samples to sweep (e.g. 50 100 250 500 1000 2500 4000). '
                         'Overrides the default percentage sweep.')
    ap.add_argument('--per-class-counts', type=int, nargs='+', default=None,
                    help="paper few-shot axis: #training samples PER CLASS (e.g. 2 4 8 16 32 64 128 256). "
                         "Takes priority over --sample-counts.")
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='average each sample point over these head-training seeds (e.g. 42 43 44) '
                         'to smooth curve noise. Feature extraction is unaffected (single data split).')
    ap.add_argument('--head-restarts', type=int, default=1,
                    help='train this many head inits per (task,count,seed), keep best-on-val. '
                         '>1 rejects degenerate collapsed-to-chance heads at low sample counts.')
    ap.add_argument('--project-dim', type=int, default=None,
                    help='random-project features WIDER than this down to it (fair equal-width head '
                         'comparison; e.g. 256 shrinks ResNet 512/2048-d, leaves our 256-d arms as-is).')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.synth_dir:
        if args.arm == 'transformer':
            raise SystemExit("--arm transformer (published baseline) needs precomputed demo embeddings; "
                             "it cannot run on a synthetic --synth-dir. Use transformer_synth/mamba/random_init/raw.")
        data = load_synthetic_data(args.synth_dir, seed=args.seed,
                                   val_frac=args.val_frac, test_frac=args.test_frac)
    else:
        data = load_spectro_data(seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac)

    print(f"Extracting features for arm={args.arm} patch={args.patch} pool={args.pool} head={args.head} "
          f"eval={'synth:'+os.path.basename(args.synth_dir.rstrip('/')) if args.synth_dir else 'demo'} ...")
    seq = args.head == 'cnn1d'                    # MoE arms: extract per-token sequences for the CNN head
    backbone_factory = embed_fn = None
    if args.arm == 'transformer':
        features = get_baseline_features(data, which=args.baseline)
    elif args.arm in _IMAGENET_ARMS:
        features = _imagenet_features(data, args.arm, device)
    elif args.arm == 'deepcnn':
        # end-to-end trained baseline: pass raw spectrograms; the sweep trains a fresh DeepCNN per point
        specs = data.spectrograms
        features = specs.float() if torch.is_tensor(specs) else torch.as_tensor(np.asarray(specs), dtype=torch.float32)
        channels = features.shape[1] if features.dim() == 4 else 1
        backbone_factory = lambda: DeepCNN(in_channels=channels)   # noqa: E731
        embed_fn = lambda bb, x: bb(x)                             # noqa: E731
    elif args.arm == 'random_init':
        features = _random_init_features(data, device, arch=args.moe_arch, seed=args.seed,
                                         patch=args.patch, pool=args.pool, as_sequence=seq)
    elif args.arm in _MOE_ARMS:
        arch = _MOE_ARMS[args.arm]
        features = _moe_features(data, device, args.routing, arch, args.patch, pool=args.pool,
                                 weights_suffix=args.weights_suffix, as_sequence=seq)
    else:
        features = _raw_features(data, patch=args.patch)
    if args.project_dim and features.dim() == 2:   # equal-width head comparison (pooled features only)
        pre = features.shape[1]
        features = _random_project(features, args.project_dim, seed=args.seed)
        if features.shape[1] != pre:
            print(f"random-projected features {pre} -> {features.shape[1]} (equal-width head comparison)")
    print(f"features: {tuple(features.shape)}  finite={bool(torch.isfinite(features.float()).all())}")

    # keep in-domain (_heldout) and representation (_grid) results separate from the demo/STFT sweeps
    tag = (('_heldout' if args.synth_dir else '') + (f'_{args.weights_suffix}' if args.weights_suffix else '')
           + (f'_{args.run_tag}' if args.run_tag else ''))
    out_dir = os.path.join(_SUBMISSIONS, f'submission_spectro_{args.arm}_p{args.patch}{tag}')
    run_sweep(args.arm, features, data, out_dir, seed=args.seed, seeds=args.seeds,
              sample_counts=args.sample_counts, per_class_counts=args.per_class_counts,
              head_restarts=args.head_restarts, device=device, epochs_override=args.epochs,
              backbone_factory=backbone_factory, embed_fn=embed_fn)
    print(f"\nDone. Results -> {out_dir}/aggregated_results.json")


if __name__ == '__main__':
    main()
