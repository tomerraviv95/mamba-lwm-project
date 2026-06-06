"""Download the LWM-Spectro baseline artifacts and source from Hugging Face.

The repo ``wi-lab/lwm-spectro`` ships the pretrained Transformer MoE baseline
(per-protocol experts + router), the demo spectrogram dataset, and the original
source code. We mirror the relevant pieces into ``spectro/hf_cache/`` so the rest
of the spectro pipeline can import the HF backbone class and load weights.

Usage::

    python spectro/scripts/download_spectro_hf.py            # download everything we need
    python spectro/scripts/download_spectro_hf.py --inspect  # download + print contract info
"""
import argparse
import os
import sys

from huggingface_hub import hf_hub_download

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_HF_CACHE = os.path.join(_REPO_ROOT, 'spectro', 'hf_cache')
_REPO_ID = 'wi-lab/lwm-spectro'

# Files we need: source (to import the backbone + router), config, weights, demo data.
_SOURCE_FILES = [
    'pretraining/pretrained_model.py',
    'mixture/train_embedding_router.py',
    'MoE/train_embedding_router.py',
    'MoE/train_top1_router.py',
    'task1/train_mcs_models.py',
    'task2/train_joint_snr_mobility.py',
    'task2/mobility_utils.py',
    'utils.py',
    '__init__.py',
    'config.json',
    'README_model.md',
    'README_code.md',
    'example_inference.py',
    'hf_minimal_inference.py',
]
_WEIGHT_FILES = [
    'moe_checkpoint.pth',
    'checkpoints/checkpoint.pth',
    'experts/LTE_expert.pth',
    'experts/WiFi_expert.pth',
    'experts/5G_expert.pth',
]
_DATA_FILES = [
    'demo_data.pt',
    'demo_data_moe.pt',
]


def download(files):
    """Download ``files`` from the HF repo into the local cache, return local paths."""
    paths = {}
    for fn in files:
        try:
            local = hf_hub_download(repo_id=_REPO_ID, filename=fn, local_dir=_HF_CACHE)
            paths[fn] = local
            print(f"  ok   {fn}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL {fn}: {exc}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true',
                        help='Print dataset/checkpoint contract info after download.')
    parser.add_argument('--skip-weights', action='store_true',
                        help='Skip the large .pth weight files.')
    args = parser.parse_args()

    os.makedirs(_HF_CACHE, exist_ok=True)
    print(f"Downloading source files into {_HF_CACHE} ...")
    download(_SOURCE_FILES)
    print("Downloading demo data ...")
    download(_DATA_FILES)
    if not args.skip_weights:
        print("Downloading weight files ...")
        download(_WEIGHT_FILES)

    if args.inspect:
        inspect()


def inspect():
    """Print the data/checkpoint contract that downstream code depends on."""
    import torch

    print("\n=== demo_data.pt ===")
    demo = os.path.join(_HF_CACHE, 'demo_data.pt')
    samples = torch.load(demo, weights_only=False)
    print(f"type={type(samples)} len={len(samples)}")
    s0 = samples[0]
    print(f"sample keys: {list(s0.keys())}")
    for k, v in s0.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v!r}")
    for field in ('tech', 'snr', 'mod', 'mob'):
        if field in s0:
            vals = sorted({str(s.get(field)) for s in samples})
            print(f"unique {field} ({len(vals)}): {vals}")

    print("\n=== moe_checkpoint.pth top-level keys ===")
    ckpt = os.path.join(_HF_CACHE, 'moe_checkpoint.pth')
    if os.path.exists(ckpt):
        obj = torch.load(ckpt, map_location='cpu', weights_only=False)
        if isinstance(obj, dict):
            for k in list(obj.keys())[:40]:
                v = obj[k]
                shape = tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__
                print(f"  {k}: {shape}")


if __name__ == '__main__':
    sys.exit(main())
