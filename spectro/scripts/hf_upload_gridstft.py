"""Upload the dual [STFT|grid] pretrain corpus + held-out eval to an HF dataset repo.

Run this ONCE from the machine that has the generated data (e.g. this local box); the cluster then
pulls it with hf_download_gridstft.py. Both dirs go under prefixes in a single dataset repo:

    <repo>/corpus/{manifest.json, shard_*.pt}   <- spectro_deepmimo_mult8_vary_gridstft (40k)
    <repo>/eval/{manifest.json, shard_*.pt}      <- spectro_eval_heldout_cities_gridstft (6k)

Usage (needs `huggingface-cli login` or HF_TOKEN):
    .venv/bin/python spectro/scripts/hf_upload_gridstft.py --repo tomerraviv95/lwm-spectro-gridstft
"""
from __future__ import annotations

import argparse
import os

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
CORPUS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'spectro_deepmimo_mult8_vary_gridstft')
EVAL = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'spectro_eval_heldout_cities_gridstft')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='tomerraviv95/lwm-spectro-gridstft',
                    help='target HF dataset repo id (created if missing).')
    ap.add_argument('--private', action='store_true', help='create the repo private.')
    args = ap.parse_args()
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(args.repo, repo_type='dataset', exist_ok=True, private=args.private)
    for sub, path in [('corpus', CORPUS), ('eval', EVAL)]:
        if not os.path.isfile(os.path.join(path, 'manifest.json')):
            raise SystemExit(f"missing {path}/manifest.json — generate it before uploading.")
        print(f"uploading {path} -> {args.repo}:{sub}/ ...")
        api.upload_folder(folder_path=path, path_in_repo=sub, repo_id=args.repo,
                          repo_type='dataset', allow_patterns=['*.pt', 'manifest.json'])
    print(f"done -> https://huggingface.co/datasets/{args.repo}")


if __name__ == '__main__':
    main()
