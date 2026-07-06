"""Upload a dual [STFT|grid] corpus (and optionally the held-out eval) to an HF dataset repo.

Run from the machine that has the generated data (e.g. this local box); the cluster then pulls it
with hf_download_gridstft.py. Layout in the repo:

    <repo>/corpus/{manifest.json, shard_*.pt}
    <repo>/eval/{manifest.json, shard_*.pt}   (only if --eval-dir is given)

Examples (needs `huggingface-cli login` / `hf auth login` or HF_TOKEN):
    # original 40k corpus + eval
    .venv/bin/python spectro/scripts/hf_upload_gridstft.py --repo tomerraviv95/lwm-spectro-gridstft \
        --corpus-dir spectro/outputs/spectro_deepmimo_mult8_vary_gridstft \
        --eval-dir   spectro/outputs/spectro_eval_heldout_cities_gridstft
    # wider-diversity corpus only (eval unchanged, already uploaded)
    .venv/bin/python spectro/scripts/hf_upload_gridstft.py --repo tomerraviv95/lwm-spectro-gridstft-diverse \
        --corpus-dir spectro/outputs/spectro_deepmimo_diverse_gridstft
"""
from __future__ import annotations

import argparse
import os

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, help='target HF dataset repo id (created if missing).')
    ap.add_argument('--corpus-dir', default=os.path.join(_REPO_ROOT, 'spectro', 'outputs',
                                                          'spectro_deepmimo_mult8_vary_gridstft'))
    ap.add_argument('--eval-dir', default=None, help='optional held-out eval dir to also upload under eval/.')
    ap.add_argument('--private', action='store_true')
    args = ap.parse_args()
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(args.repo, repo_type='dataset', exist_ok=True, private=args.private)
    uploads = [('corpus', args.corpus_dir)] + ([('eval', args.eval_dir)] if args.eval_dir else [])
    for sub, path in uploads:
        if not os.path.isfile(os.path.join(path, 'manifest.json')):
            raise SystemExit(f"missing {path}/manifest.json — generate it before uploading.")
        print(f"uploading {path} -> {args.repo}:{sub}/ ...")
        api.upload_folder(folder_path=path, path_in_repo=sub, repo_id=args.repo,
                          repo_type='dataset', allow_patterns=['*.pt', 'manifest.json'])
    print(f"done -> https://huggingface.co/datasets/{args.repo}")


if __name__ == '__main__':
    main()
