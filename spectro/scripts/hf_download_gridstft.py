"""Download the dual [STFT|grid] corpus + held-out eval from HF into the expected local dirs.

Mirror of hf_upload_gridstft.py — pulls <repo>/corpus/ and <repo>/eval/ into
spectro/outputs/spectro_deepmimo_mult8_vary_gridstft and .../spectro_eval_heldout_cities_gridstft
so the pretrain/sweep scripts find them at their usual paths. Idempotent: skips a dir that already
has a manifest unless --force.

Usage:
    .venv/bin/python spectro/scripts/hf_download_gridstft.py --repo tomerraviv95/lwm-spectro-gridstft
"""
from __future__ import annotations

import argparse
import os
import shutil

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
CORPUS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'spectro_deepmimo_mult8_vary_gridstft')
EVAL = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'spectro_eval_heldout_cities_gridstft')


def _fetch(repo, sub, target, force):
    from huggingface_hub import HfApi, hf_hub_download
    if os.path.isfile(os.path.join(target, 'manifest.json')) and not force:
        print(f"{target} already present — skip (use --force to re-download)")
        return
    os.makedirs(target, exist_ok=True)
    files = [f for f in HfApi().list_repo_files(repo, repo_type='dataset')
             if f.startswith(f'{sub}/') and (f.endswith('.pt') or f.endswith('manifest.json'))]
    if not files:
        raise SystemExit(f"no files under {sub}/ in {repo} — did the upload run?")
    for f in files:
        cached = hf_hub_download(repo, f, repo_type='dataset')
        shutil.copyfile(cached, os.path.join(target, os.path.basename(f)))
    print(f"{repo}:{sub}/ -> {target}  ({len(files)} files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='tomerraviv95/lwm-spectro-gridstft')
    ap.add_argument('--force', action='store_true', help='re-download even if the dir already exists.')
    args = ap.parse_args()
    _fetch(args.repo, 'corpus', CORPUS, args.force)
    _fetch(args.repo, 'eval', EVAL, args.force)


if __name__ == '__main__':
    main()
