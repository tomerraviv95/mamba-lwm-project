"""Push/pull the synthetic corpus and pretrained checkpoints to/from the Hugging Face Hub.

Keeps both the generated dataset and the trained weights in your HF account, so cluster runs
are reproducible and portable (generate/pretrain on a compute node -> publish from the login
node -> pull anywhere). Auth uses the ``HF_TOKEN`` env var (or a prior ``huggingface-cli login``;
the cached token in ``~/.cache/huggingface`` persists across Slurm jobs on shared home).

Examples::

    # publish (login node, needs internet + a WRITE token)
    python spectro/scripts/hf_sync.py push-dataset --repo <user>/lwm-spectro-synthetic --dir spectro/outputs/synthetic --private
    python spectro/scripts/hf_sync.py push-ckpts   --repo <user>/wimamba-spectro-ckpts  --dir spectro/outputs/pretrained_models --private

    # fetch (anywhere)
    python spectro/scripts/hf_sync.py pull-dataset --repo <user>/lwm-spectro-synthetic --dir spectro/outputs/synthetic
    python spectro/scripts/hf_sync.py pull-ckpts   --repo <user>/wimamba-spectro-ckpts  --dir spectro/outputs/pretrained_models
"""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download


def _token() -> str | None:
    # huggingface_hub also auto-reads HF_TOKEN, but we pass explicitly for clear errors.
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _push(repo: str, local_dir: str, repo_type: str, private: bool):
    if not os.path.isdir(local_dir):
        sys.exit(f"local dir not found: {local_dir}")
    api = HfApi(token=_token())
    api.create_repo(repo_id=repo, repo_type=repo_type, private=private, exist_ok=True)
    print(f"uploading {local_dir} -> {repo_type}:{repo} (private={private}) ...")
    api.upload_folder(repo_id=repo, repo_type=repo_type, folder_path=local_dir,
                      commit_message="sync from cluster")
    print(f"done: https://huggingface.co/{'datasets/' if repo_type=='dataset' else ''}{repo}")


def _pull(repo: str, local_dir: str, repo_type: str):
    os.makedirs(local_dir, exist_ok=True)
    print(f"downloading {repo_type}:{repo} -> {local_dir} ...")
    snapshot_download(repo_id=repo, repo_type=repo_type, local_dir=local_dir, token=_token())
    print("done.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, rt, push in [("push-dataset", "dataset", True), ("pull-dataset", "dataset", False),
                          ("push-ckpts", "model", True), ("pull-ckpts", "model", False)]:
        p = sub.add_parser(cmd)
        p.add_argument("--repo", required=True, help="HF repo id, e.g. user/name")
        p.add_argument("--dir", required=True, help="local folder")
        if push:
            p.add_argument("--private", action="store_true")
        p.set_defaults(repo_type=rt, is_push=push)

    args = ap.parse_args()
    if args.is_push and _token() is None:
        print("WARNING: no HF_TOKEN set; relying on cached `huggingface-cli login`.", file=sys.stderr)
    if args.is_push:
        _push(args.repo, args.dir, args.repo_type, getattr(args, "private", False))
    else:
        _pull(args.repo, args.dir, args.repo_type)


if __name__ == "__main__":
    main()
