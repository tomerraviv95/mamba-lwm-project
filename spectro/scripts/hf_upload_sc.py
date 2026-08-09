"""Robust, resumable, memory-bounded upload of a single-carrier corpus/eval to an HF dataset repo.

Three problems this solves, all hit in practice on the 44 GB / 664-shard corpus:

1. **Xet OOM.** ``hf-xet`` (installed here) does content-defined chunking and dedup IN MEMORY. On
   66 MB shards it repeatedly took down a 25 GB WSL box mid-upload. ``cluster/download_data.sh``
   already disables Xet for the download path for the same reason; this sets
   ``HF_HUB_DISABLE_XET=1`` before huggingface_hub is imported, so uploads use classic streaming
   multipart HTTP with bounded memory.
2. **One commit per file.** Uploading 664 files individually means 664 commits — slow and easy to
   rate-limit. Files are batched into commits of ``--batch`` shards instead (~21 commits).
3. **Manifest ordering.** ``manifest.json`` is written LAST. ``download_data.sh`` verifies a
   dataset by reading the manifest's shard list, so a manifest that lands before its shards makes a
   truncated dataset verify CLEAN — the exact failure that silently cut a 132k corpus to 30k and
   cost a multi-day run. Until the manifest exists the dataset is visibly incomplete.

Resumable: remote files whose size already matches are skipped, so re-running after a crash only
sends what is missing.

Usage:
    python spectro/scripts/hf_upload_sc.py --repo tomerraviv95/lwm-spectro-scmax \
        --corpus-dir spectro/outputs/spectro_corpus_scmax_s1 \
        --eval-dir   spectro/outputs/spectro_eval_scmax_indist_s1
"""
from __future__ import annotations

import argparse
import json
import os
import time

# MUST precede the huggingface_hub import — the backend is selected at import time.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')


def remote_sizes(api, repo: str, sub: str) -> dict:
    try:
        return {f.path: getattr(f, 'size', None) for f in api.list_repo_tree(
            repo, repo_type='dataset', path_in_repo=sub, recursive=True)}
    except Exception:
        return {}


def upload_dir(api, repo: str, sub: str, path: str, batch: int, retries: int = 6) -> None:
    from huggingface_hub import CommitOperationAdd

    man_p = os.path.join(path, 'manifest.json')
    if not os.path.isfile(man_p):
        raise SystemExit(f"missing {man_p} — generate it before uploading.")
    shards = list(json.load(open(man_p))['shards'])

    remote = remote_sizes(api, repo, sub)
    todo = [s for s in shards
            if remote.get(f"{sub}/{s}") != os.path.getsize(os.path.join(path, s))]
    gb = sum(os.path.getsize(os.path.join(path, s)) for s in todo) / 1e9
    print(f"[{sub}] {len(shards)} shards | {len(shards)-len(todo)} already remote | "
          f"{len(todo)} to send ({gb:.1f} GB) in batches of {batch}", flush=True)

    t0 = time.time()
    sent = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        ops = [CommitOperationAdd(path_in_repo=f"{sub}/{s}",
                                  path_or_fileobj=os.path.join(path, s)) for s in chunk]
        for attempt in range(retries):
            try:
                api.create_commit(repo_id=repo, repo_type='dataset', operations=ops,
                                  commit_message=f"add {sub} shards {i}..{i+len(chunk)-1}")
                break
            except Exception as e:
                wait = min(120, 2 ** attempt)
                print(f"  batch {i}: {type(e).__name__}: {str(e)[:120]} "
                      f"(attempt {attempt+1}/{retries}) -> retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise SystemExit(f"[{sub}] giving up at batch {i} — re-run to resume")
        sent += len(chunk)
        el = time.time() - t0
        print(f"  [{sub}] {sent}/{len(todo)} shards  {el/60:.1f} min  "
              f"ETA {el/sent*(len(todo)-sent)/60:.0f} min", flush=True)

    # manifest LAST
    from huggingface_hub import CommitOperationAdd as _Add
    api.create_commit(repo_id=repo, repo_type='dataset',
                      operations=[_Add(path_in_repo=f"{sub}/manifest.json", path_or_fileobj=man_p)],
                      commit_message=f"add {sub} manifest")

    n_remote = sum(1 for p in remote_sizes(api, repo, sub) if str(p).endswith('.pt'))
    ok = n_remote == len(shards)
    print(f"[{sub}] {'OK' if ok else 'MISMATCH'}: {n_remote} shards remote vs {len(shards)} "
          f"in manifest{'' if ok else ' — re-run to resume'}", flush=True)
    if not ok:
        raise SystemExit(f"[{sub}] incomplete upload")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--corpus-dir', default=None)
    ap.add_argument('--eval-dir', default=None)
    ap.add_argument('--batch', type=int, default=25, help='shards per commit (~1.6 GB at 66 MB each)')
    ap.add_argument('--private', action='store_true')
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo
    print(f"xet disabled={os.environ.get('HF_HUB_DISABLE_XET')}", flush=True)
    api = HfApi()
    create_repo(args.repo, repo_type='dataset', exist_ok=True, private=args.private)
    for sub, path in [('corpus', args.corpus_dir), ('eval', args.eval_dir)]:
        if path:
            upload_dir(api, args.repo, sub, path, args.batch)
    print(f"done -> https://huggingface.co/datasets/{args.repo}")


if __name__ == '__main__':
    main()
