"""Robust, resumable, disk- and memory-bounded download of a corpus/eval from an HF dataset repo.

Mirror of hf_upload_sc.py. Replaces hf_download_gridstft.py for the single-carrier datasets, which
are 44 GB / 664 shards and hit three failure modes the old fetcher does not handle:

1. **Xet.** ``download_data.sh`` claims "the xet backend is disabled", but nothing ever set
   ``HF_HUB_DISABLE_XET`` — the comment is stale. ``hf-xet`` does content-defined chunking in
   memory and killed a 25 GB host twice during upload of these same shards. Disabled here before
   huggingface_hub is imported (the backend is chosen at import time).
2. **No retries.** The old ``_fetch`` loops over ``hf_hub_download`` with no error handling, so a
   single transient network error aborts the whole download. Uploading these shards produced 30
   DNS resolution failures; over 664 files a bare loop will not finish. Each file is retried with
   exponential backoff here.
3. **Double disk.** The old path downloads into ``~/.cache/huggingface`` and then copies to the
   target — 88 GB for a 44 GB corpus, which overruns a login-node quota. ``local_dir=`` writes
   straight to the destination, so peak disk equals the dataset size.

Resumable: a file whose local size already matches the remote size is skipped, so re-running after
a failure only fetches what is missing. Verifies the final file count against the manifest.

Usage:
    python spectro/scripts/hf_download_sc.py --repo tomerraviv95/lwm-spectro-scmax \
        --corpus-dir spectro/outputs/spectro_corpus_scmax_s1 \
        --eval-dir   spectro/outputs/spectro_eval_scmax_indist_s1
"""
from __future__ import annotations

import argparse
import json
import os
import time

# MUST precede the huggingface_hub import.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')


def fetch_sub(api, repo: str, sub: str, target: str, retries: int = 8, force: bool = False) -> bool:
    from huggingface_hub import hf_hub_download

    remote = {f.path: getattr(f, 'size', None)
              for f in api.list_repo_tree(repo, repo_type='dataset',
                                          path_in_repo=sub, recursive=True)}
    files = sorted(p for p in remote if p.endswith('.pt') or p.endswith('manifest.json'))
    if not files:
        print(f"(no files under {sub}/ in {repo} — skipping)")
        return True
    os.makedirs(target, exist_ok=True)

    todo = []
    for p in files:
        loc = os.path.join(target, os.path.basename(p))
        if not force and os.path.isfile(loc) and remote[p] and os.path.getsize(loc) == remote[p]:
            continue
        todo.append(p)
    have = len(files) - len(todo)
    gb = sum(remote[p] or 0 for p in todo) / 1e9
    print(f"[{sub}] {len(files)} files | {have} already local | {len(todo)} to fetch ({gb:.1f} GB)",
          flush=True)

    t0 = time.time()
    for i, p in enumerate(todo, 1):
        loc = os.path.join(target, os.path.basename(p))
        for attempt in range(retries):
            try:
                # local_dir= writes straight to the destination: no cache copy, so peak disk is
                # the dataset size rather than twice it.
                got = hf_hub_download(repo, p, repo_type='dataset', local_dir=target)
                if os.path.abspath(got) != os.path.abspath(loc):
                    os.replace(got, loc)
                if remote[p] and os.path.getsize(loc) != remote[p]:
                    raise IOError(f"size mismatch {os.path.getsize(loc)} != {remote[p]}")
                break
            except Exception as e:
                wait = min(120, 2 ** attempt)
                print(f"  {os.path.basename(p)}: {type(e).__name__}: {str(e)[:100]} "
                      f"(attempt {attempt+1}/{retries}) -> retry in {wait}s", flush=True)
                try:
                    if os.path.isfile(loc):
                        os.remove(loc)          # drop a partial file so the retry is clean
                except OSError:
                    pass
                time.sleep(wait)
        else:
            print(f"[{sub}] FAILED on {p} after {retries} attempts — re-run to resume", flush=True)
            return False
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  [{sub}] {i}/{len(todo)}  {el/60:.1f} min  "
                  f"ETA {el/i*(len(todo)-i)/60:.0f} min", flush=True)

    # nested dirs can appear when local_dir mirrors the repo layout; flatten then clean up
    nested = os.path.join(target, sub)
    if os.path.isdir(nested):
        for f in os.listdir(nested):
            os.replace(os.path.join(nested, f), os.path.join(target, f))
        try:
            os.rmdir(nested)
        except OSError:
            pass

    man_p = os.path.join(target, 'manifest.json')
    if not os.path.isfile(man_p):
        print(f"[{sub}] NO manifest.json — incomplete", flush=True)
        return False
    man = json.load(open(man_p))
    missing = [s for s in man['shards'] if not os.path.isfile(os.path.join(target, s))]
    ok = not missing
    print(f"[{sub}] {'OK' if ok else 'INCOMPLETE'}: n_samples={man.get('n_samples')} "
          f"shards={len(man['shards']) - len(missing)}/{len(man['shards'])}"
          f"{'' if ok else f' — missing {len(missing)}, re-run to resume'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--corpus-dir', default=None)
    ap.add_argument('--eval-dir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    from huggingface_hub import HfApi
    print(f"xet disabled={os.environ.get('HF_HUB_DISABLE_XET')}", flush=True)
    api = HfApi()
    ok = True
    for sub, path in [('corpus', args.corpus_dir), ('eval', args.eval_dir)]:
        if path:
            ok &= fetch_sub(api, args.repo, sub, path, force=args.force)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
