"""Publish downstream sweep results (submission dirs) to Hugging Face + Weights & Biases.

Reads one or more ``submission_*/aggregated_results.json`` produced by ``spectro_train_heads.py`` /
``spectro_finetune.py`` and:
  - pushes each submission folder to an HF *dataset* repo under ``results/<basename>/`` (so all arms'
    curves live in one browsable repo), and
  - logs the per-(task, x-point) macro-F1 + accuracy to a W&B run named after the submission.

Both sinks are independent and fault-tolerant: a failure in one (e.g. no network on a compute node)
is reported but does NOT abort the other or the job. HF uploads force the classic HTTP path
(``HF_HUB_DISABLE_XET=1``) which is the one that works on the restricted cluster network.

Example::
    HF_TOKEN=... python spectro/scripts/upload_results.py \
        --submissions spectro/outputs/submissions/submission_spectro_mamba_p4_heldout_alluser \
        --hf-repo tomerraviv95/lwm-spectro-results --wandb-project lwm-spectro
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # classic HTTP upload works on the cluster; xet doesn't


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _push_hf(subs, repo, private):
    try:
        from huggingface_hub import HfApi
    except Exception as e:                                    # pragma: no cover
        print(f"[hf] huggingface_hub unavailable: {e}", file=sys.stderr); return
    try:
        api = HfApi(token=_token())
        api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)
        for d in subs:
            base = os.path.basename(d.rstrip("/"))
            print(f"[hf] uploading {d} -> dataset:{repo}/results/{base}/ ...", flush=True)
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=d,
                              path_in_repo=f"results/{base}", commit_message=f"results: {base}")
        print(f"[hf] done -> https://huggingface.co/datasets/{repo}")
    except Exception as e:
        print(f"[hf] UPLOAD FAILED ({e}); re-run on the login node:\n"
              f"     HF_HUB_DISABLE_XET=1 HF_TOKEN=... uv run python spectro/scripts/upload_results.py "
              f"--submissions {' '.join(subs)} --hf-repo {repo}{' --private' if private else ''}",
              file=sys.stderr)


def _log_wandb(subs, project):
    try:
        import wandb
    except Exception as e:                                    # pragma: no cover
        print(f"[wandb] not available: {e}", file=sys.stderr); return
    for d in subs:
        jp = os.path.join(d, "aggregated_results.json")
        if not os.path.isfile(jp):
            print(f"[wandb] no aggregated_results.json in {d}, skip", file=sys.stderr); continue
        agg = json.load(open(jp))
        base = os.path.basename(d.rstrip("/"))
        cfg = agg.get("experiment_config", {})
        try:
            run = wandb.init(project=project, name=base, group=cfg.get("arm", base),
                             config=cfg, reinit=True)
        except Exception as e:
            print(f"[wandb] init failed ({e}); set WANDB_MODE=offline and `wandb sync` later.",
                  file=sys.stderr); return
        for tkey, tblock in agg.get("results_by_task", {}).items():
            task = tblock.get("name", tkey)
            # order x-points by #training samples so the W&B step axis is monotonic
            items = sorted(tblock.get("results", {}).items(),
                           key=lambda kv: kv[1].get("n_samples") or 0)
            for _, r in items:
                n = r.get("n_samples") or 0
                log = {f"{task}/macro_f1": r.get("score"), f"{task}/accuracy": r.get("accuracy")}
                if r.get("per_class") is not None:
                    log[f"{task}/per_class"] = r["per_class"]
                wandb.log({k: v for k, v in log.items() if v is not None}, step=int(n))
        run.finish()
        print(f"[wandb] logged {base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", nargs="+", required=True, help="submission_* result dirs")
    ap.add_argument("--hf-repo", default=None, help="HF dataset repo id for results (skip if unset)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--wandb-project", default=None, help="W&B project (skip if unset)")
    args = ap.parse_args()

    subs = [d for d in args.submissions if os.path.isdir(d)]
    missing = [d for d in args.submissions if d not in subs]
    if missing:
        print(f"[warn] missing submission dirs skipped: {missing}", file=sys.stderr)
    if not subs:
        sys.exit("no valid submission dirs given")

    if args.hf_repo:
        _push_hf(subs, args.hf_repo, args.private)
    if args.wandb_project:
        _log_wandb(subs, args.wandb_project)
    if not args.hf_repo and not args.wandb_project:
        print("(nothing to do: pass --hf-repo and/or --wandb-project)")


if __name__ == "__main__":
    main()
