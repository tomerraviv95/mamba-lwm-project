"""Parse a pretraining Slurm log (cluster/logs/pretrain-*.out) into loss curves.

No wandb needed — reads the stdout we already print and plots per-expert train/val MSE plus the
router val-accuracy. Use it to inspect a finished run.

Usage::

    python spectro/scripts/plot_pretrain_curves.py cluster/logs/pretrain-18020858.out
    python spectro/scripts/plot_pretrain_curves.py cluster/logs/pretrain-*.out --out spectro/outputs/plots/pretrain_curves.png
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

_EXPERT = re.compile(r"\[Expert (\S+)\]\s+(\d+) training")
_EPOCH = re.compile(r"epoch (\d+)/\d+\s+train_mse=([\d.]+)\s+val_mse=([\d.]+)")
_ROUTER = re.compile(r"router epoch (\d+)/\d+\s+val_acc=([\d.]+)")


def parse(path: str):
    experts = defaultdict(lambda: {"epoch": [], "train": [], "val": []})
    router = {"epoch": [], "acc": []}
    cur = None
    in_router = False
    with open(path) as f:
        for line in f:
            m = _EXPERT.search(line)
            if m:
                cur, in_router = m.group(1), False
                continue
            if "[Router]" in line:
                in_router = True
                continue
            mr = _ROUTER.search(line)
            if mr and in_router:
                router["epoch"].append(int(mr.group(1)))
                router["acc"].append(float(mr.group(2)))
                continue
            me = _EPOCH.search(line)
            if me and cur and not in_router:
                experts[cur]["epoch"].append(int(me.group(1)))
                experts[cur]["train"].append(float(me.group(2)))
                experts[cur]["val"].append(float(me.group(3)))
    return experts, router


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to a pretrain-*.out log")
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "spectro/outputs/plots/pretrain_curves.png"))
    args = ap.parse_args()

    experts, router = parse(args.log)
    if not experts:
        raise SystemExit(f"No expert epoch lines found in {args.log}")

    has_router = bool(router["epoch"])
    n = len(experts) + (1 if has_router else 0)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (name, d) in zip(axes, experts.items()):
        ax.plot(d["epoch"], d["train"], "-o", ms=3, label="train")
        ax.plot(d["epoch"], d["val"], "-s", ms=3, label="val")
        best = min(d["val"])
        ax.axhline(best, ls=":", c="gray", lw=1)
        ax.set_title(f"{name} expert (best val={best:.0f})")
        ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE"); ax.grid(alpha=0.3); ax.legend()

    if has_router:
        ax = axes[-1]
        ax.plot(router["epoch"], router["acc"], "-o", ms=3, c="#2ca02c")
        ax.set_title(f"router (best acc={max(router['acc']):.3f})")
        ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy"); ax.set_ylim(0, 1); ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.suptitle(f"Pretraining curves — {os.path.basename(args.log)}")
    fig.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
