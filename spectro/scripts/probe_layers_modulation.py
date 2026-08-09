"""Where does modulation live inside each backbone? Per-layer probe, pretrained vs random-init.

The transformer shows ZERO downstream modulation lift over its random-init control while mamba
shows +0.12. Three explanations are distinguishable by this measurement:

  (a) READOUT   -- modulation is present at some depth but destroyed by the final layer (standard
                   for a reconstruction-pretrained encoder: the last layer specializes to the
                   pixel-prediction task). Fix = extract features from an intermediate layer.
  (b) POOLING   -- present per-token but invisible to a mean pool. Probing [mean || std] over
                   tokens separates this from (a); std is the statistic modulation actually lives in.
  (c) ENCODING  -- never present at any depth, in which case pretraining cannot help and the fix
                   is architectural/objective-level, not a readout change.

Run:
  CUDA_VISIBLE_DEVICES=0 python spectro/scripts/probe_layers_modulation.py --n 3000
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectro_backbones import build_expert  # noqa: E402
from spectro_patchify import set_corpus_prenormalized, spectrogram_patchify  # noqa: E402

MODS = ["BPSK", "QPSK", "QAM16", "QAM64", "QAM256"]


def macro_f1(pred, true, n_cls=5):
    out = []
    for c in range(n_cls):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


def probe(F_, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    F_ = np.nan_to_num(F_, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.random.RandomState(seed).permutation(len(F_))
    ntr = int(0.7 * len(F_))
    tr, te = g[:ntr], g[ntr:]
    sc = StandardScaler().fit(F_[tr])
    clf = LogisticRegression(max_iter=1500, C=1.0).fit(sc.transform(F_[tr]), y[tr])
    return macro_f1(clf.predict(sc.transform(F_[te])), y[te])


@torch.no_grad()
def layer_stats(net, tokens, device, batch=64):
    """Run tokens through the backbone, capturing [mean || std] over the token axis per layer."""
    layers = net.layers
    grabbed = {}

    def mk(i):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            grabbed.setdefault(i, []).append(
                torch.cat([o.mean(dim=1), o.std(dim=1)], dim=1).float().cpu())
        return hook

    handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    emb_out = []
    eh = net.embedding.register_forward_hook(
        lambda _m, _i, o: emb_out.append(
            torch.cat([o.mean(dim=1), o.std(dim=1)], dim=1).float().cpu())) \
        if hasattr(net, 'embedding') else None
    try:
        for i in range(0, len(tokens), batch):
            net(tokens[i:i + batch].to(device))
    finally:
        for h in handles:
            h.remove()
        if eh is not None:
            eh.remove()
    res = {}
    if emb_out:
        res[0] = torch.cat(emb_out).numpy()
    for i, chunks in grabbed.items():
        res[i + 1] = torch.cat(chunks).numpy()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--patch', type=int, default=8)
    ap.add_argument('--suffix', default='sc1')
    ap.add_argument('--proto', default='LTE')
    ap.add_argument('--eval-dir', default='spectro/outputs/spectro_eval_sc_indist_s1')
    args = ap.parse_args()
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_per_process_memory_fraction(0.45, 0)

    rows = []
    for f in sorted(glob.glob(os.path.join(args.eval_dir, 'shard_*.pt'))):
        rows += torch.load(f, weights_only=False)
        if len(rows) >= args.n * 4:
            break
    rows = [r for r in rows if str(r['tech']) == args.proto][:args.n]
    print(f"{args.proto}: {len(rows)} samples")
    X = torch.stack([r['data'] for r in rows])
    y = np.array([MODS.index(str(r['mod'])) for r in rows])

    set_corpus_prenormalized(True, 'sc')
    P = spectrogram_patchify(X, patch=args.patch, normalize=True)
    cls = np.full((P.shape[0], 1, P.shape[2]), 0.2, dtype=np.float32)
    tokens = torch.tensor(np.concatenate([cls, P], axis=1), dtype=torch.float32)
    el = tokens.shape[2]
    print(f"tokens {tuple(tokens.shape)}  element_length={el}")
    print(f"raw-token baseline (no backbone): {probe(np.concatenate([P.mean(1), P.std(1)], 1), y):.3f}")
    print()

    for arch in ('mamba', 'transformer'):
        for kind in ('pretrained', 'random'):
            net = build_expert(arch, d_model=128, n_layers=12, element_length=el,
                               max_len=tokens.shape[1])
            if kind == 'pretrained':
                wd = os.path.join('spectro/outputs/pretrained_models',
                                  f'spectro_{arch}_p{args.patch}_{args.suffix}_weights')
                ck = torch.load(os.path.join(wd, f'{args.proto}_expert.pth'), map_location='cpu')
                sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
                net.load_state_dict(sd)
            net = net.to(dev).eval()
            inner = net.net if hasattr(net, 'net') else net
            try:
                stats = layer_stats(inner, tokens, dev)
            except Exception as e:
                print(f"{arch}/{kind}: hook failed ({type(e).__name__}: {e})")
                continue
            scores = {L: probe(F_, y) for L, F_ in sorted(stats.items())}
            best = max(scores, key=scores.get)
            print(f"{arch:12s} {kind:11s} " +
                  " ".join(f"L{L}:{s:.3f}" for L, s in sorted(scores.items())) +
                  f"   | BEST L{best}={scores[best]:.3f}")
            del net, stats
            torch.cuda.empty_cache()
        print()
    print("chance = 0.200")


if __name__ == '__main__':
    main()
