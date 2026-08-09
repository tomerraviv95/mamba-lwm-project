"""Acceptance test: is a single-carrier spectrogram config USEFUL ENOUGH for the tasks?

This is the gate that the OFDM corpus would have failed. It answers, on real DeepMIMO channels:

  * modulation 5-class  -- overall and **per SNR bucket**, plus the BPSK-vs-QPSK confusion that
    the OFDM/magnitude pipeline could never resolve;
  * joint SNR/Doppler 21-class, and its two factors separately;
  * whether SNR is solvable by the guard-band brightness shortcut (must stay near 0 sigma).

Unlike ``sweep_sc_config.py`` (which scans configs cheaply) this trains the CNN probe long enough
to be a fair estimate of what a real from-scratch baseline achieves, so the numbers are directly
comparable to the DeepCNN arm of the study.

Run:  CUDA_VISIBLE_DEVICES=0 python spectro/datagen/validate_sc_dataset.py --n 4000 --win 8
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace  # noqa: E402

from phy_params import MODULATIONS, SC_CONFIGS, SNRS_DB  # noqa: E402
from sweep_sc_config import (MOBILITIES3, cnn_probe, guard_shortcut_sigma,  # noqa: E402
                             load_pdp, macro_f1, moment_probe, renorm, synth_batch, TinyCNN)


def cnn_predict(X, y, n_cls, dev, epochs, bs=64, seed=0, lr=2e-3):
    """Same probe as cnn_probe but returns final predictions + the test index for breakdowns."""
    import torch.nn.functional as F
    torch.manual_seed(seed)
    n = X.shape[0]
    g = np.random.RandomState(seed).permutation(n)
    ntr = int(0.7 * n)
    tr, te = g[:ntr], g[ntr:]
    Xtr, ytr = X[tr].to(dev).float(), torch.as_tensor(y[tr], device=dev)
    Xte, yte = X[te].to(dev).float(), torch.as_tensor(y[te], device=dev)
    m = TinyCNN(n_cls, X.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * max(1, ntr // bs))
    best_f1, best_pred = 0.0, None
    for _ in range(epochs):
        m.train()
        idx = torch.randperm(ntr, device=dev)
        for i in range(0, ntr - bs + 1, bs):
            j = idx[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(m(Xtr[j]), ytr[j]).backward()
            opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            pr = torch.cat([m(Xte[i:i + 128]).argmax(1) for i in range(0, len(te), 128)]).cpu().numpy()
        f1 = macro_f1(pr, yte.cpu().numpy(), n_cls)
        if f1 > best_f1:
            best_f1, best_pred = f1, pr
    del Xtr, Xte, m
    torch.cuda.empty_cache()
    return best_f1, best_pred, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--tech', default='LTE')
    ap.add_argument('--win', type=int, default=8)
    ap.add_argument('--num-symbols', type=int, default=131072)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--norm', default='global')
    ap.add_argument('--sps', type=int, default=None)
    ap.add_argument('--rolloff', type=float, default=None)
    ap.add_argument('--mem-frac', type=float, default=0.45)
    args = ap.parse_args()

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    rng = np.random.RandomState(1)
    _ov = {'num_symbols': args.num_symbols}
    if args.sps: _ov['sps'] = args.sps
    if args.rolloff is not None: _ov['rolloff'] = args.rolloff
    cfg = replace(SC_CONFIGS[args.tech], **_ov)
    print(f"{args.tech}: sr={cfg.sample_rate/1e6:.2f}MHz sps={cfg.sps} beta={cfg.rolloff} "
          f"burst={cfg.duration_s*1e3:.1f}ms occ={cfg.occupied_frac:.0%} win={args.win} "
          f"({args.win/cfg.sps:.0f} sym) norm={args.norm} n={args.n}")

    n = args.n
    mods = rng.choice(MODULATIONS, n)
    snrs = rng.choice(SNRS_DB, n)
    mobs = rng.choice(MOBILITIES3, n)
    pdp = load_pdp(n, rng)

    chunks = []
    for s in range(0, n, args.batch):
        sl = slice(s, min(s + args.batch, n))
        chunks.append(synth_batch(args.tech, list(mods[sl]), list(snrs[sl]), list(mobs[sl]),
                                  {k: v[sl] for k, v in pdp.items()}, cfg, rng, dev,
                                  args.win).cpu())
    X = renorm(torch.cat(chunks), args.norm)
    del chunks

    y_mod = np.array([MODULATIONS.index(m) for m in mods])
    y_sd = np.array([SNRS_DB.index(s) * 3 + MOBILITIES3.index(m) for s, m in zip(snrs, mobs)])
    y_snr = np.array([SNRS_DB.index(s) for s in snrs])
    y_mob = np.array([MOBILITIES3.index(m) for m in mobs])

    print("\n--- task separability (CNN = trained-from-scratch probe, mom = moment probe) ---")
    f_mod, pred_mod, te = cnn_predict(X, y_mod, 5, dev, args.epochs)
    print(f"modulation  5-cls : cnn {f_mod:.3f} | mom {moment_probe(X, y_mod, 5):.3f}  (chance 0.200)")
    f_sd, _, _ = cnn_predict(X, y_sd, 21, dev, args.epochs)
    print(f"snr_doppler 21-cls: cnn {f_sd:.3f} | mom {moment_probe(X, y_sd, 21):.3f}  (chance 0.048)")
    f_snr, _, _ = cnn_predict(X, y_snr, 7, dev, args.epochs)
    print(f"snr          7-cls: cnn {f_snr:.3f} | mom {moment_probe(X, y_snr, 7):.3f}  (chance 0.143)")
    f_mob, _, _ = cnn_predict(X, y_mob, 3, dev, args.epochs)
    print(f"mobility     3-cls: cnn {f_mob:.3f} | mom {moment_probe(X, y_mob, 3):.3f}  (chance 0.333)")

    # --- modulation per SNR bucket -------------------------------------------------------
    print("\n--- modulation macro-F1 by SNR (test split) ---")
    ytrue = y_mod[te]
    snr_te = snrs[te]
    for lo, hi, name in [(-5, 0, 'low  (-5..0dB)'), (5, 10, 'mid  ( 5..10dB)'),
                         (15, 25, 'high (15..25dB)')]:
        m = (snr_te >= lo) & (snr_te <= hi)
        if m.sum() > 20:
            print(f"  {name}: {macro_f1(pred_mod[m], ytrue[m], 5):.3f}  (n={m.sum()})")

    print("\n--- modulation confusion (rows=true, cols=pred), all SNR ---")
    cm = np.zeros((5, 5), int)
    for t, p_ in zip(ytrue, pred_mod):
        cm[t, p_] += 1
    print("        " + " ".join(f"{m:>7s}" for m in MODULATIONS))
    for i, m in enumerate(MODULATIONS):
        print(f"{m:>7s} " + " ".join(f"{v:7d}" for v in cm[i]))

    # --- BPSK vs QPSK: the pair OFDM/magnitude could never separate -----------------------
    hi = snr_te >= 15
    bq = hi & np.isin(ytrue, [0, 1])
    if bq.sum() > 20:
        acc = (pred_mod[bq] == ytrue[bq]).mean()
        conf = np.zeros((2, 2), int)
        for t, p in zip(ytrue[bq], pred_mod[bq]):
            if p in (0, 1):
                conf[t, p] += 1
        print(f"\nBPSK/QPSK @SNR>=15dB: acc {acc:.3f}  confusion[true,pred] "
              f"BPSK->[{conf[0,0]},{conf[0,1]}] QPSK->[{conf[1,0]},{conf[1,1]}]")

    print(f"\nguard-band SNR shortcut: {guard_shortcut_sigma(X, snrs):.2f} sigma "
          f"(OFDM corpus measured 27-60; want < ~2)")


if __name__ == '__main__':
    main()
