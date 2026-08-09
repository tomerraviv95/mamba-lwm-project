"""Measure whether a single-carrier spectrogram config is INFORMATIVE for the downstream tasks.

Purpose: before regenerating a 132k corpus (multi-hour) we must know that the stored 128x128
representation actually carries (a) modulation order and (b) joint SNR/Doppler. The OFDM corpus
failed exactly this test after the fact -- an approximate-Bayes probe on it measured a modulation
ceiling of macro-F1 0.555 (all-SNR), and 7-way SNR turned out to be a trivial guard-band
brightness ratio separable at 27-60 sigma.

What this does, per candidate config:
  1. Synthesize single-carrier bursts (eq. 4) through a DeepMIMO-parameterized multipath channel
     with per-ray Doppler + AWGN -- the same physics as the real generator, but with a light
     explicit tap-convolution instead of Sionna's ApplyTimeChannel (which materializes a
     (B,T,l_tot) tensor and will not fit alongside a 50%-capped GPU).
  2. Build the 128x128 power spectrogram.
  3. Score separability two ways:
       - a cheap moment probe (logistic regression on hand-built envelope/spectral statistics) =
         roughly "how much is trivially there";
       - a small from-scratch CNN = "how much a real model can get", the honest number.
  4. Report the guard-band shortcut strength for SNR, so we do not re-introduce it.

Run:  CUDA_VISIBLE_DEVICES=0 python spectro/datagen/sweep_sc_config.py --n 1800
"""
from __future__ import annotations

import argparse

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace  # noqa: E402
from phy_params import (CARRIER_FREQUENCY_HZ, MOBILITY_SPEED_RANGE, MOD_BITS,  # noqa: E402
                        MODULATIONS, SC_CONFIGS, SNRS_DB)
from pulse_shaping import apply_freq_offset, matched_filter, pulse_shape  # noqa: E402
from spectrogram import sc_power_spectrogram  # noqa: E402

C_LIGHT = 3e8
MOBILITIES3 = ["static", "pedestrian", "vehicular"]


# --------------------------------------------------------------------------------------------
# light physics
# --------------------------------------------------------------------------------------------
def multipath_apply(x, delays_s, powers, phases, aoas, speeds, sample_rate, fc=CARRIER_FREQUENCY_HZ,
                    n_subrays=12, angle_spread_rad=0.52, rng=None):
    """y[n] = sum_l h_l[n] x[n - d_l], with each tap faded per the 3GPP Doppler spectrum (eq. 5-6).

    Each ray-traced path becomes a TAP whose complex gain is a sum of ``n_subrays`` sub-rays whose
    arrival angles are spread around the path's AoA:

        alpha_l[n] = (1/sqrt(M)) sum_m exp( j*2*pi*f_d*cos(aoa_l + delta_m)*n/fs + j*phi_lm )

    The angular spread is what makes this work. With a single deterministic Doppler shift per path
    (the previous behaviour), |alpha_l[n]| is CONSTANT -- Doppler becomes a pure phase rotation,
    which a magnitude spectrogram throws away, so a LOS-dominated user shows no mobility signature
    at all. Giving the sub-rays different Doppler shifts makes the tap AMPLITUDE fade with
    coherence time ~1/spread, which is the "Doppler-induced temporal fluctuation" the paper's
    Fig. 2 shows and the only thing a |.|^2 representation can see.

    delays/powers/phases/aoas: (B, K) numpy. speeds: (B,) m/s.
    """
    dev = x.device
    B, T = x.shape
    rng = rng or np.random
    t = torch.arange(T, device=dev, dtype=torch.float32) / sample_rate
    d = torch.as_tensor(delays_s, dtype=torch.float32, device=dev)
    p = torch.as_tensor(powers, dtype=torch.float32, device=dev)
    ph = torch.as_tensor(phases, dtype=torch.float32, device=dev)
    ao = torch.as_tensor(aoas, dtype=torch.float32, device=dev)
    sp = torch.as_tensor(speeds, dtype=torch.float32, device=dev).reshape(-1, 1)

    p = p / torch.clamp(p.sum(dim=1, keepdim=True), min=1e-20)      # normalize total power
    lag = torch.round(d * sample_rate).long().clamp_(0, T - 1)       # delay -> integer taps
    g = torch.sqrt(torch.clamp(p, min=0.0)) * torch.exp(1j * ph.to(torch.complex64))
    v_over_c = (sp / C_LIGHT) * fc                                   # (B,1) max Doppler [Hz]

    arange = torch.arange(T, device=dev)
    y = torch.zeros_like(x)
    K = d.shape[1]
    for k in range(K):
        gk = g[:, k:k + 1]
        if torch.all(gk.abs() == 0):
            continue
        # temporally-correlated fading for this tap: sum of sub-rays about the path AoA
        alpha = torch.zeros(B, T, dtype=torch.complex64, device=dev)
        dtheta = torch.as_tensor(
            rng.uniform(-angle_spread_rad, angle_spread_rad, (B, n_subrays)),
            dtype=torch.float32, device=dev)
        phi = torch.as_tensor(rng.uniform(0, 2 * np.pi, (B, n_subrays)),
                              dtype=torch.float32, device=dev)
        for m in range(n_subrays):
            fdm = v_over_c * torch.cos(ao[:, k:k + 1] + dtheta[:, m:m + 1])
            arg = 2 * np.pi * fdm * t[None, :] + phi[:, m:m + 1]
            alpha = alpha + torch.exp(1j * arg.to(torch.complex64))
        alpha = alpha / np.sqrt(n_subrays)

        idx = (arange[None, :] - lag[:, k:k + 1]).clamp_(min=0)
        xs = torch.gather(x, 1, idx)
        xs = xs * (arange[None, :] >= lag[:, k:k + 1])
        y = y + gk * alpha * xs
    return y


def synth_batch(tech, mods, snrs, mobs, pdp, cfg, rng, dev, win_length, freq_jitter=True,
                mf=True):
    """One batch -> (b,1,128,128) float16 spectrograms."""
    from sionna.phy.mapping import BinarySource, Mapper
    b = len(mods)
    sr = cfg.sample_rate
    specs = []
    src = BinarySource().to(dev)
    # group by modulation so the Mapper is shared
    order = np.argsort(mods)
    out = [None] * b
    for mod in sorted(set(mods)):
        sel = [i for i in range(b) if mods[i] == mod]
        bits_n = MOD_BITS[mod]
        ctype = "pam" if mod == "BPSK" else "qam"
        mapper = Mapper(ctype, bits_n).to(dev)
        nb = len(sel)
        bits = src([nb, 1, 1, cfg.num_symbols * bits_n])
        s = mapper(bits).reshape(nb, -1)
        x = pulse_shape(s, cfg.sps, cfg.rolloff, cfg.span_symbols)
        if freq_jitter:                       # random placement in-band -> no fixed guard region
            room = max(0.0, (1.0 - cfg.occupied_frac)) / 2.0
            f = torch.as_tensor(rng.uniform(-room, room, nb), device=dev)
            x = apply_freq_offset(x, f)
        speeds = np.array([rng.uniform(*MOBILITY_SPEED_RANGE[mobs[i]]) for i in sel], np.float32)
        y = multipath_apply(x, pdp['delay'][sel], pdp['power_linear'][sel], pdp['phase'][sel],
                            pdp["aoa_az"][sel], speeds, sr, rng=rng)
        # AWGN at each sample's SNR (measured on the received signal)
        pw = y.abs().pow(2).mean(dim=1, keepdim=True)
        snr_lin = torch.as_tensor([10.0 ** (snrs[i] / 10.0) for i in sel],
                                  device=dev, dtype=torch.float32).reshape(-1, 1)
        no = pw / snr_lin
        y = y + torch.sqrt(no / 2) * torch.complex(torch.randn_like(y.real), torch.randn_like(y.real))
        if mf:                                 # receiver matched filter (suppresses OOB noise)
            y = matched_filter(y, cfg.sps, cfg.rolloff, cfg.span_symbols)
        sp = sc_power_spectrogram(y, n_fft=128, win_length=win_length, out_size=128, norm='none')
        for j, i in enumerate(sel):
            out[i] = sp[j]
    return torch.stack(out)


def renorm(X_db: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a normalization mode to raw-dB spectrograms (so one generation serves both modes)."""
    x = X_db.float()
    if mode == 'global':
        return ((x - x.mean()) / torch.clamp(x.std(), min=1e-6)).half()
    if mode == 'sample':
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        s = torch.clamp(x.std(dim=(1, 2, 3), keepdim=True), min=1e-6)
        return ((x - m) / s).half()
    return x.half()


# --------------------------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------------------------
class TinyCNN(nn.Module):
    """Small from-scratch CNN -- the 'can a real model get it' probe."""
    def __init__(self, n_cls, ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.net(x)


def moment_features(X: torch.Tensor) -> np.ndarray:
    """Hand-built sufficient-ish statistics of a dB spectrogram -> (n, F).

    This is the 'is the information PRESENT' instrument. A CNN ending in global average pooling
    computes a MEAN of features, but the modulation cue is an envelope VARIANCE and the mobility
    cue is a temporal AUTOCORRELATION -- neither is a mean, so a weak CNN probe can read ~chance
    on data that is actually separable. Measuring both tells us whether a disappointing number
    means "no signal in the data" or "the readout cannot form the statistic".
    """
    x = X.float().numpy()[:, 0]                                  # (n, K, T) freq x time
    n = x.shape[0]
    fr_pow = x.mean(axis=1)                                      # (n,T) per-frame mean dB
    fq_pow = x.mean(axis=2)                                      # (n,K) per-bin mean dB
    xc = x - x.mean((1, 2), keepdims=True)
    f = [x.mean((1, 2)), x.std((1, 2)),
         (xc ** 3).mean((1, 2)), (xc ** 4).mean((1, 2)),
         np.percentile(x, 5, axis=(1, 2)), np.percentile(x, 95, axis=(1, 2)),
         fr_pow.std(axis=1), fq_pow.std(axis=1),
         fq_pow.max(axis=1) - fq_pow.min(axis=1)]
    # WITHIN-frame vs ACROSS-frame decomposition. The modulation cue is the instantaneous
    # envelope variance, which lives WITHIN a frame; slow fading lives ACROSS frames and (for a
    # near-flat channel, delay spread 10ns << 130ns symbol period here) is almost constant across
    # frequency. Separating the two stops fading variance from masking the constellation cue.
    within = x.std(axis=1)                                       # (n,T) per-frame spectral std
    f += [within.mean(axis=1), within.std(axis=1),
          np.percentile(within, 90, axis=1) - np.percentile(within, 10, axis=1)]
    xw = x - x.mean(axis=1, keepdims=True)                       # de-mean each frame
    f += [(xw ** 4).mean((1, 2)) / (np.clip((xw ** 2).mean((1, 2)), 1e-9, None) ** 2),  # kurtosis
          np.abs(xw).mean((1, 2))]
    xt = x - x.mean(axis=2, keepdims=True)                       # de-mean each frequency bin
    f += [(xt ** 2).mean((1, 2)), (xt ** 4).mean((1, 2)) /
          (np.clip((xt ** 2).mean((1, 2)), 1e-9, None) ** 2)]
    # temporal autocorrelation of the frame-power series -> Doppler rate
    z = (fr_pow - fr_pow.mean(1, keepdims=True)) / (fr_pow.std(1, keepdims=True) + 1e-9)
    for lag in (1, 2, 4, 8, 16, 32):
        f.append((z[:, :-lag] * z[:, lag:]).mean(axis=1))
    # spectral occupancy: how much of the band is above the noise floor
    lo = np.percentile(x, 10, axis=(1, 2), keepdims=True)
    hi = np.percentile(x, 90, axis=(1, 2), keepdims=True)
    f.append(((fq_pow > (lo[:, 0] + 0.5 * (hi[:, 0] - lo[:, 0]))).mean(axis=1)))
    return np.stack(f, axis=1)


def moment_probe(X, y, n_cls, seed=0):
    """Logistic regression on moment_features -> macro-F1. Cheap, CPU, no GPU footprint."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    F_ = moment_features(X)
    F_ = np.nan_to_num(F_, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.random.RandomState(seed).permutation(len(F_))
    ntr = int(0.7 * len(F_))
    tr, te = g[:ntr], g[ntr:]
    sc = StandardScaler().fit(F_[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(F_[tr]), y[tr])
    return macro_f1(clf.predict(sc.transform(F_[te])), y[te], n_cls)


def macro_f1(pred, true, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def cnn_probe(X, y, n_cls, dev, epochs=18, bs=64, seed=0):
    torch.manual_seed(seed)
    n = X.shape[0]
    g = np.random.RandomState(seed).permutation(n)
    ntr = int(0.7 * n)
    tr, te = g[:ntr], g[ntr:]
    Xtr = X[tr].to(dev).float(); ytr = torch.as_tensor(y[tr], device=dev)
    Xte = X[te].to(dev).float(); yte = torch.as_tensor(y[te], device=dev)
    m = TinyCNN(n_cls, X.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, epochs * max(1, ntr // bs))
    best = 0.0
    for ep in range(epochs):
        m.train()
        idx = torch.randperm(ntr, device=dev)
        for i in range(0, ntr - bs + 1, bs):
            j = idx[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(m(Xtr[j]), ytr[j])
            loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            pr = torch.cat([m(Xte[i:i + 128]).argmax(1) for i in range(0, len(te), 128)])
        best = max(best, macro_f1(pr.cpu().numpy(), yte.cpu().numpy(), n_cls))
    del Xtr, Xte, m
    torch.cuda.empty_cache()
    return best


def guard_shortcut_sigma(X, snr_lab):
    """How separable are the 7 SNR classes from a single edge/centre brightness ratio?

    Reproduces the statistic that made the OFDM corpus degenerate. Returns the mean
    adjacent-class separation in sigma. Low = no trivial shortcut.
    """
    x = X.float().numpy()[:, 0]                     # (n,128,128) freq x time
    edge = np.concatenate([x[:, :16, :], x[:, -16:, :]], axis=1).mean(axis=(1, 2))
    ctr = x[:, 48:80, :].mean(axis=(1, 2))
    r = ctr - edge
    seps = []
    for a, b in zip(SNRS_DB[:-1], SNRS_DB[1:]):
        ra, rb = r[snr_lab == a], r[snr_lab == b]
        if len(ra) < 3 or len(rb) < 3:
            continue
        s = np.sqrt(0.5 * (ra.var() + rb.var())) + 1e-9
        seps.append(abs(rb.mean() - ra.mean()) / s)
    return float(np.mean(seps)) if seps else float('nan')


# --------------------------------------------------------------------------------------------
def load_pdp(n, rng, city="city_0_newyork_3p5_lwm", max_paths=10):
    from deepmimo_channel import extract_city_pdp
    p = extract_city_pdp(city, bs_idx=1, max_paths=max_paths)
    u = p['delay'].shape[0]
    idx = rng.permutation(u)[:n]
    return {k: p[k][idx] for k in ('delay', 'power_linear', 'phase', 'aoa_az')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1800)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--tech', default='LTE')
    ap.add_argument('--mem-frac', type=float, default=0.45)
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--num-symbols', type=int, default=None,
                    help='override the burst length (symbols). Longer burst = more Doppler cycles, '
                         'which is what makes the 3-way mobility label separable.')
    ap.add_argument('--windows', type=int, nargs='+', default=[16, 32, 64, 128])
    ap.add_argument('--norms', nargs='+', default=['global', 'sample'])
    args = ap.parse_args()

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    rng = np.random.RandomState(0)
    cfg = SC_CONFIGS[args.tech]
    if args.num_symbols:
        cfg = replace(cfg, num_symbols=args.num_symbols)
    print(f"tech={args.tech} sr={cfg.sample_rate/1e6:.2f}MHz sps={cfg.sps} beta={cfg.rolloff} "
          f"N_s={cfg.num_symbols} -> {cfg.num_samples} samples = {cfg.duration_s*1e3:.2f} ms, "
          f"occupied={cfg.occupied_frac:.0%}")

    n = args.n
    mods = rng.choice(MODULATIONS, n)
    snrs = rng.choice(SNRS_DB, n)
    mobs = rng.choice(MOBILITIES3, n)
    pdp = load_pdp(n, rng)

    # Doppler budget: cycles of phase rotation across the burst per mobility class. Below ~0.3
    # cycles a class is indistinguishable from static, which is what pinned mobility at chance.
    print("  Doppler cycles across burst: " + " | ".join(
        f"{m} {MOBILITY_SPEED_RANGE[m][0]/C_LIGHT*CARRIER_FREQUENCY_HZ*cfg.duration_s:.2f}"
        f"-{MOBILITY_SPEED_RANGE[m][1]/C_LIGHT*CARRIER_FREQUENCY_HZ*cfg.duration_s:.2f}"
        for m in MOBILITIES3))

    # candidate STFT windows: window length in SAMPLES; /sps = symbols per analysis window
    candidates = args.windows
    results = {}
    for win in candidates:
        t0 = time.time()
        chunks = []
        for s in range(0, n, args.batch):
            sl = slice(s, min(s + args.batch, n))
            sub = {k: v[sl] for k, v in pdp.items()}
            chunks.append(synth_batch(args.tech, list(mods[sl]), list(snrs[sl]), list(mobs[sl]),
                                      sub, cfg, rng, dev, win).cpu())
        X_db = torch.cat(chunks)
        gen_s = time.time() - t0

        y_mod = np.array([MODULATIONS.index(m) for m in mods])
        y_sd = np.array([SNRS_DB.index(s) * 3 + MOBILITIES3.index(m) for s, m in zip(snrs, mobs)])
        y_mob = np.array([MOBILITIES3.index(m) for m in mobs])

        for nm in args.norms:
            X = renorm(X_db, nm)
            m_mod = moment_probe(X, y_mod, len(MODULATIONS))
            m_sd = moment_probe(X, y_sd, 21)
            m_mob = moment_probe(X, y_mob, 3)
            f_mod = cnn_probe(X, y_mod, len(MODULATIONS), dev, epochs=args.epochs)
            f_sd = cnn_probe(X, y_sd, 21, dev, epochs=args.epochs)
            f_mob = cnn_probe(X, y_mob, 3, dev, epochs=args.epochs)
            sig = guard_shortcut_sigma(X, snrs)
            results[(win, nm)] = (f_mod, f_sd, f_mob, sig)
            print(f"  win={win:4d} ({win/cfg.sps:.0f}sym) norm={nm:6s} | mod cnn {f_mod:.3f} / mom "
                  f"{m_mod:.3f} | snr_dop cnn {f_sd:.3f} / mom {m_sd:.3f} | mob cnn {f_mob:.3f} / "
                  f"mom {m_mob:.3f} | guard {sig:.2f}")
            del X
            torch.cuda.empty_cache()
        del X_db
        torch.cuda.empty_cache()

    print("\nchance: mod 0.200 | snr_dop 0.048 | mob 0.333")
    best = max(results, key=lambda k: results[k][0] + results[k][1])
    print(f"best (win, norm) by (mod + snr_dop): {best}")


if __name__ == '__main__':
    main()
