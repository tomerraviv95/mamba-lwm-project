"""Sanity visualization: a grid of synthetic spectrograms per (tech, modulation).

Generates a few samples on the fly and saves a PNG so we can eyeball that the three protocols
look structurally distinct and that modulation/SNR have visible effects.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phy_params import PROTOCOL_CONFIGS, PROTOCOLS  # noqa: E402
from sionna_blocks import generate_iq  # noqa: E402
from spectrogram import iq_to_spectrogram  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_PLOTS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'plots')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mods', nargs='*', default=['BPSK', 'QPSK', 'QAM16', 'QAM64', 'QAM256'])
    ap.add_argument('--snr', type=int, default=20)
    ap.add_argument('--mobility', default='vehicular')
    ap.add_argument('--out', default=os.path.join(_PLOTS, 'synthetic_spectrogram_grid.png'))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows, cols = len(PROTOCOLS), len(args.mods)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    for r, tech in enumerate(PROTOCOLS):
        for c, mod in enumerate(args.mods):
            iq = generate_iq(PROTOCOL_CONFIGS[tech], mod, snr_db=args.snr, mobility=args.mobility)
            spec = iq_to_spectrogram(iq).float().squeeze(0).numpy()
            ax = axes[r][c] if rows > 1 else axes[c]
            ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(mod, fontsize=10)
            if c == 0:
                ax.set_ylabel(tech, fontsize=11)
    fig.suptitle(f'Synthetic spectrograms (SNR={args.snr}dB, {args.mobility})')
    fig.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
