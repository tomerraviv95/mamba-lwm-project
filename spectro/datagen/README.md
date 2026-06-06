# spectro/datagen — synthetic spectrogram generator (placeholder)

An OFDM spectrogram generator built on **NVIDIA Sionna PHY** (v2.x, PyTorch-native, Apache-2.0)
that synthesizes LTE/WiFi/5G spectrograms matching the LWM-Spectro demo contract (128×128
single-channel, labels tech/mod/snr/mobility). It exists so the Mamba MoE can pretrain on a
corpus larger than the 10.5k-sample `demo_data.pt`, **until the authors share their real
generator** (pinged; the real `gen_real.py` + waveform→spectrogram code is not in the HF repo).

Building blocks (`Mapper`, `ResourceGrid`/`OFDMModulator`, `TDL` channel with Doppler, `AWGN`)
come from Sionna; we add per-protocol numerology (`phy_params.py`) and the `torch.stft`
spectrogram step. Install: `pip install sionna>=2.0` (needs PyTorch ≥2.9; runs CPU or GPU).

> ⚠️ Approximate, **not standard-compliant**. Protocols are made structurally distinguishable
> via distinct PHY parameters (subcarrier spacing / FFT / CP), enough for the router to learn
> LTE vs WiFi vs 5G — not for spec conformance. Swap for the real data when available.

See `PLAN.md` for the full design. Intended usage once implemented:

```bash
# generate a scaled corpus (mirrors demo label grid)
python spectro/datagen/generate.py --per-combo 500 --out spectro/outputs/synthetic/
# pretrain the Mamba MoE on the synthetic corpus
python spectro/scripts/spectro_pretrain.py --data synthetic
# evaluate transfer on the REAL demo tasks (baseline embeddings stay valid)
python spectro/scripts/spectro_train_heads.py --arm mamba
python spectro/scripts/spectro_plot_sample_variation.py
```

## Files
`phy_params.py` (numerology + label grid), `sionna_blocks.py` (Sionna chain: Mapper → OFDM →
TDL-Doppler → AWGN, with cached layers), `spectrogram.py` (`torch.stft` → 128×128 dB + norm),
`generate.py` (CLI sweep → sharded `.pt` + `manifest.json`), `visualize.py` (sanity grid),
`check_separability.py` (train the router, report protocol accuracy).

## Validation (done)
- End-to-end Sionna chain runs on CPU; output spectrograms are (1,128,128) float16, per-sample
  normalized — matching the demo contract.
- **Protocol separability: 94.7%** (chance 33%) on a 2,520-sample synthetic set
  (`check_separability.py`) — LTE/WiFi/5G are distinguishable; WiFi perfectly, mild LTE↔5G
  closeness (same FFT, only SCS differs). Confirms the router/MoE works on synthetic data.
- `spectro_pretrain.py --data {synthetic,mixed}` consumes the corpus (synthetic loader in
  `spectro_data.load_synthetic_data`).
- **Throughput**: ~0.24 s/sample on CPU (~600 s for 2,520). A full `--per-combo 500` corpus
  (~157k) is ~10 h single-process — run on GPU, lower `--per-combo`, or shard across processes.

Status: **implemented + validated**. Next: generate a real corpus and pretrain the Mamba MoE on it.
