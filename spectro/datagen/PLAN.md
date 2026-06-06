# Synthetic spectrogram data generation — PLAN

## Context / why

LWM-Spectro's real pretraining corpus is not public, and the generator (`gen_real.py` +
the upstream waveform→spectrogram step) is **not in the HF repo** — only `demo_data.pt`
(10,500 samples) is available. We pinged the authors; until they answer, this folder builds a
**self-contained synthetic generator** so the Mamba MoE can be pretrained on a larger corpus
than the 10.5k demo set.

Decisions (confirmed): build the DSP blocks on **NVIDIA Sionna PHY** (v2.x, PyTorch-native,
Apache-2.0, CPU+GPU — verified maintained April 2026); **mirror the demo label grid, scaled**
(configurable samples-per-combo). Sionna provides the battle-tested primitives (constellation
`Mapper`, `ResourceGrid`+`OFDMModulator`, 3GPP `TDL` channel with Doppler, `AWGN`); we add only
the per-protocol numerology configs + the spectrogram step.

This is an **approximate placeholder**: Sionna is 5G-NR/OFDM-centric, so LTE and WiFi are
**approximated via OFDM numerology** (subcarrier spacing / FFT / CP), making the three techs
structurally distinct enough for the router to learn — but it is not standard-compliant. It will
be swapped for the authors' real data if/when they share it.

**Dependency:** add `sionna>=2.0` (pulls into the existing `.venv`; needs PyTorch ≥2.9, we have
2.10). Sionna PHY runs CPU-only fine (LLVM is only needed for Sionna RT, which we don't use).

## Output contract (must match `spectro/scripts/spectro_data.py`)

Each sample is a dict compatible with the demo format:
```
{ 'tech': 'LTE'|'WiFi'|'5G', 'snr': 'SNR-5dB'..'SNR25dB',
  'mod': 'BPSK'|'QPSK'|'QAM16'|'QAM64'|'QAM256', 'mob': 'static'|'pedestrian'|'vehicular',
  'data': float16 tensor (1,128,128) }      # per-sample-normalized dB spectrogram
```
No `moe_embedding`/`tech_embedding` (those come from the HF model). Therefore synthetic data is
used for **Mamba pretraining** (and optionally raw/mamba downstream), while the
apples-to-apples downstream sweep still uses the real `demo_data.pt` (which carries the
baseline embeddings). See "Integration" below.

## Label grid (mirror demo, scaled)
- tech ∈ {LTE, WiFi, 5G}; mod ∈ {BPSK, QPSK, QAM16, QAM64, QAM256};
  snr ∈ {-5,0,5,10,15,20,25} dB; mob ∈ {static, pedestrian, vehicular}.
- `--per-combo` samples for each (tech, mod, snr, mob) combination (default 500 ⇒ ~157k
  samples; set lower for quick runs). Deterministic via `--seed`.

## Protocol PHY parameterization (makes techs structurally distinct)

| Tech | SCS | FFT | Used SC | CP | Footprint intuition |
|------|-----|-----|---------|----|--------------------|
| WiFi (802.11a/g) | 312.5 kHz | 64 | 52 | 16 (0.8µs) | short symbols, wide bins, bursty |
| LTE | 15 kHz | 1024/2048 | 600/1200 | normal CP | long dense symbols, fine bins |
| 5G NR (µ=1) | 30 kHz | 1024 | ~600 | normal CP | between LTE and WiFi; mini-slot-ish |

A common baseband sample rate + STFT geometry is used across techs so the 128×128 spectrograms
are comparable; the SCS/symbol-duration differences then show up as distinct time-frequency
structure the `RouterNet` can separate.

## Signal chain (per sample) — Sionna PHY primitives

1. **Bits → symbols**: `sionna.phy.mapping.Mapper` with a QAM `Constellation`
   (`num_bits_per_symbol`: QPSK=2, QAM16=4, QAM64=6, QAM256=8; BPSK via 1-bit PAM/custom
   constellation). Random bits from `sionna.phy.mapping.BinarySource`.
2. **OFDM**: `sionna.phy.ofdm.ResourceGrid` configured per protocol (`fft_size`,
   `subcarrier_spacing`, `cyclic_prefix_length`, `num_guard_carriers`, pilots) →
   `ResourceGridMapper` → `OFDMModulator` → baseband time-domain I/Q. Use enough OFDM symbols
   to fill ~128 STFT frames.
3. **Multipath + Doppler**: `sionna.phy.channel.tr38901.TDL(model, delay_spread,
   carrier_frequency, min_speed, max_speed)` → CIR → `cir_to_time_channel` /
   `sionna.phy.channel.TimeChannel`. Mobility sets speed → Doppler (static≈0, pedestrian≈3 km/h,
   vehicular≈60–120 km/h at carrier ~3.5 GHz).
4. **AWGN**: `sionna.phy.channel.AWGN` with noise variance from the target SNR
   (`sionna.phy.utils.ebnodb2no` or direct `no` from SNR on signal power).
5. **Spectrogram**: `torch.stft` (n_fft=512, hann, hop chosen for ~128 frames) on the I/Q →
   magnitude → dB (20·log10) → crop/resize to **128×128** → per-sample z-score (matches
   `spectro_patchify.normalize_per_sample`). Sionna 2.x returns torch tensors, so this stays
   in-framework.

## Files to create
```
spectro/datagen/
  __init__.py
  phy_params.py     # @dataclass ProtocolConfig (fft_size, subcarrier_spacing, cp_len, bw, ...) for LTE/WiFi/5G
  sionna_blocks.py  # thin builders: mapper(mod), ofdm_modulator(cfg), tdl_channel(mobility), awgn(snr)
  spectrogram.py    # iq -> 128x128 dB spectrogram (torch.stft) + per-sample normalize
  generate.py       # CLI: sweep grid x per-combo, run Sionna chain, save sharded .pt + manifest
  visualize.py      # save a grid of example spectrograms per (tech,mod) for sanity
  README.md         # usage + caveats
```
Dependency: **`sionna>=2.0`** (+ existing numpy/torch/scipy). No hand-rolled DSP — the
`modulation`/`ofdm`/`channel` math is delegated to Sionna; we own only numerology + spectrogram.

**To verify at implementation time** (Sionna API specifics): exact import paths and tensor/batch
conventions for `OFDMModulator` time-domain output, the `TDL`→time-channel application path, BPSK
constellation handling, and the SNR→noise-variance mapping. Pin the Sionna version once it imports.

## Integration with the existing pipeline
- `generate.py --out spectro/outputs/synthetic/` writes sharded `.pt` files + a `manifest.json`.
- Add an optional `source` arg to `spectro_data.load_spectro_data` (or a thin
  `load_synthetic_data`) returning the same `SpectroData` minus the precomputed embeddings.
- `spectro_pretrain.py` gains `--data {demo,synthetic,mixed}` to pretrain experts on the
  synthetic corpus (per-protocol subsets) — the larger corpus is the whole point.
- Downstream evaluation (`spectro_train_heads.py`) keeps using the **real `demo_data.pt`** so
  the Transformer baseline (precomputed `moe_embedding`) stays valid; the Mamba arm now loads
  weights pretrained on synthetic data. This frames it as **sim-pretrain → real-transfer**.
  Optional `--data mixed` pretrains on synthetic + the demo train split to reduce sim-to-real gap.

## Validation
1. **Visual**: `visualize.py` grid shows visibly different LTE/WiFi/5G structure and
   modulation/SNR effects.
2. **Separability**: train `RouterNet` on a synthetic batch → high protocol accuracy
   (techs are distinguishable as intended).
3. **Stats match**: synthetic 128×128 dB spectrogram mean/std/range comparable to `demo_data`
   (which we measured: per-sample-normalized, roughly min≈-1.6, max≈5.3 pre-normalization).
4. **Smoke**: generate a few hundred samples fast (`--per-combo 4`), run `spectro_pretrain.py
   --data synthetic --smoke`, confirm masked-MSE decreases.
5. **End-to-end**: pretrain Mamba on a scaled synthetic corpus, run the demo-data sweep, regenerate
   `spectro_performance_vs_samples.png`; check the Mamba arm improves over the demo-only pretrain.

## Risks / notes
- **Sim-to-real gap**: synthetic-pretrained features may transfer imperfectly to the real demo
  spectrograms; `--data mixed` and honest reporting mitigate this. Documented in README.
- Approximate PHY (not standard-compliant) — fidelity is "good enough to be protocol-separable",
  not for spec conformance. Replace with authors' real generator when available.
- Keep generation deterministic (seeded) and sharded so large corpora are resumable.
