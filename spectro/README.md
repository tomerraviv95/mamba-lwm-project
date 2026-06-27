# Spectrogram domain: WiMamba MoE vs. LWM-Spectro Transformer

This directory extends the repo to a **second domain** — wireless baseband I/Q signals
rendered as **128×128 time-frequency spectrograms** — and reproduces the "performance vs.
number of training samples" transfer story there, comparing a **Mamba Mixture-of-Experts**
against the pretrained **LWM-Spectro Transformer MoE** baseline
([`wi-lab/lwm-spectro`](https://huggingface.co/wi-lab/lwm-spectro)).

It is **fully isolated** from the existing DeepMIMO-channel pipeline in `scripts/` and
writes only under `spectro/outputs/`. Domain-agnostic pieces (Mamba primitives, the
fine-tune/probe harness) live in the repo-level `shared/` package.

## What's here

```
spectro/scripts/
  download_spectro_hf.py        # pull LWM-Spectro source + demo_data.pt from HF
  spectro_patchify.py           # single-channel 4x4 patchify (element_length=16) + MLM masking
  spectro_data.py               # load demo_data.pt; labels + protocol-stratified split
  spectro_mamba_model.py        # lwm_mamba_spectro: single-channel bidirectional Mamba backbone
  spectro_moe.py                # MambaMoE: per-protocol experts + CNN RouterNet
  spectro_transformer_model.py  # baseline features (precomputed moe_embedding)
  spectro_pretrain.py           # masked-spectrogram pretraining of experts + router training
  spectro_train_heads_config.py # ClassificationHead + per-task configs
  spectro_train_heads.py        # MAIN: extract per-arm features, run the sample sweep
  spectro_sweep.py              # percentage sweep -> aggregated_results.json + radar
  spectro_plot_sample_variation.py  # accuracy vs #samples comparison plot
spectro/outputs/                # weights, submissions (per-arm results), plots
spectro/hf_cache/               # downloaded HF artifacts (gitignored heavy files)
```

## Arms compared

| Arm | Backbone | Features |
|-----|----------|----------|
| `transformer` | LWM-Spectro Transformer **MoE** (pretrained on the full corpus) | precomputed `moe_embedding` (128-d) shipped in `demo_data.pt` |
| `mamba` | **WiMamba MoE** — 3 per-protocol bidirectional Mamba experts + CNN router, pretrained here | routed 128-d embedding |
| `raw` | none | mean-pooled raw 4×4 patches (16-d) — lower bound |

## Downstream tasks
Modulation (5-class), SNR (7-class), Mobility (3-class). Protocol (LTE/WiFi/5G) is handled
by the router, so it is not a downstream task.

## Usage

```bash
pip install -r requirements.txt          # adds huggingface_hub
# 1. download baseline + demo data (also mirrors the HF source for reference)
python spectro/scripts/download_spectro_hf.py --inspect
# 2. pretrain the Mamba MoE (experts + router) on the demo spectrograms
python spectro/scripts/spectro_pretrain.py            # add --smoke for a fast sanity run
# 3. run the sample-variation sweep for each arm
python spectro/scripts/spectro_train_heads.py --arm transformer
python spectro/scripts/spectro_train_heads.py --arm mamba       # --routing oracle to bypass the router
python spectro/scripts/spectro_train_heads.py --arm raw
# 4. plot accuracy vs #training samples
python spectro/scripts/spectro_plot_sample_variation.py
```

## Important caveat: pretraining data

Only the **~10,500-sample `demo_data.pt`** is publicly available — the full LWM-Spectro
pretraining corpus is not in the HF repo. So each Mamba expert pretrains on only ~2,400
spectrograms of its protocol. The Transformer baseline, by contrast, uses embeddings from a
model pretrained on the full (unavailable) corpus. The comparison should therefore be read as
**"how close does a small-data Mamba MoE get to the fully-pretrained Transformer baseline as
labelled samples grow"**, not as a like-for-like pretraining comparison.

Additionally, the HF *pretraining* used complex (real/imag-interleaved) spectrograms
(`element_length=32`), but the demo data we have is single-channel real (128×128), so the
Mamba arm uses single-channel patches (`element_length=16`).

## Mobility task: the 0.44-vs-0.69 ceiling (known limitation)

Our pretrained arms reach **~0.44** on the 3-class mobility task while the published
`moe_embedding` reaches **0.69** (both on the same held-out demo split, plain mean-pool).
A focused diagnostic (see `MISSIONS.md` M5) ruled out the usual suspects: the demo is
*magnitude* (the published model consumes magnitude too — not complex/phase), both sides are
z-scored `20·log10` dB, mean-pool is not fatal (their 0.69 *is* mean-pool), the supervised
mobility contrastive is degenerate (`sc_mob` never leaves its init — mobility is too weakly
separable for SupCon to bootstrap), and time-column masking does not help (the model
interpolates a missing time slice without encoding Doppler rate).

By elimination the gap is **pretraining strength**: the published embedding comes from a
*large + in-domain* corpus (their generator: many cities × FFT sizes × balanced mobility,
~100 epochs). The two factors we can control each cap at ~0.44 — large-but-out-of-domain
(our 150k DeepMIMO synthetic) and in-domain-but-small (the 10.5k demo). Only large **and**
in-domain reaches 0.69, and that corpus is not public. **What does work and is the locked
recipe:** the `meanstd_t` readout (mean ++ per-frequency temporal-std; `--pool meanstd_t`,
the default) plus a Doppler-bearing corpus (`--symbol-mult 8 --vary-speed`); pretraining then
beats random-init on mobility (mean-pool 0.36→0.41, meanstd_t 0.42→0.44) — small but real and
leakage-safe. Closing to 0.69 would require reproducing their generator's mobility corpus at
scale (deferred).
