# LWM-Spectro: single-carrier study

Reproduction and extension of **LWM-Spectro: A Foundation Model for Wireless Baseband Signal
Spectrograms** (Kim, Alikhani, Alkhateeb — [arXiv:2601.08780](https://arxiv.org/abs/2601.08780),
PDF in the repo root). We pretrain per-protocol MoE backbones — a **Mamba-MoE** and a
**Transformer-MoE** — with masked spectrogram modelling on synthetic DeepMIMO-channel spectrograms,
then evaluate frozen on two downstream tasks: **modulation** (5-class) and **joint SNR/Doppler**
(21-class).

**Goal figure:** both pretrained LWMs beat their matched random-init controls, and beat the
no-pretraining baselines (raw patches, from-scratch DeepCNN, frozen ImageNet ResNet-18 /
MobileNetV3-S), with the gap largest few-shot.

---

## 1. Why the pipeline was rebuilt (read this before changing datagen)

The study previously produced results that made no physical sense — every token-based arm at chance
on modulation, ImageNet CNNs beating the domain model. Three independent root causes were measured:

**(a) Wrong transmit waveform.** The paper generates a **single-carrier pulse-shaped** waveform
(eq. 4: `x[n] = Σ s[i]·g[n − i·N_os]`). There is no IFFT, subcarrier, cyclic prefix or resource grid
anywhere in it. This repo used Sionna **OFDM**, where summing 52–624 subcarriers makes the time
signal ~complex-Gaussian by the CLT and **erases the constellation**. Measured ceiling on the OFDM
corpus: macro-F1 **0.555** all-SNR from blind 4th-order cumulants, with **BPSK recall 0.00**.
The whole magnitude→complex/IQ pivot was chasing a problem created by the waveform substitution;
on a single-carrier waveform the constellation survives in the envelope and single-channel
magnitude is sufficient, exactly as the paper does it.

**(b) Dead pretext task.** Masked data-bearing REs of an unequalized complex grid are
information-theoretically unpredictable, so MSE's optimal prediction is the mean. Measured: `mlm`
pinned at **0.906** against a computed trivial floor of **0.8997** — reconstruction contributed no
gradient, and 4-neighbour interpolation scored *worse* than predicting zero (MSE 1.313, optimal
α = −0.001). On the single-carrier corpus interpolation cuts token MSE by **88–92%**, and `mlm`
trains to ~0.17 against a floor of 0.974.

**(c) SNR was a shortcut.** Guard subcarriers carried zero signal and full noise, so a single
occupied-vs-guard brightness ratio separated the 7 SNR classes at **27–60σ**. Any feature extractor
solves that, so the task could not discriminate representations at all. Now **0.5σ**.

A fourth, architecture-level bug: the Transformer's `nn.Embedding` positional init (N(0,1)) made
position ~**75%** of the input variance, so attention stayed diffuse and averaged away the per-token
statistic modulation lives in. At std **0.02** (standard ViT) the random-init transformer's
modulation probe goes 0.253 → **0.316**, matching mamba's 0.309.

---

## 2. Pipeline

### 2.1 Generation — `spectro/datagen/`

```
bits → Mapper → ×N_os upsample + RRC pulse shape            (eq. 4, pulse_shaping.py)
     → random in-band frequency offset                       (removes the guard-band SNR shortcut)
     → DeepMIMO rays as taps, each faded by 12 sub-rays      (eq. 5-6, deepmimo_channel.sc_channel_apply)
     → AWGN at the sample's SNR → receive matched filter
     → |STFT|² → 10·log10 → corpus-normalized 128×128 ×1ch   (eq. 7-10, spectrogram.sc_power_spectrogram)
```

Locked config (each value measured, not guessed — see §4):
`sps=2`, roll-off 0.35/0.25/0.15 (LTE/WiFi/5G), 131072 symbols (**17.1 ms** LTE burst),
`win_length=8` (4 symbols), `n_fft=128`, **no resize**, corpus-level dB normalization
(mean −4.5, std 13.3).

```bash
bash spectro/datagen/regen_sc_corpus.sh      # corpus + both eval sets, ~4 min on one GPU
```

| dataset | n | purpose |
|---|---|---|
| `spectro_corpus_sc_s1` | 132,748 | pretraining (85% users, 20 cities) |
| `spectro_eval_sc_indist_s1` | 23,426 | downstream "seen" (15% users, **user-disjoint**) |
| `spectro_eval_sc_heldout_s1` | 6,000 | downstream "unseen" (3 held-out scenarios) |

Patch size is **not** baked in — the corpus stores 128×128 spectrograms and patchification happens
at train time, so one corpus serves patch 4 and patch 8.

### 2.2 Pretraining — `spectro_pretrain_real.py`

3 per-protocol experts (LTE/WiFi/5G) + a CNN router, 12 layers × d_model 128, masked-spectrogram
modelling at 70% mask. **Reconstruction-only** (`--w-cont 0`, eq. 21) — the paper adds contrastive at
*fine-tuning* (eq. 22), and folding SupCon into pretraining also trains on the downstream label sets,
which undercuts the self-supervised claim.

### 2.3 Downstream — `spectro_train_heads.py` → `collate_csv.py` → `plot_from_csv.py`

Every arm is a **frozen backbone + trainable head**, except the end-to-end arms. Same head and same
recipe for all arms. Few-shot axis = per-class counts {2,5,10,20,50,100}; metric = macro-F1.

Arms: `mamba`, `transformer_synth`, `random_init` (`--moe-arch {mamba,transformer}`), `resnet18`,
`mobilenet_v3_small`, `raw`, `deepcnn`.

**Always run `random_init` with the SAME `--moe-arch` as the pretrained arm you are comparing to.**
A transformer measured against a mamba control is not a pretraining lift.

---

## 3. Running it

**Locally** (patch 8 fits a 8 GB GPU; ~2 h end to end):
```bash
bash spectro/scripts/run_sc_local.sh          # pretrain both arches + all downstream arms
bash spectro/scripts/collate_plot_sc.sh       # -> spectro/outputs/plots/study_sc2_p8_{seen,unseen}.png
```

**On the cluster** — the corpus must be on HF first:
```bash
# once, from the box holding the data
python spectro/scripts/hf_upload_gridstft.py --repo $HF_USER/lwm-spectro-sc \
    --corpus-dir spectro/outputs/spectro_corpus_sc_s1 --eval-dir spectro/outputs/spectro_eval_sc_indist_s1
python spectro/scripts/hf_upload_gridstft.py --repo $HF_USER/lwm-spectro-sc-heldout \
    --eval-dir spectro/outputs/spectro_eval_sc_heldout_s1

# then, on the cluster
git pull
REPR=sc bash cluster/download_data.sh         # verifies local vs REMOTE manifest (n_samples + shards)
REPR=sc STUDY_SEEDS=1 bash cluster/run_study.sh
REPR=sc bash cluster/12_publish_study.sh
python spectro/scripts/plot_from_csv.py --hf-repo $HF_USER/lwm-spectro-results --variant sc
```

`REPR=sc` is the default and the only live study. Override patches with `SC_PATCHES="4 8"`.

---

## 4. What is measured, and what is still open

**Corpus acceptance** (from-scratch CNN / moment probe, chance in parens):

| task | seen | unseen | chance |
|---|---|---|---|
| modulation 5-cls | 0.382 | 0.341 | 0.200 |
| snr_doppler 21-cls | 0.280 | 0.255 | 0.048 |
| snr 7-cls | 0.620 | 0.553 | 0.143 |
| mobility 3-cls | 0.583 | 0.562 | 0.333 |
| BPSK/QPSK @SNR≥15 | 0.710 | 0.561 | 0.500 |
| guard-band shortcut | 0.59σ | 0.51σ | (OFDM: 27–60σ) |

**Latest study** (patch 8, seed 1, 3000 steps ≈ 4 epochs — *direction only, not citable*). Lift over
matched random-init at 100/class:

| | modulation | snr_doppler |
|---|---|---|
| mamba, seen | +0.115 | +0.057 |
| transformer, seen | +0.082 | +0.087 |
| mamba, unseen | +0.083 | +0.059 |
| transformer, unseen | +0.073 | +0.048 |

**Open issues — do not present results without addressing these:**

1. **No error bars.** `--seeds` is not passed by `11_downstream_grid.sbatch`, so `score_std = 0` and
   the ± bands in every figure are identically zero.
2. **Lift is ~zero at 2–5 samples/class and grows with N** — the opposite of the "largest gap
   few-shot" claim the study is built around. Seen in two independent runs.
3. **The LWM does not beat the CV baselines on modulation** (mamba 0.335 / transformer 0.316 vs
   MobileNet 0.365, DeepCNN 0.384, ResNet-18 0.331). The domain win is snr_doppler only.
4. **The pos-embed fix traded absolute snr_doppler for modulation** (transformer 0.420 → 0.348).
   An intermediate std (0.1, probe 0.295) is untested and may keep both.
5. **CV arms are handicapped** — no ImageNet mean/std normalization, bilinear 128→224 resize
   (`spectro_train_heads.py`). Fix before claiming a win over them, and add a *fine-tuned* ImageNet
   arm, which is what the paper actually compares against.
6. **`--project-dim` is dead for 3-D features**; there is **no `--layer` flag**, so only the final
   layer is ever probed — the standard worst case for a reconstruction-pretrained encoder.
7. **Masks are built once** and frozen for the whole run.
8. **DeepCNN collapses** at some counts (0.103 / 0.130 at 10 and 20 per class).

---

## 5. Layout

```
spectro/datagen/
  pulse_shaping.py              RRC taps, pulse shaping (eq. 4), matched filter, freq offset
  spectrogram.py                sc_power_spectrogram (eq. 7-10) + legacy OFDM representations
  deepmimo_channel.py           ray tables + sc_channel_apply (sub-ray Doppler fading, eq. 5-6)
  phy_params.py                 SC_CONFIGS (locked numerology) + label grid
  generate_deepmimo_spectro.py  main generator (--waveform sc)
  regen_sc_corpus.sh            corpus + both eval sets
  sweep_sc_config.py            config sweep: CNN + moment probes, guard-shortcut metric
  validate_sc_dataset.py        acceptance gate (per-SNR, BPSK/QPSK confusion)
spectro/scripts/
  spectro_pretrain_real.py      masked-spectrogram pretraining
  spectro_train_heads.py        downstream arms + few-shot sweep
  spectro_backbones.py          expert factory; HF-Transformer patches (SDPA, pos-embed, [x,x²])
  spectro_{moe,patchify,data,sweep,mamba_model,train_heads_config}.py
  probe_layers_modulation.py    per-layer probe: where a task lives in a backbone
  run_sc_local.sh               local end-to-end run
  collate_plot_sc.sh            collate + plot the local run
cluster/                        config.env (REPR=sc), 10_pretrain / 11_downstream / 12_publish
```

**Gotchas that cost real time before:**
- Legacy OFDM representations (`grid_stft`, `grid_complex`) produce **identically shaped** tensors,
  so a repr mismatch between pretraining and eval was undetectable. The manifest now stamps
  `waveform / sc_win / sc_norm / channels / sc_config`, and `download_data.sh` compares the local
  manifest against the **remote** one.
- The corpus is stored **already normalized** (corpus-level dB). `spectro_patchify` disables its
  per-sample z-score when the manifest says so — re-normalizing would delete the modulation cue.
- The `second_order_embed` flag changes `proj`'s weight **shape** and is stamped into the
  checkpoint, so a pretrain/eval mismatch raises rather than silently degrading.
