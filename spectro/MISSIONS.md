# Spectro Missions — strengthening & verifying the transformer-vs-mamba result

Post-validation work to harden the LWM-Spectro spectrogram study. Each mission has a goal, status,
concrete steps, the run-log paths, and a findings section filled in as we go.

**Context (current validated result — single seed, patch=4, magnitude, paper recipe):**
Downstream demo accuracy @100% train (router routing), vs raw floor and the published baseline:

| task | raw | TF-baseline(pub) | TF-ours | mamba-ours |
|---|---|---|---|---|
| modulation | 0.58 | 0.96 | 0.91 | 0.90 |
| SNR | 0.38 | 1.00 | 0.85 | 0.84 |
| **mobility** | **0.38** | **0.69** | **0.38** | **0.45** |

Both backbones beat raw on mod+SNR and tie each other; **mobility shows ~no gain (≈ raw floor)** — M1.
Recipe: `spectro_pretrain_real.py` (step-based, demo-probe every 2000 steps); paper Table I
(λ_recon=1.0, λ_cont=0.3, τ=0.2, mask 0.7, AdamW wd=0.05, lr 5e-4 warmup→cosine 1e-8).
Checkpoints: HF `tomerraviv95/wimamba-spectro-ckpts`; W&B offline `real-transformer`/`real-mamba`.

---

## M1 — Fix the mobility task (no downstream gain)  ·  STATUS: DONE
**RESULT:** mobility downstream 0.35 (old) → **~0.44 (both arches)**, clears the 0.38 raw floor.
Fix = three parts, all needed: (1) `--symbol-mult 8` (burst long enough for Doppler), (2) `--vary-speed`
(per-class speed ranges → train mobility distribution overlaps test; cross-transfer 0.331→0.365),
(3) `meanstd_t` pooling (per-freq temporal-std readout — mean-pool discarded mobility). vary-speed also
fixed mamba's degradation (fixed-speed mult8 hurt mamba −0.056 → vary-speed ~0.00). Pretraining itself is
~neutral on mobility (gain is data+readout); residual gap to published 0.69 = channel-model difference
(our DeepMIMO-TDL vs demo's gen) — unfixable without their generator. **Recipe for our arms going forward:
gen with `--symbol-mult 8 --vary-speed`, downstream `--pool meanstd_t`.** (verbose working notes below.)

## M1 (superseded — DIAGNOSED; PARTIAL FIX)
**CONCLUSION:** root causes fully found; mobility is now **recoverable above the 0.38 floor (~0.42–0.44)**
via data + readout fixes, but **pretraining itself does not drive the mobility gain** (objective can't
capture it cheaply). Best config: **mult=8 corpus + `meanstd_t` downstream pooling, plain mean contrastive**
(do NOT use the temporal mobility contrastive — it hurt: TF 0.42→0.37, mamba 0.44→0.42).
Mobility results (demo, before→after pretrain):
  mean-pool:      TF 0.36→0.34, mamba 0.42→0.35   (both below/at floor)
  meanstd_t:      TF 0.42→0.44, mamba 0.44→0.38   (cleared floor; pretrain ~neutral)
  meanstd_t+mobSupCon: TF 0.42→0.37, mamba 0.44→0.42  (worse — sc_mob never bootstraps)
Three causes: (1) gen didn't encode Doppler (burst<<coherence) → FIXED `--symbol-mult` (mult=8).
(2) mean-pool readout discards temporal mobility → FIXED `meanstd_t` pooling (mean++per-freq std-over-time).
(3) pretraining objective doesn't target mobility & sc_mob can't bootstrap (degenerate gradient even
with temporal pooling) → NOT fixed cheaply; would need complex spectrograms (authors' rep) — deferred.
**Decision needed:** accept data+readout win (mobility clears floor; document the pretrain-objective
limit) and move to M2, vs invest in complex spectrograms to make pretraining capture mobility.

(original notes below)
## M1 — Fix the mobility task (no downstream gain)  ·  (working notes)
**Problem:** our pretrained models sit at the raw floor on mobility (~0.38), while the published
baseline reaches 0.69. `sc_mob` (mobility contrastive) stayed frozen at ln(batch) the entire run.
**Goal:** after re-pretraining BOTH arches, see *some* mobility gain over raw (below 0.69 is fine).

Steps:
- [ ] **Diagnose** whether mobility is recoverable from the magnitude spectrogram at all
      (probe demo + synthetic with mean/std AND temporal-structure features; mobility = Doppler →
      lives in time-coherence across STFT frames, not in pooled stats).
- [ ] **Check the generator** encodes mobility distinguishably (static vs pedestrian vs vehicular
      Doppler → measurable spectrogram difference). If washed out: fix gen (more time frames /
      keep Doppler structure / less aggressive normalization).
- [ ] **Pick a fix** (one or more of): (a) generator preserves Doppler, (b) contrast on mobility
      with a representation where it's accessible, (c) a temporal-aware feature for the head.
- [ ] **Re-pretrain** both arches on a small subset, confirm mobility > raw in the before/after probe.
- [ ] **Sweep** to confirm the mobility gain holds on the full demo eval.
Run logs: `cluster/logs/m1_*.log`
Findings:
- **ROOT CAUSE = generator, not architecture.** Mobility probe (linear, 3-class, chance 0.333):
  DEMO(real) pooled=0.333 but **temporal features=0.465** (mobility lives in time-variation across
  STFT frames). SYNTH(mine) ≈ chance (0.31–0.32) for **every** feature → my spectrograms encode
  NO mobility signal. So no model can learn it from my corpus.
- **Why:** OFDM burst is ~0.5–0.9 ms but Doppler coherence time is ~1.4 ms (vehicular, 350 Hz) to
  ~40 ms (pedestrian) — burst << coherence ⇒ channel ~constant across the STFT window ⇒ no Doppler
  signature. (phy_params: WiFi 160 sym/0.64ms, LTE 12/0.86ms, 5G 12/0.43ms; speeds static0/ped1/veh30.)
- **Fix = lengthen the observed time** so Doppler varies across the 128 STFT frames. Implemented
  `--symbol-mult` (sionna_blocks `build_resource_grid`/`_ofdm_chain` + generator + manifest).
- **VALIDATED at data level:** mult=8 (~5 ms window) → synthetic mobility temporal-probe **0.445**
  (vs 0.340 at mult=1, vs demo 0.465). Gen cheap: 1000 samples / 35 s, no OOM at batch 4.
- **Transfer test (mult=8, 40k corpus, before/after on demo mobility):** transformer 0.364→0.344
  (−0.02, NOT better); sc_mob FROZEN at 1.820 throughout. So fixing the data was necessary but NOT
  sufficient — **2nd blocker = the mean-pooling readout.** Mobility is per-frame temporal variation,
  but the contrastive proj head AND the downstream head both read the MEAN-pooled embedding (avg over
  1024 patches) which discards temporal variation → sc_mob can't organize it, head can't see it.
- **Temporal pooling (`meanstd_t` = mean ++ per-freq std-over-time) implemented** in both backbones'
  `embed()` (`pool_tokens`) + threaded through SpectroMoE/extract_embeddings (2*d_model) + validator.
- **mult=8 + meanstd_t result (demo mobility):** TF random 0.420→trained 0.436 (+0.016);
  mamba 0.440→0.384 (−0.056). So data+pooling lift mobility **above the 0.38 floor (~0.42–0.44)**, but
  PRETRAINING still doesn't add it — `sc_mob` froze because the mobility contrastive head also
  mean-pools. **Last lever:** give the **mobility SupCon head temporal pooling** so `sc_mob` engages
  and training organizes mobility. Implementing + testing now.

---

## M2 — Add a `random-init` arm to the sweep  ·  STATUS: DONE
**RESULT (acc @100%, demo):** ordering raw < random-init < ours < published holds for mod+SNR.
  modulation: raw .58 | rand-init .85 | TF-ours .91 | mamba .90 | pub .96
  SNR:        raw .38 | rand-init .62 | TF-ours .85 | mamba .84 | pub 1.00
  mobility:   raw .38 | rand-init .40 | TF-ours .38 | mamba .45 | pub .69  (mean-pool; M1 fix lands in M3)
Insight: the untrained MoE arch alone is strong (mod .85 / SNR .62); pretraining's LIFT over random-init
is +.06 mod but +.22 SNR (clearly helps SNR; mod is mostly architectural). `_random_init_features` in
spectro_train_heads.py (--arm random_init, oracle routing); added to plot + 02 sbatch ARMS.
**Goal:** put the pretraining "lift" explicitly on the same axes as raw / pub-baseline / ours.
Steps:
- [ ] Add `random_init` arm to `spectro_train_heads.py` (`_MOE_ARMS`-style, untrained MoE, fixed seed).
- [ ] Include it in `02_downstream_spectro.sbatch` ARMS and the plot's MODEL_TYPES.
- [ ] Re-run the sweep; confirm ordering raw ≤ random_init ≤ ours ≤ pub-baseline.
Run logs: `cluster/logs/m2_*.log`
Findings: _(to fill)_

---

## M3 — Patch-size-parameterized pretrain + downstream (patch ∈ {4,6,8})  ·  STATUS: DONE (plumbing+smoke)
**Goal:** run pretraining/downstream with a chosen patch size; **persist patch size in every
output name + config** so runs are retrievable later. (patch 4→elem16/seq1025, 6→elem36/seq442,
8→elem64/seq257; expert `element_length` and `max_len` follow the patch. All three `side` values
32/21/16 are perfect squares so `meanstd_t` temporal pooling works for every patch.)
**Decisions (this session):** scope = plumbing + launchers + smoke (NOT the full multi-hour runs —
user triggers those for the patch/target they pick). M1 recipe baked in as the default: corpus
generated `--symbol-mult 8 --vary-speed` (use `spectro/outputs/spectro_deepmimo_mult8_vary`),
downstream `--pool meanstd_t`.
Steps:
- [x] `patch_geometry(patch,channels)` helper in spectro_patchify (element_length/n_patches/max_len/side).
- [x] Thread `--patch` through `spectro_pretrain_real.py` (build_masked_tensors, build_expert max_len,
      _embed/demo_probe), `SpectroMoE` (`patch` attr → `_expert_embed` patchify), and the sweep
      `spectro_train_heads.py` (raw/random_init/moe arms + `--pool` default meanstd_t).
- [x] Weights dir → `spectro_{arch}_p{patch}_weights/` (`weights_dir(arch,patch)`); sweep submissions →
      `submission_spectro_{arm}_p{patch}/`; plot → `..._p{patch}.png`; `patch`+element_length+max_len
      recorded in each `{proto}_expert.pth`; W&B run name `real-{arch}-p{patch}`.
- [x] Launchers: local `spectro/scripts/run_patch_study.sh` (PATCH/MODE/ARCHES env, GPU0-only) +
      cluster `01`/`02` sbatch parameterized by `PATCH`/`POOL` (config.env SPECTRO_PATCH/SPECTRO_POOL).
- [x] Shapes smoke — patch 4/6/8 × {transformer,mamba}: patchify/masked-tensors/expert-forward/MoE-embed
      all correct (p4 1024/16, p6 441/36, p8 256/64; meanstd_t → 256-d), finite.
- [x] Short pretrain smoke (patch 6, transformer, 200 steps, mult8_vary corpus) → wrote
      `spectro_transformer_p6_weights/` with `{patch:6, element_length:36, max_len:442}` in each
      `{proto}_expert.pth`; router val_acc 0.996; demo probe ran each eval. Downstream smoke (p6,
      transformer_synth, meanstd_t, 2 epochs) loaded those checkpoints → 256-d features (finite),
      wrote `submission_spectro_transformer_synth_p6/`; plot wrote `..._p6.png`. (Toy artifacts then
      deleted — they're 200-step/2-epoch, not real numbers.)
Run logs: `cluster/logs/m3_p{4,6,8}_*.log`
**HOW TO RUN (real):** local — `PATCH=6 bash spectro/scripts/run_patch_study.sh` (GPU0; full
pretrain+downstream+plot, M1 recipe defaults). Cluster — `sbatch --export=ALL,ARCH=transformer,PATCH=6
cluster/01_pretrain_spectro.sbatch` (per arch), then `sbatch --export=ALL,PATCH=6
cluster/02_downstream_spectro.sbatch`. Corpus must be the M1 one (`--symbol-mult 8 --vary-speed`,
e.g. `spectro/outputs/spectro_deepmimo_mult8_vary`). Pick PATCH ∈ {4,6,8}; everything is stamped p{PATCH}.
Findings: plumbing validated end-to-end for patch 4/6/8 × {transformer,mamba}. Geometry: p4 1024
tokens/elem16/max1025, p6 441/36/442, p8 256/64/257; sides 32/21/16 all perfect squares → `meanstd_t`
temporal pooling valid at every patch. Nothing in the pipeline hard-codes 1024/16/1025 anymore.

---

## M4 — 2 more seeds per patch size (after M3 validates)  ·  STATUS: TODO
**Goal:** statistical confidence — run 2 additional seeds for each patch size; report mean ± 95% CI.
Steps:
- [ ] For each validated patch size, re-run pretrain+downstream with 2 more seeds (seed in name/config).
- [ ] Aggregate mean ± CI per (patch, arch, task); update the plot with error bars.
Run logs: `cluster/logs/m4_p{patch}_seed{N}_*.log`
Findings: _(to fill)_

---

### Conventions
- Run logs live under `cluster/logs/` with the `m{N}_` prefix shown above.
- Checkpoints carry patch (and later seed) in the dir name; configs/manifests record patch+seed.
- Update each mission's STATUS (TODO → IN PROGRESS → DONE) and Findings as we go.
