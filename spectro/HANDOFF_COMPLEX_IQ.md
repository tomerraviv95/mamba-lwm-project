# LWM-Spectro — handoff: complex/IQ study & the "LWM loses to ImageNet CNNs" puzzle

Written 2026-08-06 for a fresh agent. Branch `feat/spectograms`. Companion to `spectro/PROVENANCE.md`
(data map) and the memories `spectro-domain-contract`, `spectro-paper-narrative-verdict`,
`spectro-complex-iq-validation`. **Read this before touching the complex/IQ (grid_complex) study.**

---

## 0. The one-paragraph situation

We are building a **wireless foundation model** (LWM): per-protocol MoE backbones (a **Mamba-MoE** and a
**Transformer-MoE**, arch-parameterized) pretrained with masked spectrogram modeling on synthetic
DeepMIMO-channel spectrograms, then evaluated on downstream wireless tasks (**modulation 5-cls**,
**joint SNR/Doppler 21-cls**). **The goal figure:** transformer ≈ or ≥ mamba; both pretrained LWMs beat
the no-pretraining baselines (random-init LWM, from-scratch DeepCNN, frozen ImageNet ResNet/MobileNet, raw
patches); the gap is largest in the few-shot regime and shrinks as training samples grow.

**The open puzzle the user flagged:** the ImageNet-pretrained CV models (ResNet-18, MobileNetV3) are
*competitive with or beating* our wireless-tailored LWM on some tasks/regimes. That is philosophically wrong
— the whole premise is that a model tailored to wireless channel spectrograms should beat a generic vision
model. Where exactly this happens, and why, is the core thing to resolve. See §6.

**Current state:** we pivoted magnitude→complex representation (the right move; see §5), but **no complete,
correct run exists yet**: the only complete-on-correct-data run is patch 8, where a patchification artifact
destroys the modulation signal (§5.4). The definitive run (patch 4 + clean 132k + snr-fix, ideally 5 seeds)
is **still incomplete** (§7).

---

## 1. Data generation (`spectro/datagen/`)

Pipeline per sample: **DeepMIMO ray-tracing → per-user PDP → Sionna OFDM waveform → 3GPP-TDL fading + per-ray
Doppler + AWGN → spectrogram.**

- **`generate_deepmimo_spectro.py`** — main generator. For each of 20 LWM cities, sample user PDPs
  (ray-traced delay/power/phase/AoA). Each PDP → one spectrogram with a random `(tech∈{LTE,WiFi,5G},
  mod∈{BPSK,QPSK,QAM16,QAM64,QAM256}, snr∈7 levels, mobility∈{static,ped,vehicular})`. AWGN at the sample's
  SNR; Doppler from AoA+speed. Output = sharded `.pt` dicts `{tech,snr,mod,mob,city,data}` + `manifest.json`.
  Key flags: `--repr`, `--complex`, `--per-city`, `--all-users`, `--user-split-frac/-part` (disjoint
  train/downstream user split, fixed seed), `--symbol-mult` (OFDM symbols per burst → slow-time window for
  Doppler; **memory scales with it, lower `--batch` accordingly**), `--vary-speed`, `--cities` (held-out
  scenarios: `asu_campus_3p5:1,boston5g_3p5:2,o1_3p5:3`), `--snr-range`.
- **`spectrogram.py`** — the representations (all → `(C,128,128)` float16, per-sample z-scored):
  - `iq_batch_to_spectrogram` — `|STFT|` of the time-domain OFDM waveform. **Modulation is INVISIBLE here**
    (OFDM waveform ~Gaussian by CLT regardless of constellation).
  - `grid_mag_to_spectrogram` — `|Y[k,n]|` of the demodulated received resource grid. Modulation *amplitude*
    partly visible, but **BPSK vs QPSK are indistinguishable** (both constant |·|). NEAREST resize (bilinear
    would average the constellation away).
  - `grid_complex_to_spectrogram` (**NEW, commit c05ca39**) — 2-channel `[Re(Y), Im(Y)]` of the received
    grid. **KEEPS PHASE → full constellation separable incl. BPSK vs QPSK.** This is the key addition.
  - `iq_batch_to_complex_spectrogram` — `[Re,Im]` of the STFT (exists, not used in the current study).
- `--repr` options: `stft` (magnitude STFT), `grid` (magnitude grid), `grid_stft` (2ch [STFT|grid], the OLD
  magnitude study), **`grid_complex`** (the NEW phase-bearing study).

**Physics of why representation matters (the crux of the whole project):** modulation lives in the *phase/
quadrature* of the constellation. Magnitude representations throw phase away → modulation caps ~0.5 and a
random SSM already reaches it (no headroom for pretraining). Complex/IQ keeps phase → modulation becomes
genuinely hard-from-scratch, so pretraining can win. **De-risk proof (local, high-SNR, from-scratch CNN):
BPSK recall 0.00 (magnitude) → 0.97 (complex).**

**Datasets on HF (all verified 2026-08-06 after fixing a manifest bug, see §8):**
- `tomerraviv95/lwm-spectro-complex` : `corpus/` (132,748, 85%-user, grid_complex, symbol_mult 8) +
  `eval/` (23,426, disjoint 15%-user, in-dist "seen").
- `tomerraviv95/lwm-spectro-complex-heldout` : `eval/` (6,000, 3 held-out cities, cross-env "unseen").
- (Magnitude study data: `lwm-spectro-alluser` (132k corpus + 23k eval), `lwm-spectro-gridstft` (heldout).)

---

## 2. Pretraining (`spectro/scripts/spectro_pretrain_real.py`)

Step-based masked-spectrogram pretraining of a per-protocol MoE (3 experts LTE/WiFi/5G + a CNN RouterNet).

- One expert per protocol, `build_expert(arch)` (`spectro_backbones.py`): 12-layer, d_model=128. Transformer
  uses `F.scaled_dot_product_attention` (already flash/mem-efficient — NOT the bottleneck). Mamba = 12-layer
  bidirectional SSM.
- **Objective:** `loss = w_mlm·MLM + w_cont·(SupCon_mod [+ SupCon_mob])`.
  - **MLM** = MSE on masked patch tokens (mask 70%). This is the paper's reconstruction objective.
  - **SupCon** = supervised contrastive on modulation and mobility labels (projection heads, temp 0.2).
  - Paper pretraining is recon-only (`w_cont=0`); **on complex we set `w_cont=0.3`** because MLM-only
    COLLAPSES on the complex grid (masked data-bearing REs are ~unpredictable → trivial "predict-mean"
    minimizer). SupCon engages on complex (modulation IS separable) — it was frozen at ln(batch) on magnitude.
- **CLI:** `--arch --patch --seed --steps --weights-suffix --pretrain-dir --batch-size --accum-steps
  --w-mlm --w-cont --mask-percent --lr --warmup-frac --early-stop-patience --eval-every --eval-task`.
  Warmup(0.1)→cosine, lr 5e-4, AdamW wd 0.05. Early-stop on an in-corpus mean-pool probe (OFF for complex —
  the mean-pool probe understates per-token modulation, so run full `--steps`).
- **Checkpoints:** `spectro/outputs/pretrained_models/spectro_{arch}_p{patch}_{suffix}_weights/`
  = `{LTE,WiFi,5G}_expert.pth` + `router.pth` + `{proto}_steps.csv` (live training log). **SEED is NOT in the
  dir name** → multi-seed must stamp it into the suffix (the study does: `{STUDY_SUFFIX}_s{seed}`).
- **`done3` guard** (in the sbatch): skips pretraining if 3 experts + router already exist → bump the suffix
  for a new recipe, or `rm -rf` stale dirs, or it silently reuses old weights.
- **Params:** LWM transformer expert **2.54M**, mamba **2.79M** (see §6 — this matters for the CV comparison).

---

## 3. Downstream eval (`spectro_train_heads.py` + `spectro_sweep.py`)

Every arm = **FROZEN backbone + a trainable head**, EXCEPT the end-to-end arms. Same head + same training
recipe across all arms (fair). Few-shot axis = **per-class counts** {2,5,10,20,50,100}. Metric = macro-F1
(primary) + accuracy. 70/10/20 split; val-based early stop; best-of-N head restarts.

- **Heads** (`--head`): `cnn1d` = paper's residual 1-D CNN over the **token sequence** (no pooling collapse;
  this is the study default — it respects the LWM's per-token structure) ; `mlp` = MLP on `meanstd_t`-pooled
  (mean⊕std over tokens) 256-d vector.
- **Arms** (`--arm`): FROZEN+head = `mamba`, `transformer_synth`, `random_init` (per `--moe-arch`),
  `resnet18`, `mobilenet_v3_small`, `raw`. END-TO-END = `deepcnn` (from-scratch CNN, full-res),
  `resnet18_ft` (ImageNet fine-tuned). ImageNet frozen arms feed their pre-global-pool conv map as a token
  sequence into the SAME cnn1d head (ResNet 7×7×512 → 49 tokens; MobileNet → 576-d tokens).
- **`collate_csv.py`** → one tidy CSV per (patch,seed): `patch,seed,arm,arm_label,eval,task,per_class,
  n_samples,macro_f1,accuracy`. `--variant` filters by a namespace token stamped into run-tags.
- **`plot_from_csv.py`** → per (patch,eval) figure, 2 task subplots, mean±std over seeds. `--variant`
  selects the study subfolder (`study_csv_{variant}/`). `--hf-repo` pulls first.

---

## 4. Cluster orchestration (`cluster/`)

- **`config.env`** — all knobs. `REPR=complex` switch (commit 17f47f9) flips the whole study to grid_complex:
  data dirs + HF repos → complex; `SPECTRO_W_CONT 0→0.3`; early-stop off; ckpt suffix → `gridcomplex_mob`;
  CSV/plot namespace `STUDY_VARIANT → cxcnn1d`. Magnitude defaults unchanged when `REPR` unset.
  `STUDY_FROZEN_ARMS="mamba transformer_synth resnet18 mobilenet_v3_small raw"`, `STUDY_E2E_ARMS="deepcnn"`,
  `STUDY_SEEDS`, `PATCHES`, `SPECTRO_STEPS`.
- **`10_pretrain_grid.sbatch`** — array over (arch × patch × seed); pretrains one combo per task.
- **`11_downstream_grid.sbatch`** — array over (patch × seed); runs ALL arms × {seen,unseen} → collates CSV.
  Needs BOTH arches' checkpoints for its (patch,seed).
- **`12_publish_study.sh`** — login node: push checkpoints + `study_csv*/` CSVs to HF.
- **`run_study.sh`** — submits 10 (array) then 11 (array, `afterok` dependency) and prints the publish line.
- **`download_data.sh`** — login node: pull corpus+evals (`REPR=complex` for the complex data;
  `FORCE=1` to re-pull), pre-cache ImageNet weights for frozen-vision arms (offline compute nodes).
- **Run recipe:** `git pull` → `REPR=complex FORCE=1 bash cluster/download_data.sh`
  (verify `grep n_samples .../spectro_corpus_complex_s1/manifest.json` == **132748**) →
  `REPR=complex STUDY_SEEDS=... PATCHES=... [SPECTRO_STEPS=...] bash cluster/run_study.sh` →
  `REPR=complex bash cluster/12_publish_study.sh` → locally
  `python spectro/scripts/plot_from_csv.py --hf-repo tomerraviv95/lwm-spectro-results --variant cxcnn1d`.

---

## 5. What we ran and saw (chronological)

### 5.1 Magnitude study (grid_stft) — DONE, documented in `spectro-paper-narrative-verdict`
Verdict: on magnitude, transformer CANNOT beat mamba (mamba's random bidirectional-SSM init already extracts
magnitude features); raw+conv and frozen ResNet TIE/beat the LWM; modulation caps ~0.5. Clean pretraining
win only vs *from-scratch* models on snr_doppler. **Root cause = magnitude has no representation headroom.**

### 5.2 Complex de-risk (local) — PROVEN
`grid_complex` recovers the constellation: high-SNR from-scratch CNN BPSK recall 0.00→0.97, QPSK 0.95. On
complex, raw+conv and random-init LWM **collapse to chance** on modulation (they were ~0.5 on magnitude) =
the ceiling lifted; pretraining now has headroom.

### 5.3 Complex patch-4 run (on BUGGY 30k data) — ENCOURAGING but NOT on correct data
Modulation, seen, few-shot (2/5/10 per-class): **transformer 0.36/0.42/0.43** — the clear leader, **transformer
≥ mamba** the whole curve (impossible on magnitude!). raw/random at chance. This is the result we WANT — BUT
it was trained on the stale 30k Frankenstein corpus (§8 manifest bug), NOT the clean 132k. So it is not a
trustworthy final result; it only shows the mechanism can work at patch 4.

### 5.4 Complex patch-8 run (CORRECT 132k data, snr-fix) — COMPLETE, and it exposes a NEW artifact
Files: `spectro/outputs/plots/study_cxcnn1d_p8_{seen,unseen}.png`; CSV `study_csv_cxcnn1d/results_p8_s1.csv`.
- **Modulation: both LWMs FLAT AT CHANCE (~0.20) at every count** — total failure. So are `raw` and
  `random_init`. But ResNet (→0.44), MobileNet (→0.35), DeepCNN (→0.48) all learn it.
- **CAUSE (confident):** patch-8 patchification pools 8×8×2 = 128 values (64 resource-elements) into one
  token; averaging 64 REs smears the constellation back to ~Gaussian — the SAME CLT effect as magnitude.
  **The tell:** `raw` also patchifies at 8×8 and is ALSO at chance, while the full-resolution CNN baselines
  (no patchify) learn modulation fine. So it is the coarse patch, not the data. **→ modulation needs patch 4**
  (4×4 = 16 REs/patch preserves it).
- **snr_doppler: healthy.** mamba leads few-shot (0.175→0.33); **transformer now ABOVE raw** (0.145→0.29 vs
  raw 0.08→0.25) — the snr-fix (§5.5) worked. Same story on unseen cities.

### 5.5 The snr_doppler fix (commit ba05369) — CONFIRMED WORKING
Symptom: transformer was BELOW raw patches on snr_doppler — wrong for a foundation model. Cause: the SupCon
term was modulation-only (the `mobility` label wasn't built by `load_synthetic_data`, so the SupCon-mobility
term was silently skipped → `sc_mob=0`). Fix: `load_synthetic_data` now builds TASKS+EXTRA_TASKS so
`mobility` exists → SupCon-mobility engages (`sc_mob` 0.000→3.43 live, then descends). Verified in the patch-8
run: transformer now clearly > raw on snr_doppler.

---

## 6. THE PUZZLE: why do ImageNet CNNs rival the wireless LWM? (unresolved — top priority)

This is what the user wants understood. Nuances (do NOT over-summarize as "CV just wins"):

- **On modulation (patch 4, the correct representation), the LWM transformer BEATS the CV models few-shot**
  (transformer 0.36 vs ResNet 0.26 vs MobileNet 0.23 @2/cls, on the 30k run). And the **param-matched**
  MobileNetV3-S (2.54M == LWM transformer 2.54M) is well behind the LWM. So on the task+representation where
  the domain model *should* win, it does. ResNet-18 (11.7M, 4.5× bigger) was only close by being large.
- **On snr_doppler, the CV models are genuinely competitive/ahead few-shot** (ResNet/MobileNet ≈ mamba).
  Reason: SNR≈global noise texture, Doppler≈spectral spread — exactly the edge/texture statistics ImageNet
  encodes, so generic vision transfers almost for free. Modulation (fine phase/constellation) is NOT
  texture-like → generic vision is weak there → domain model wins.
- **Why the LWM doesn't dominate everywhere — candidate causes (each testable):**
  1. **Objective.** LWM is reconstruction (MLM)-pretrained; MAE/MLM features are famously weak under a
     FROZEN shallow head and need fine-tuning to shine. ImageNet models are SUPERVISED-pretrained → features
     immediately linearly separable. We evaluate frozen → LWM's worst case. **Test: fine-tune the LWM
     end-to-end** (`spectro_finetune.py` exists) vs frozen.
  2. **Capacity/scale.** LWM expert 2.5–2.8M pretrained on 132k synthetic; ImageNet = 11.7M (ResNet) trained
     on 1.28M diverse labeled images. Partly controlled by the MobileNet (2.54M) baseline — and there the LWM
     wins modulation, so capacity isn't the whole story. Keep MobileNet as the fair anchor.
  3. **Task is texture-like (snr_doppler)** → no domain advantage available; this may be a genuine "tie" and
     that's honest. The domain story should be told on modulation + cross-env.
  4. **Patchification bottleneck (§5.4)** — coarse patches destroy fine structure. Patch 4 mandatory; even
     finer (patch 2?) untested.
  5. **Normalization.** Per-sample z-score removes absolute amplitude — the cue SNR needs — while CNNs get
     the same input, so this is symmetric, but worth checking a no-normalize or amplitude-preserving variant.
  6. **Feature width into the head.** ResNet 512-d / MobileNet 576-d per token vs LWM 128-d — the CV models
     hand the head more features. **Test: `--project-dim 128`** to equalize width (isolates representation
     quality from feature count).
  7. **Under-convergence.** The correct-data patch-4 run never finished; the only patch-4 win is on 30k.

---

## 7. What is INCOMPLETE / immediate next steps

1. **THE definitive run does not exist yet:** patch 4 + clean 132k + snr-fix recipe. The patch-4 cluster run
   stalled — the transformer expert is ~13h each at seq 1025 (heavy but SDPA is already efficient; likely
   full 15k steps and/or GPU-shared). **Next:** relaunch patch 4 at reduced steps (proof showed the lift by
   ~2500 steps; use `SPECTRO_STEPS=8000`, drop to 6000 if still slow):
   ```
   scancel -u $USER
   rm -rf spectro/outputs/pretrained_models/*p4_gridcomplex_mob*
   REPR=complex STUDY_SEEDS=1 PATCHES=4 SPECTRO_STEPS=8000 bash cluster/run_study.sh
   ```
   Consider a per-arch step split (transformer fewer steps than mamba) — not yet wired; would need a
   `TF_STEPS`/`MAMBA_STEPS` knob in `10_pretrain_grid.sbatch`.
2. **Resolve the CV puzzle on correct data (§6):** once patch-4/132k is in, run the diagnostics — (a)
   `--project-dim 128` equal-width probe, (b) an end-to-end fine-tuned LWM arm, (c) confirm MobileNet (matched
   params) stays behind the LWM on modulation.
3. **Then 5 seeds** (`STUDY_SEEDS="1 2 3 4 5"`) for error bars — only after 1 seed confirms direction.
4. **Optional bigger corpus:** the 132k is all-users × ~1.5 draws; can 3–5× it via fresh noise/param draws
   per user (run gen at multiple `--seed`, concat) if more data is wanted for the final run.

---

## 8. Gotchas learned (so the next agent doesn't repeat them)

- **HF manifest-skip bug (cost us a multi-day wasted run).** The uploader skipped re-pushing `manifest.json`
  because it already existed → a stale 30k manifest sat on top of 132k shards → `load_synthetic_data` reads
  the manifest's shard list and loaded only 30k. **Always overwrite the manifest; verify `n_samples` on HF
  after upload.** Fixed via `fix_hf_complex.py` (delete folder + per-file upload with retries, force
  manifest). `upload_folder` also can't survive transient S3 timeouts on ~8.7GB — use per-file + retries.
- **`download_data.sh` skips existing files** → after fixing HF you MUST `rm -rf` the local data dirs or use
  `FORCE=1`, else the cluster keeps the stale copy. Habit: `grep n_samples ...manifest.json` before every run.
- **`done3` guard** silently reuses checkpoints if 3 experts+router exist → bump suffix or `rm -rf` for a new
  recipe.
- **Datagen OOM on 8GB (and even large scenarios on 24GB):** WiFi bursts × `symbol_mult` blow up the
  time-channel tensor. Use `--batch 2..4`; the `o1_3p5` held-out scenario (497k users) needs `--batch 2`.
- **Patch 8 destroys modulation** (§5.4). Patch 4 is mandatory for the modulation headline.
- **Mamba on the cluster uses the eager conv path** (`causal_conv1d` build targets the wrong arch and is
  uninstalled by `setup_env.sh`) → slower but works; `selective_scan` stays CUDA so not the 8× worst case.
- **Local WSL box = RTX 3060 Ti 8GB** — fine for datagen and small proofs, too small for patch-4 seq-1025
  transformer pretraining. Cluster = RTX 3090 24GB (`--gpus=rtx_3090:1`).
- **mean-pool probes understate per-token modulation** on complex — judge modulation with the cnn1d
  (token-sequence) head, not the pooled probe.

---

## 9. Key files & commits
- Datagen: `spectro/datagen/generate_deepmimo_spectro.py`, `spectrogram.py` (grid_complex = c05ca39).
- Pretrain: `spectro/scripts/spectro_pretrain_real.py` (mobility-optional SupCon = a700916; snr-fix in
  `spectro_data.py` = ba05369).
- Downstream: `spectro_train_heads.py` (cnn1d-for-all-frozen-arms = a68bf0f; MobileNet baseline = 427717b),
  `spectro_sweep.py`, `collate_csv.py`, `plot_from_csv.py`.
- Cluster: `config.env` (REPR=complex = 17f47f9), `10_pretrain_grid.sbatch`, `11_downstream_grid.sbatch`,
  `12_publish_study.sh`, `run_study.sh`, `download_data.sh`.
- Figures (current): `spectro/outputs/plots/study_cxcnn1d_p8_{seen,unseen}.png` (patch 8, correct data,
  modulation collapsed), `study_cxcnn1d_p4_{seen,unseen}.png` (patch 4, on the buggy 30k data — the
  encouraging-but-untrustworthy modulation win).

## 10. One-line status
Complex/IQ is the right representation (proven), the snr fix works, but **we still have NO complete run on the
correct data at patch 4** — patch 8 (the only complete correct-data run) destroys modulation via coarse
patching. Get the patch-4/132k/snr-fix run done, then resolve §6 (LWM vs ImageNet) with the equal-width and
fine-tune diagnostics.
