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

## M4 — 2 more seeds per patch size (after M3 validates)  ·  STATUS: TODO (seed-1 pretrains running)
**Goal:** statistical confidence — run 2 additional seeds for each patch size; report mean ± 95% CI.
**Seed-1 (42) pretrains — apples-to-apples (batch 32, mult8_vary corpus, 12k steps, router-ep 15):**
- MAMBA p4/p6/p8: DONE locally (GPU0) → `spectro_mamba_p{4,6,8}_weights/` (3 experts + router each; in-
  training demo-mod probe lifts WiFi 0.70→0.83-0.88, 5G 0.77→0.80-0.82, LTE ~0.88 flat).
- TRANSFORMER p4/p6/p8: DONE locally too (GPU0, RTX 3060 Ti 8GB) via gradient accumulation
  (`--batch-size 8 --accum-steps 4` = eff 32; peak ~5.7GB at p4). Detached driver
  `cluster/run_tf_pretrain_detached.sh` (setsid, idempotent) to survive session teardowns. All 9
  experts + 3 routers saved. demo-mod init→final: p4 LTE .71→.92 / WiFi .71→**.50 (weak — patch-4 bad
  run, redo candidate)** / 5G .70→.79; p6 .75→.92 / .71→.91 / .71→.80; p8 .79→.93 / .79→.92 / .77→.81.
  (Also runnable on cluster: `sbatch ARCH=transformer,PATCH=N`, same recipe via config.env; corpus on HF
  `tomerraviv95/lwm-spectro-deepmimo-mult8vary` private.) Contrastive caveat: with accum, SupCon sees
  8-sample micro-batches (not 32) — MLM unaffected; cluster batch-32 run is the true-parity cross-check.
- Downstream finetune+test (user-run): `PATCH=N MODE=downstream ARCHES=... bash run_patch_study.sh`
  (`--pool meanstd_t`); transformer_synth arm needs the cluster checkpoints pulled local first.
Steps:
- [ ] For each validated patch size, re-run pretrain+downstream with 2 more seeds (seed in name/config).
- [ ] Aggregate mean ± CI per (patch, arch, task); update the plot with error bars.
Run logs: `cluster/logs/m3_mamba_pretrain_p468.log`, `cluster/logs/m4_p{patch}_seed{N}_*.log`
Findings: _(to fill)_

---

## M5 — Push mobility on the DOWNSTREAM task (re-open of M1)  ·  STATUS: DONE (ceiling accepted)
**VERDICT (decision: accept ~0.44, lock `meanstd_t`, document the ceiling).** By elimination, our
mobility ceiling is ~0.44 and the published 0.69 requires a *large + in-domain* pretraining corpus
(their generator) that we don't have and didn't reproduce. Both controllable factors cap at 0.44:
large-synthetic-OOD (M1/M2 sweep) ≈ 0.44 and small-demo-in-domain (M5, 10.5k×15-20ep) ≈ 0.44; only
large+in-domain (their corpus) → 0.688. We DID meet the stated bar — pretraining > random-init on
mobility, leakage-safe (mean 0.359→0.406, meanstd_t 0.416→0.444). **Locked recipe for our arms:
`--pool meanstd_t`** (sweep default + cluster `SPECTRO_POOL`); mobility contrastive is dead (sc_mob
frozen) so leave `--contrast`/SupCon off the mobility head; `time_col` masking retained as a no-help
option. Gap to 0.69 = pretraining-corpus limitation (documented in spectro/README.md). Did NOT pursue:
generator reproduction (option D) — heaviest, deferred.

**Goal:** get transformer/mamba to beat random-init on mobility on a held-out demo TEST split (gains
can be < published 0.69). Continues M1, but with the published side now reverse-engineered.

**Reframe (verified against `spectro/hf_cache/`):** the three "obvious" fixes are dead ends —
(1) demo `data` is `(1,128,128)` *magnitude* float16, so the published model consumes magnitude too
(NOT complex/phase); (2) both sides are z-scored `20·log10` dB (our generator already matches); (3)
their downstream head is `outputs[:,1:,:].mean(dim=1)` and the precomputed `moe_embedding` (a single
mean-pooled 128-d vector) **already scores 0.69 on our 3-class mobility probe** — so mean-pool is NOT
fatal and `meanstd_t` was a symptom fix. Conclusion: mobility is recoverable from a mean-pooled
magnitude embedding IF pretraining bakes it in. Ours doesn't (random-init ≈ pretrained on mobility);
prime suspect = **pretraining-data domain mismatch** (synth-trained mobility features → demo cross-
transfer was 0.331 ≈ chance). Their pretrain masks 0.6 (we used 0.7); their contrastive proj also
mean-pools (`x.mean(dim=1)`).

**Decisive diagnostic (running):** in-domain pretrain — `validate_contrastive_fixed.py
--pretrain-on demo` pretrains experts on demo `train_idx`, fine-tunes the head on that split, tests on
the **disjoint** `test_idx` (backbone never saw it → leakage-free), reports random-init vs pretrained.
Run with `--pool mean` (apples-to-apples with the published proof) for both arches.
  - trained >> random (mean pool)  → objective+representation OK; whole gap is data-domain → fix the
    generator / mix demo-domain data into pretraining.
  - trained ≈ random              → the pretext is too weak for mobility → time-column masking +
    weak-supervised Doppler/speed aux head.
Run logs: `cluster/logs/m1_indomain_{arch}_{pool}.log`
Findings:
- **[transformer, in-domain demo, MEAN pool] random-init 0.359 → pretrained 0.406 (+0.047).** KEY:
  this DISPROVES "pure data-domain" — even in-domain our recipe only reaches 0.41 vs published 0.69,
  so the **objective is also too weak**, not just the corpus. `sc_mob` stayed FROZEN at ~1.84 the whole
  run (no gradient) while `mlm` (0.54→0.33) and `sc_mod` (1.6→0.85) descended → the +0.047 came from
  **MLM alone**; the mobility contrastive contributes nothing because it mean-pools (temporal Doppler
  signal washed out before SupCon). Two fixes to test: (a) temporal-aware pooling for `sc_mob` so it can
  engage, (b) time-column masking so MLM is forced to model temporal dynamics.
- **[transformer, in-domain demo, meanstd_t pool + mob-pool meanstd_t] random 0.416 → pretrained 0.444
  (+0.029).** (1) the `meanstd_t` readout is a real win — lifts BOTH random (0.359→0.416) and pretrained
  (0.406→0.444); keep it. (2) `sc_mob` STILL froze at ~1.81 even with temporal pooling on the mobility
  head → the mobility contrastive is a genuine dead end (mobility too weakly separable for SupCon to
  bootstrap). All gains come from MLM, so the remaining lever is making MLM mobility-aware →
  **time-column masking** (implemented `mask_mode='time_col'` in spectro_patchify/build_masked_tensors +
  pretrain_expert + validate `--mask-mode`; unit-tested: 19/32 cols × 32 freq = 608 masked positions).

| in-domain recipe (transformer) | random-init | pretrained |
|---|---|---|
| mean pool, random mask | 0.359 | 0.406 |
| meanstd_t pool, random mask | 0.416 | 0.444 |
| meanstd_t pool, **time_col** mask | 0.416 | 0.433 |
| **published moe_embedding (mean-pool)** | — | **0.688** |

**CONCLUSION (elimination):** published `moe_embedding` confirmed **0.688** under our exact held-out
harness (1575 test). All our in-domain recipes cap at ~0.44. Ruled OUT as the cause: representation
(both magnitude), dB (both), readout (meanstd_t helps a bit; pub's 0.69 is plain mean-pool), mobility
contrastive (frozen/dead), time-column masking (no help — model interpolates a missing column without
encoding Doppler rate). Remaining suspect by elimination = **pretraining strength**: their corpus is
large + diverse (many cities × FFT × balanced mobility) + in-domain + ~100 epochs; their pretraining
injects +0.33 mobility into a MEAN embedding (0.36→0.69) vs our +0.05 (0.36→0.41) — ~7× gap. Our
in-domain test was only 10.5k demo × 15–20 ep. `time_col` masking kept (option, documented no-help).
NEXT FORK: (a) test scale/epochs in-domain, or (b) practical fix = pretrain synthetic+demo-mix at scale
w/ meanstd_t and re-run the sweep (transformer/mamba vs random on held-out demo).

---

## M6 — In-domain eval on held-out cities (clean lift over random-init)  ·  STATUS: DONE (seed 1)
**Goal:** measure honest pretraining lift in-domain (pretrain & eval share our DeepMIMO generator),
removing the cross-generator confound of the demo tasks. Eval set = **held-out cities** asu_campus(BS1)/
boston5g(BS2)/o1(BS3) — disjoint from the 20 pretrain `city_*`, same recipe (mult8/vary, seed 1234),
6000 samples, leakage-free (never in pretraining as inputs OR contrastive labels). Frozen embedding ->
MLP head: train fits / val early-stops / test (899) reported. `--cities name:bs_idx` (gen) + `--synth-dir`
(sweep) + chained driver `cluster/run_indomain_eval.sh`. Gen needed per-city BS + `--batch 2`
(+expandable_segments) — o1 path count × mult8 OOM'd at batch 8 on the 8GB card.

**RESULT (acc @100%, held-out test; chance mod .20 / snr .14 / mob .33):**
| patch | arm | mod | snr | mob |
|---|---|---|---|---|
| p4 | mamba / TF-ours / rand-init | .398/.207/.264 | .953/.882/.881 | .557/.501/.492 |
| p6 | mamba / TF-ours / rand-init | .331/.258/.211 | .924/.924/.821 | .525/.499/.481 |
| p8 | mamba / TF-ours / rand-init | .406/.293/.234 | .919/.909/.855 | .493/.471/.447 |

**LIFT over random-init — mamba p4/p6/p8:** mod **+.135/+.120/+.172**, snr +.072/+.103/+.063,
mob +.066/+.044/+.046. TF-ours: mod −.057/+.047/+.059, snr +.001/+.103/+.053, mob +.009/+.019/+.023.
**Findings:** (1) IN-DOMAIN pretraining lift is clear & consistent for mamba on all 3 tasks, and MUCH
larger than the cross-generator demo (mamba mod +.12–.17 in-domain vs ~+.02 demo) → the small demo lift
was domain mismatch, NOT weak pretraining. (2) mobility lifts in-domain too (+.04–.07, abs ~.5 vs demo
.42) → the M5 ceiling was partly the cross-generator gap. (3) mamba > transformer in-domain and
generalizes cross-environment far better (TF-ours lift weak/mixed, even −.06 p4 mod) — favorable for the
Mamba MoE. Caveats: modulation is hard here (multipath magnitude → ~.2–.4, read as lift not absolute);
single seed (confirm with M4). Per-arm radar charts written; no combined `_heldout` line plot yet.

---

## M7 — Why modulation is near-chance, and the fix  ·  STATUS: ROOT-CAUSED + FIX VALIDATED (not integrated)
**Question:** in-domain modulation was ~chance (M6), unlike the demo's 0.96. Why, and how to fix.
**ROOT CAUSE (proven):** our spectrograms are `|STFT|` of the **time-domain OFDM waveform**. OFDM sums
52–624 subcarriers → by CLT the time signal is ~Gaussian regardless of constellation, so BPSK and
QAM256 are statistically identical in magnitude. Probe (raw spectrogram → mod, chance 0.20):
  - OURS (held-out, all features): ~chance EVEN AT 25 dB (spatial 0.21, amp-hist 0.22).
  - DEMO (their gen): mod 0.957 @25 dB / 0.92 @15 dB / 0.63 @−5 dB (spatial mean-pool) → demo encodes it.
So it is a **data-representation problem in our generator**, not SNR/difficulty (user was right).
**FIX (validated, partial):** generate from the **demodulated received resource grid** `|Y[k,n]|`
(subcarrier×symbol), where per-subcarrier constellation amplitude is visible. Added
`grid_mag_to_spectrogram` + generator `--repr grid` (OFDMDemodulator). Findings:
  - Full-res grid, linear `|Y|` amp-hist (WiFi 25 dB, 4-mod): **0.66** vs 0.28 for time `|y|`. ✓ grid encodes mod.
  - 128×128 grid with **bilinear** resize: mod back to ~chance — resize AVERAGES neighbouring REs, washing
    out the constellation. Switched to **nearest** resampling (samples native REs, no averaging).
  - 128×128 grid-nearest, dB+z-score: mod rises with SNR (−5→25 dB: 0.25→0.47); aggregate ~0.35.
    Better than chance & SNR-sensible, but FAR below demo 0.96 and the full-res 0.66.
**OPEN:** (a) more representation tuning to close the gap — linear `|Y|` (dB compresses the amplitude
levels), native 128×128 crop instead of subsample, maybe equalization; (b) the pretrained MoE arms were
trained on STFT, so lifting the actual MoE mod numbers needs **regenerating the corpus + re-pretraining
both arches on `--repr grid`** (~2 days). Stopped the loop here to get a decision before that recompute.
Probes/logs: cluster/logs/m7_grid_validate.log; eval sets spectro/outputs/spectro_eval_heldout_cities_grid,
spectro_eval_grid_nn. snr/mob also present in grid (snr 0.54, mob 0.36 patch_std @nearest).

---

## M8 — Grid re-pretrain: modulation SOLVED (representation fix integrated)  ·  STATUS: DONE
**Committed to the grid re-pretrain (M7 fix).** Regenerated corpus (40k) + held-out-cities eval (6k) in
`--repr grid` (nearest resample), re-pretrained BOTH arches × p4/6/8 on grid (`_grid` weights, apples-to-
apples recipe; STFT weights preserved), ran the in-domain grid sweep, regenerated the score-vs-patch plot
(`spectro_score_vs_patch_heldout_grid.png`). Driver `cluster/run_grid_repretrain.sh`, log m8_grid_repretrain.

**RESULT (in-domain held-out cities, grid, acc @100%; chance mod .20 / snr .14 / mob .33):**
| patch | arm | mod | snr | mob |
|---|---|---|---|---|
| p4 | mamba / TF-ours / rand-init / raw | **.504**/.454/.284/.197 | .911/.914/.789/.341 | .327/.345/.326/.350 |
| p6 | mamba / TF-ours / rand-init | **.467**/.413/.253 | .904/.893/.727 | .359/.328/.330 |
| p8 | mamba / TF-ours / rand-init | **.455**/.402/.227 | .882/.878/.715 | .355/.343/.347 |

**LIFT over random-init — mamba p4/p6/p8:** mod **+.220/+.215/+.228**, snr +.122/+.177/+.167,
mob +.001/+.029/+.008. TF-ours mod +.170/+.160/+.175.
**Findings:** (1) MODULATION SOLVED — on grid it is genuinely learnable (raw at chance .20, backbone
reads it) and shows the LARGEST, cleanest pretraining lift of any task (mamba **+0.22** consistently),
vs the STFT representation where mod carried ~no real signal. (2) SNR strong with a clear lift (+.12–.18).
(3) TRADE-OFF: grid REGRESSED mobility to ~chance (.33) vs STFT+meanstd_t's ~.50 (M6) — the two
representations are complementary: the resource grid preserves the constellation (modulation) but its
nearest-subsampled symbol axis loses the temporal-coherence (Doppler) that the STFT captured. (4) mamba ≥
transformer on modulation across all patches (+.05) and matches on SNR — the cross-environment
generalization edge holds. Single seed. NEXT IDEAS (if wanted): a 2-channel [grid | STFT] representation
to get mod AND mobility; or per-task representation. Plot title fixed to say grid.

---

## M10 — Modulation ceiling: fine-tune + per-SNR (not a data/capacity limit)  ·  STATUS: DONE
Tested the two levers for lifting modulation above the frozen-probe ~0.50 (grid mamba p4, in-domain):
- **Fine-tune the backbone** (spectro_finetune.py, oracle routing, backbone+head, val early-stop):
  TEST **0.452** < frozen 0.504 -> fine-tuning HURTS (overfits ~4.2k samples). More downstream data also
  flat (M's sweep curve saturates by ~1700 samples). So ~0.50 is NOT capacity/data-limited.
- **Per-SNR** breakdown (frozen): -5dB .22 (chance) / 0 .37 / 5 .51 / 10 .50 / 15 .62 / 20 .58 / 25 .59.
  Modulation is unreadable at low SNR (drags the all-SNR aggregate) and plateaus ~0.6 even at high SNR
  because the DeepMIMO multipath channel distorts the constellation in the magnitude grid.
**Verdict:** modulation ~0.50 (all-SNR) / ~0.6 (high-SNR) is an SNR + multipath physics ceiling, not a
model/data limit. Report it per-SNR. `spectro_finetune.py` added (reusable fine-tune arm).

## M9 — Dual [STFT|grid]: all three tasks solved together  ·  STATUS: DONE
Committed to the 2-channel [STFT(ch0)|grid(ch1)] representation (--repr grid_stft) so modulation (grid
channel: constellation) and mobility (STFT channel: Doppler with data averaged out) coexist without
competing. Regenerated corpus(40k)+held-out-cities eval(6k) in grid_stft, re-pretrained BOTH arches x
p4/6/8 (_gridstft weights, apples-to-apples recipe), swept, plotted (spectro_score_vs_patch_heldout_gridstft.png).

**RESULT (in-domain held-out, acc @100%; chance mod .20/snr .14/mob .33):**
| patch | arm | mod | snr | mob |
|---|---|---|---|---|
| p4 | mamba/TF/rand | .610/.525/.364 | .941/.920/.867 | .499/.376/.463 |
| p6 | mamba/TF/rand | .536/.445/.267 | .941/.910/.798 | .430/.424/.418 |
| p8 | mamba/TF/rand | .573/.440/.283 | .899/.872/.764 | .427/.423/.396 |
LIFT over rand-init (mamba): mod +.25/+.27/+.29, snr +.08/+.14/+.14, mob +.04/+.01/+.03.

**Findings:** all three tasks work in ONE model. (1) Modulation SOLVED and even higher than grid-only
(mamba .61 p4 vs .50 grid-only) with a large lift (+.25-.29) — the STFT channel adds complementary
signal. (2) SNR strong (.90-.94, lift +.08-.14). (3) Mobility RECOVERED to ~.43-.50 (vs grid-only's
chance .33) — the STFT channel restores Doppler; pretraining lift is small (rand-init already ~.4-.46
from the STFT channel + meanstd_t readout, consistent with M5/M6: mobility is representation-driven, not
pretraining-driven). mamba > transformer on modulation across all patches (+.05-.13). p4 mamba is the best
all-rounder (.610/.941/.499). Reframing: not "STFT vs grid" — it's two STFT alignments (generic->Doppler,
symbol-aligned/grid->constellation), consistent with the paper's STFT pipeline. Single seed.

### Conventions
- Run logs live under `cluster/logs/` with the `m{N}_` prefix shown above.
- Checkpoints carry patch (and later seed) in the dir name; configs/manifests record patch+seed.
- Update each mission's STATUS (TODO → IN PROGRESS → DONE) and Findings as we go.
