# LWM-Spectro — handoff: "why doesn't pretraining clearly WIN downstream?"

Written 2026-07-22 for a fresh agent to continue. Companion to `spectro/PROVENANCE.md` (data/run map) and
the memory `spectro-domain-contract.md`. Branch `feat/spectograms`.

## The open problem (what "win" means and why we're not there)

We pretrain per-protocol MoE backbones (Mamba **and** Transformer, arch-parameterized) with masked
spectrogram modeling (MLM) on a synthetic DeepMIMO-spectrogram corpus, then evaluate downstream on
**modulation (5-cls)** and **joint SNR/Doppler (21-cls)** (protocol is trivially ~1.0 = the router).
Goal: pretraining should give the LWM a clear, decisive advantage over no-pretraining baselines
(DeepCNN, ResNet, random-init, raw). **It does not yet.** Fairly measured it *ties* frozen ResNet and
only modestly beats random init — and absolute accuracies are low (mod ~0.44, SNR ~0.28). The user
(correctly) expects a foundation model to WIN, not tie.

## What is ALREADY RULED OUT — do NOT re-investigate these

1. **Finetune recipe is not the cause.** LoRA (≤5% adapters) vs CE-only partial-FT (unfreeze top blocks)
   vs frozen: on the same 5k backbone these were ~equal; partial-CE even slightly hurt mamba. Finetuning
   *hurts mamba* (frozen-seq 0.43 → partial-FT 0.39) and mildly *helps the transformer*. Current pipeline
   default: **mamba frozen (head-only), transformer light 2-block FT** (`MAMBA_TUNE_LAST_N=0`,
   `TF_TUNE_LAST_N=2`).
2. **Pretraining under-convergence is fixed.** 5k steps left the transformer still rising; now 15k-step
   cap + early-stop on `probe_val` (save best). Transformer improved 0.41→0.43 mod — real but small;
   mamba converges by ~5–6k and is unchanged. Not the bottleneck.
3. **The "DeepCNN >> LWM" gap was a regime confound, not representation.** DeepCNN was trained
   END-TO-END (0.52 mod); the *frozen random* DeepCNN scores only **0.40**. So its edge was full-backbone
   adaptation on the task, not a better feature space.
4. **A probe bug (now fixed).** The frozen linear probe lacked `StandardScaler`; the LWM's 128-d features
   under-probed (0.38 → **0.44** once standardized). Downstream heads have BatchNorm so were unaffected.
5. **Pretraining DOES learn useful features.** In-corpus modulation probe rises during pretraining
   (transformer LTE 0.31→0.53 over 15k steps). Under a fair frozen+standardized linear probe the
   pretraining lift is real: **transformer +0.10 mod / +0.09 SNR**; mamba small (+0.02 mod / +0.06 SNR)
   because its random bidirectional-SSM init already extracts these well.

## Current FAIR results (all frozen, standardized linear probe, N=6000 alluser15 in-dist, acc @600)

| frozen extractor | modulation | SNR/Doppler |
|---|---|---|
| LWM mamba pretrained | 0.44 | 0.28 |
| LWM transformer pretrained | 0.44 | 0.26 |
| LWM mamba random | 0.41 | 0.26 |
| LWM transformer random | 0.34 | 0.17 |
| ResNet-18 frozen (ImageNet) | 0.44 | 0.27 |
| DeepCNN random (frozen) | 0.40 | 0.28 |
| raw patches | 0.26 | 0.11 |

End-to-end DeepCNN (for reference, NOT frozen): 0.52 mod / 0.32 SNR. Random-init **Mamba** with the
Conv1d-**sequence** head (downstream, not probe) hits ~0.51 mod — the seq head extracts a lot from random
mamba (0.37 mean-pool → 0.48 seq), which is why the downstream figures looked like "pretraining loses."

## Leading hypotheses for why it doesn't WIN (ranked — start here)

1. **Representation ceiling: magnitude spectrograms.** (Biggest lever.) Modulation is ~invisible in
   `|STFT|` of the OFDM waveform (CLT → time signal ~Gaussian regardless of constellation; the M7
   finding in `MISSIONS.md`). The task caps ~0.5 and a strong *random* SSM already reaches it, so there's
   no headroom for pretraining. **Try: complex / IQ or resource-grid `|Y[k,n]|` representation** where the
   constellation is genuinely hard to extract without learned features (the paper's domain). This likely
   requires regenerating corpus+eval and re-pretraining, but it's the path to a *large* pretraining win.
2. **Input fidelity / normalization.** The LWM sees 4×4 patches with **per-sample z-score**
   (`spectrogram_patchify(normalize=True)`), i.e. a compressed, scale-removed view; the CNNs see full-res.
   Per-sample z-score may destroy the absolute-amplitude cue SNR needs. **Try: probe with `normalize=False`,
   larger/other patchings, or CLS vs mean-pool vs `meanstd_t`.** Cheap.
3. **Pretraining corpus/objective.** MLM on synthetic DeepMIMO spectrograms — maybe not diverse/large
   enough, or MLM isn't the objective that separates these classes. **Try: more data/diversity (the M11
   all-user corpus), or the paper's supervised-contrastive fine-tuning objective, or a harder mask.**
4. **Downstream target-domain mismatch.** The in-dist eval (same cities/generator as pretraining) is
   saturable from random features; the *real* demo set is the true target but is **1-channel magnitude**,
   architecturally incompatible with the 2-channel `[STFT|grid]` backbones. **Try: pretrain a 1-channel
   backbone and evaluate on the real demo**, or make an in-domain-to-real bridge.

## Concrete next experiments (ranked, with how-to)

- **(A) Confirm the ceiling is the problem.** Probe the *raw eval* per-SNR and check whether ANY method
  (incl. the published `moe_embedding` on the real demo, which scores 0.96 mod) beats ~0.5 on our synthetic
  data. If nothing does, the representation is the ceiling → go to (B).
- **(B) Switch representation to grid/complex** (M7/M8/M9 in `MISSIONS.md` built `--repr grid` and
  `--repr grid_stft`; the current corpus is already `grid_stft`). Consider a **complex/IQ** eval where
  modulation is separable, regenerate a small eval, and re-probe pretrained-vs-random. This is where a
  foundation model should decisively win.
- **(C) Normalization/pooling ablation** (cheap, local): re-run the frozen probe with `normalize=False`,
  `pool ∈ {mean, cls, meanstd_t}`, and different patch sizes; see if SNR/mod jump.
- **(D) Scale/diversity**: pretrain on the larger all-user corpus (M11) and re-probe.

## How to reproduce / run (this machine + cluster)

- **Fair frozen probe (the headline metric)** — `spectro/scripts/spectro_probe.py`, unified over
  `--extractor {lwm, deepcnn, resnet18/50/effnet/mobilenet, raw}`, StandardScaler + logreg:
  ```
  .venv/bin/python spectro/scripts/spectro_probe.py --extractor lwm --arch mamba --patch 4 \
    --weights-suffix alluser_15k --synth-dir spectro/outputs/spectro_eval_alluser15_gridstft \
    --run-tag indist15 --sample-counts 50 100 200 400 600 --seeds 42 43 44           # + --random control
  .venv/bin/python spectro/scripts/spectro_probe.py --extractor deepcnn --synth-dir <eval> ...  # frozen baselines
  ```
- **Rough debug scripts** used to reach the above are in `spectro/scripts/debug/` (`debug_frozen.py`
  mean-pool pretrained-vs-random; `debug_seqhead.py` Conv1d-seq frozen; `debug_crossenv.py` in-dist vs
  cross-env lift; `debug_apples.py` all-frozen incl. frozen-CNN). RAM-safe: they subsample to ≤6000 and use
  mean-pool (the full 23k sequence tensor OOMs this WSL node — keep N small / mean-pool).
- **Cluster full run**: `cluster/01_pretrain_finetune.sbatch` (ARCH=mamba|transformer; pretrain 15k+early-stop
  → per-arch FT → frozen probe + random control), `cluster/02_baselines.sbatch` (deepcnn/random_init/raw/
  resnet18 + a frozen linear-probe of `PROBE_BASELINES`), `cluster/03_publish.sh` (login-node upload). Knobs
  in `cluster/config.env`. mamba per-patch to fit 24h; transformer both patches OK.

## Data / checkpoints / HF map

- Pretrain corpus: `lwm-spectro-alluser/corpus` (132k, 85%-user, 20 cities, dual `[STFT|grid]`, patch 4).
- Evals: `lwm-spectro-alluser/eval` (23k, in-dist 15%-user) + `lwm-spectro-gridstft/eval` (6k, held-out
  cities, cross-env). Both present locally under `spectro/outputs/spectro_eval_*_gridstft`.
- Checkpoints on HF `tomerraviv95/wimamba-spectro-ckpts`: `spectro_{mamba,transformer}_p{4,8}_alluser_15k_weights`
  (current, early-stopped) and `..._alluser_b128_weights` (older 5k). Only-copy caveats in PROVENANCE.md.
- Results on HF dataset `tomerraviv95/lwm-spectro-results` under `results/submission_*`.

## Key files
- `spectro/scripts/spectro_pretrain_real.py` — step-based MLM pretrain (15k cap, early-stop on probe_val, ETA logs).
- `spectro/scripts/spectro_finetune.py` — downstream FT (`--tune-mode partial|lora`, `--tune-last-n`, CE-only default).
- `spectro/scripts/spectro_probe.py` — **the fair frozen linear-probe metric** (use this to judge representations).
- `spectro/scripts/spectro_moe.py` / `spectro_backbones.py` / `shared/mamba_layers.py` — MoE + arch (Transformer uses
  runtime SDPA patch for batch scaling; Mamba `use_fast_path` threaded for LoRA).
- `spectro/scripts/spectro_data.py` — loaders + 70/10/20 stratified split (`val_frac`/`test_frac`), TASKS.
- `spectro/scripts/spectro_train_heads.py` — baseline arms (deepcnn e2e, imagenet, random_init, raw).
- `spectro/scripts/debug/` — the throwaway probes from this session.

## One-line conclusion to build on
Pretraining works and the LWM representation is competitive with strong frozen CNNs — but the **magnitude-
spectrogram tasks have a low ceiling a random SSM already reaches**, so pretraining can't show a decisive
win. The most promising path to "winning" is a **representation where the task is genuinely hard from
scratch** (complex/IQ or resource-grid), then re-run the fair frozen probe.
