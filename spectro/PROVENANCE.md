# Spectro run provenance — what ran on what

Single source of truth for the current LWM-Spectro results. Update this whenever the corpus,
eval sets, or run recipe changes. If a result's provenance isn't captured here, treat it as unknown.

## Current results (HF dataset `tomerraviv95/lwm-spectro-results`, under `results/`)

Produced by `cluster/pretrain_eval.sbatch` (frozen LWM + LWM-FT) and `cluster/run_baselines.sbatch`
(deepcnn / random_init / raw). Config: `cluster/config.env`.

- **Representation:** dual `[STFT | grid]` (2-channel), patch 4, `--symbol-mult 8 --vary-speed`.
- **Pretraining:** reconstruction-ONLY (paper eq. 21, `W_CONT=0`), 12k steps/expert, seed 42, single
  pretraining seed. Downstream head seeds = 42/43/44 (head-init/subsample only).
- **Downstream:** paper protocol — residual 1-D CNN head over the token sequence, macro-F1 (primary),
  few-shot axis = samples/class `{2,4,8,16,32,64,128,256}`.
- **Tasks:** `modulation` (5), `snr_doppler` (21 = 7 SNR x 3 mobility), `protocol` (3, ~1.0 = router job).

### Data map

| role | HF location | size | cities | users | used for |
|---|---|---|---|---|---|
| pretrain corpus | `lwm-spectro-alluser/corpus` | 132,748 | 20 `city_*` | 85% split (seed 777) | ALL pretraining (both arches) |
| in-dist eval | `lwm-spectro-alluser/eval` | 23,426 | same 20 `city_*` | disjoint 15% split | `alluser15` result column |
| cross-env eval | `lwm-spectro-gridstft/eval` | 6,000 | asu_campus / boston5g / o1 | n/a | `heldoutcities` result column |

`lwm-spectro-gridstft/corpus` (M9 40k) is NOT used by the current run — only its `eval/`.

### Result-dir naming: `submission_spectro_{arm}_p{patch}_heldout_{weights_suffix}_{eval_tag}`

- `{arm}` — mamba | transformer_synth | mamba_ft | transformer_synth_ft | deepcnn | random_init | raw
- `_heldout` — evaluated on a synthetic corpus (not the demo set)
- `{weights_suffix}` = `alluser` — **pretrained on** the alluser85 corpus
- `{eval_tag}` = `alluser15` (in-dist) | `heldoutcities` (cross-env) — **evaluated on**

  ⚠️ The two "alluser"s differ: `_alluser` = training corpus; `_alluser15` = eval split.

## Known gaps / caveats

- **The alluser recon-only checkpoints are NOT backed up — they exist ONLY on the cluster.**
  `pretrain_eval.sbatch` step 2 pushes them to `wimamba-spectro-ckpts`, but that push failed and the job
  continued (push is non-fatal). The HF repo was emptied of stale June weights (2026-07-18) and now holds
  only `.gitattributes`, waiting for a re-push. Local `pretrained_models/` was DELETED (2026-07-18) — it
  held only the old M9 `_gridstft` weights, not these. **Pull `spectro_{mamba,transformer}_p4_alluser_weights`
  off the cluster and re-push to HF before clearing cluster scratch.**
- **Transformer-convergence check + W&B:** the pretraining curves are on the cluster as
  `spectro/outputs/pretrained_models/spectro_{arch}_p4_alluser_weights/{LTE,WiFi,5G}_steps.csv`
  (columns `mlm,val_mlm,probe_train,probe_val,probe_gap,lr`) and in `cluster/logs/pretrain_eval_{arch}_alluser.log`.
  W&B is OFFLINE (`cluster/logs/wandb/offline-run-*`, runs `pretrain-{mamba,transformer_synth}-alluser`) —
  online only if `wandb sync` was run on the login node. Grab the `*_steps.csv` in the same trip as the ckpts.
- **Convergence CHECKED (2026-07-18) — transformer is NOT under-trained.** From the `*_steps.csv`
  (now on HF): transformer `val_mlm` is flat/healthy (~0.37 LTE, 0.45 WiFi, 0.38 5G), on par with mamba,
  lr schedule ran cleanly (warmup→cosine→1e-8), no divergence. So the mamba edge is NOT a training
  artifact. The mamba modulation advantage appears DURING pretraining in the frozen in-corpus probe:
  probe_val (mod) mamba vs transformer = LTE 0.55 vs 0.43, WiFi 0.74 vs 0.72, 5G 0.44 vs 0.43. Mechanism:
  both start equal at step 2000 (0.47 LTE); mamba's modulation-separability RISES with MLM steps while the
  transformer's DEGRADES (0.47→0.43) — the SSM retains constellation info under masked reconstruction
  better than attention does at this width (d=128). Real, mechanistically-explained effect.
  Remaining caveats: single pretraining seed; narrow d=128 transformer (a wider one may close it);
  modulation physically weak in this data (M7) so absolute numbers are low.

## Superseded / deleted

- Deleted HF datasets (2026-07-18): `lwm-spectro-deepmimo-mult8vary` (STFT-only 40k),
  `lwm-spectro-deepmimo` (magnitude "big") — old intermediate corpora, unreferenced.
- Deleted local (2026-07-18): `spectro/outputs/submissions/` and `spectro/outputs/plots/` — stale M9
  3-task artifacts, superseded by the HF results repo.
</content>
