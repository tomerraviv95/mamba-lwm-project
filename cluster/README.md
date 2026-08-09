# Cluster runbook — LWM-Spectro single-carrier study

`REPR=sc` is the default and the only live study. Everything below assumes you are in the repo root.

## 0. One-time setup

```bash
bash cluster/setup_env.sh            # env + deps (uninstalls causal_conv1d if its CUDA ext fails)
cp cluster/secrets.env.example cluster/secrets.env && $EDITOR cluster/secrets.env   # HF_TOKEN, WANDB
```

## 1. Pull the data (LOGIN node — compute nodes have no internet)

```bash
bash cluster/download_data.sh                # skips dirs that already verify
FORCE=1 bash cluster/download_data.sh        # re-fetch everything
```

Verification compares the **local manifest against the REMOTE one** (`n_samples` + shard count), not
just local file presence. That matters: a stale manifest sitting on top of newer shards passes every
local check, because its shard list is a strict subset of what is present — the failure that
silently truncated a 132k corpus to 30k and cost a multi-day run.

## 2. Preflight (seconds — do not skip)

```bash
bash cluster/00_preflight.sh
```

Checks every shard listed in each manifest exists and is non-trivial; that the generation recipe
(`waveform / sc_win / sc_norm / channels`) is **identical** across corpus and both evals; and warns
about existing checkpoints that the `done3` guard would silently reuse. The recipe check exists
because a representation mismatch between pretraining and eval is otherwise undetectable — the old
`grid_stft` and `grid_complex` tensors have identical shapes, so a mismatched model loads cleanly
and produces noise.

## 3. Run

```bash
STUDY_SEEDS=1 bash cluster/run_study.sh      # start with ONE seed; add seeds once direction is confirmed
```

Submits `10_pretrain_grid` (array over arch × patch × seed), then `11_downstream_grid` with
`--dependency=afterany`. **Not `afterok`** — one timed-out pretrain task would otherwise cancel every
downstream task, including combos whose checkpoints finished. That is what killed an earlier run.
Per-combo checkpoint guards are the real gate.

## 4. Publish + plot

```bash
bash cluster/12_publish_study.sh
python spectro/scripts/plot_from_csv.py --hf-repo $HF_USER/lwm-spectro-results --variant scmax --metric macro_f1
```

`--metric macro_f1` is deliberate: the plotter defaults to `accuracy`, and the study's primary
metric is macro-F1.

---

## Data

| dataset | samples | unique (user, BS) links |
|---|---|---|
| `spectro_corpus_scmax_s1` | 1,327,020 | 442,340 |
| `spectro_eval_scmax_indist_s1` | 78,109 | 78,109 |
| `spectro_eval_scmax_heldout_s1` | 6,000 | — |

20 LWM cities × **all 3 BS positions** × all users × 3 draws. The 85/15 user split is keyed on the
raw DeepMIMO grid index, so it is BS-independent — a UE with no link to BS1 may have one to BS3, and
splitting each BS's filtered list separately would put the same physical location in train for one BS
and eval for another. Verified on the generated data: **0 user overlap, 0 link overlap, 0 city
overlap** with the held-out set.

## Knobs that matter (`config.env`, `REPR=sc` block)

| var | default | why |
|---|---|---|
| `SC_PATCHES` | `8` | seq 257 vs 1025 → ~6× cheaper; the audit showed patch was never the cause of the modulation failure |
| `SPECTRO_W_CONT` | `0.0` | reconstruction-only, per paper eq. 21. SupCon was a crutch for a dead MLM objective and trained on the downstream label sets |
| `SC_PER_EXPERT` | `1` | one expert per invocation: the full corpus is ~44 GB in host RAM, one protocol is ~15 GB |
| `STUDY_ROUTING` | `oracle` | applied to pretrained **and** random-init arms — they must match or the lift is confounded |
| `STUDY_PROJECT_DIM` | `128` | equal head width; MobileNet/ResNet otherwise get a 347k/331k head vs the LWM's 232k |

## Failure modes this pipeline now guards against

- **Silent partial upload** — `hf_upload_sc.py` sends shard-by-shard with retries, is resumable, and
  writes `manifest.json` **last**, so an interrupted upload is detectably incomplete rather than
  plausibly complete.
- **Silent arm failure** — `11_*.sbatch` captures `PIPESTATUS[0]` (not `tee`'s status), collects
  failures and exits non-zero. Previously a crashed arm still produced a CSV and exited 0.
- **Wrong plot command** — `12_publish_study.sh` prints `--variant $STUDY_VARIANT` (it used to print
  `$STUDY_HEAD`, which plots a different study).
- **Stale checkpoints** — `00_preflight.sh` lists any checkpoint dir the `done3` guard would reuse.
  Bump `STUDY_SUFFIX` or `rm -rf` for a new recipe.

## Open issues — do not present results without addressing these

1. **No error bars.** `--seeds` is not passed by `11_downstream_grid.sbatch`, so `score_std = 0`.
2. **Lift is ~zero at 2–5 samples/class and grows with N** — the opposite of the study's
   "largest gap few-shot" claim. Seen in two independent runs.
3. **CV arms lack ImageNet mean/std normalization** and use a bilinear 128→224 resize. Both handicap
   them; fix before claiming a win, and add a *fine-tuned* ImageNet arm (what the paper compares to).
4. **No `--layer` flag** — only the final layer is probed, the standard worst case for a
   reconstruction-pretrained encoder. Measured: mamba modulation peaks at L7 (0.370) vs L12 (0.348).
5. **Masks are built once** and frozen for the whole run.
6. The in-dist eval grew 23k → 78k, so absolute numbers are **not** comparable to the earlier
   `sc2`/`sc400k` runs. Comparisons within a run remain valid.
