# Running the LWM-Spectro pipeline on BGU-HPC

Reproducible Slurm workflow with a clean **compute / login** split:

1. **`01_pretrain_finetune.sbatch`** (GPU, per arch) — pretrain a spectro MoE backbone (Transformer or
   Mamba) with **reconstruction-only** masked spectrogram modeling, then **LoRA-finetune** it on the
   downstream task. Runs for every patch in `PATCHES`. Saves results **locally**.
2. **`02_baselines.sbatch`** (GPU) — the **no-pretrain** baselines (Deep CNN end-to-end, random-init MoE,
   raw patches, frozen ResNet-18) on the same task/split. Saves results **locally**.
3. **`03_publish.sh`** (LOGIN node) — uploads all local checkpoints + results to HF + W&B.

**Downstream protocol:** one dataset (`DOWNSTREAM_DIR`) is split **70/10/20** (stratified by protocol):
train on **N ∈ {50,100,200,400,600} TOTAL samples** drawn from the 70% pool, validate on 10%
(early-stop), test on the fixed 20%. **3 seeds** (42/43/44). Tasks: **modulation** + **joint SNR/Doppler**
(+ protocol, ~router). Metric: **macro-F1** (primary) + accuracy. Head: residual **1-D CNN**.

**LoRA finetune:** the backbone is frozen and adapted with LoRA whose **adapter params are capped at
`LORA_BUDGET` (5%) of the backbone** (the task head is trainable on top, not counted); the rank is
auto-picked to fit. LoRA is a **weight parametrization** (`shared/lora.py`) so it also adapts the Mamba
SSM projections the fused kernel reads via raw `.weight` — the finetune Mamba is built with
`use_fast_path=False` so the SSM slow path is exercised.

**Why the split lives on `DOWNSTREAM_DIR`:** default is the in-distribution 15%-user eval
(2-channel `[STFT|grid]`, matching the pretrained backbones; the real 1-channel demo set is
architecturally incompatible). Point `DOWNSTREAM_DIR=$EVAL_XENV_DIR` (+ `DOWNSTREAM_TAG=xenv`) for the
cross-environment (held-out-cities) cut.

**Design:** GPU compute in `sbatch` jobs; **all internet I/O on the login node** (compute nodes have no
usable outbound internet). The HF **xet** backend is disabled everywhere (`HF_HUB_DISABLE_XET=1`).

## 0. One-time setup (login node)

```bash
ssh <bgu_user>@slurm.bgu.ac.il
git clone <this-repo> ~/lwm-competition-2025 && cd ~/lwm-competition-2025
git checkout feat/spectograms
cp cluster/secrets.env.example cluster/secrets.env    # edit: HF_TOKEN=hf_... and WANDB_API_KEY=...
bash cluster/setup_env.sh                             # uv sync (from uv.lock) + mamba-ssm + causal-conv1d
```
`setup_env.sh` builds the env from the committed `uv.lock` at `$REPO_ROOT/.venv`, then builds
`mamba-ssm 2.3.0` + `causal-conv1d v1.6.2.post1` (`--no-build-isolation`). Use `cuda/12.8` (or `12.4`)
for the build — **never** `cuda/13`.

## 1. Download data (login node)

```bash
bash cluster/download_data.sh
```
Pulls the 85%-user pretrain corpus + 15%-user eval (`lwm-spectro-alluser`) and the held-out-cities eval
(`lwm-spectro-gridstft`, eval only). Idempotent; verifies each `manifest.json`.

## 2. Pretrain + LoRA-finetune (GPU jobs, per arch)

```bash
sbatch --export=ALL,ARCH=mamba       cluster/01_pretrain_finetune.sbatch
sbatch --export=ALL,ARCH=transformer cluster/01_pretrain_finetune.sbatch
squeue --me
```
Each job loops `PATCHES` (default `4 8`): pretrains recon-only (skips if the checkpoint dir already has
3 experts + router) → LoRA-finetunes on `DOWNSTREAM_DIR` → writes `submission_*` dirs locally. Knobs in
`config.env`: `PATCHES`, `SAMPLE_COUNTS`, `DOWNSTREAM_SEEDS`, `LORA_BUDGET`, `FT_*`, `SPECTRO_STEPS`,
`WEIGHTS_SUFFIX`.

## 3. Baselines (GPU job)

```bash
sbatch --export=ALL cluster/02_baselines.sbatch      # BASELINES="deepcnn random_init raw resnet18"
```
Same task/split/axis; `random_init` runs once per `--moe-arch` (mamba + transformer) as the
pretraining-lift floor for each arm. Writes `submission_*` dirs locally.

## 4. Publish (LOGIN node)

```bash
bash cluster/03_publish.sh
```
Pushes all checkpoints (`CKPT_DIR` → `HF_MODEL_REPO`, parent dir so per-(arch,patch) subfolders are
preserved), all `submission_*` result dirs (→ `HF_RESULTS_REPO` + W&B), and `wandb sync`s the offline
pretraining runs. Best-effort and idempotent; safe to re-run.

## Files
- `setup_env.sh` — one-time env build (uv + mamba-ssm + causal-conv1d + LLVM for sionna).
- `config.env` — all knobs (HF repos, recipe, downstream axis, LoRA budget); sources `secrets.env`.
- `download_data.sh` — login-node data pull.
- `01_pretrain_finetune.sbatch` — pretrain + LoRA-finetune, one arch, all patches (compute).
- `02_baselines.sbatch` — no-pretrain baselines (compute).
- `03_publish.sh` — upload checkpoints + results + W&B (login).

## Notes / gotchas
- **GPU**: jobs request `--gpus=rtx_3090:1` (typed GRES; the bare `--constraint` is ignored on BGU-HPC).
  Add your `--account`/`--partition` at submit time. Submit **from the repo root** (scripts resolve it
  via `SLURM_SUBMIT_DIR`).
- **Transformer batch 128** fits a 24 GB 3090 because attention uses memory-efficient SDPA
  (`spectro/hf_cache/pretraining/pretrained_model.py`); both arches use batch 128, no accum.
- **W&B logs OFFLINE** by default; `03_publish.sh` runs `wandb sync`. Set `WANDB_MODE=online` only if a
  node has egress.
- Token: `cluster/secrets.env` (gitignored) **or** a cached `huggingface-cli login`.
