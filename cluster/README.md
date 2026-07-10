# Running the LWM-Spectro pipeline on BGU-HPC

Reproducible Slurm workflow that follows the **LWM-Spectro paper protocol**: pretrain a spectro MoE
backbone (Transformer or Mamba) with **reconstruction-only** masked spectrogram modeling (eq. 21),
then run the paper downstream tasks — **modulation, joint SNR/Doppler, multi-protocol** — with the
**residual 1-D CNN head**, reporting **macro-F1 + accuracy** on the paper **samples-per-class** axis,
both **frozen ("LWM")** and **fine-tuned ("LWM FT")** (eq. 22: CE + λ_rec·MLM + λ_cont·SupCon).
Checkpoints go to your HF account, results to HF + W&B.

Two eval sets frame the comparison: **all-user 15%** (in-distribution, favors the transformer in the
data-rich regime) and **held-out cities** (cross-environment, favors mamba). Baselines: **Deep CNN**
(end-to-end), **ResNet-18** (frozen ImageNet), **random-init** (untrained MoE = pretraining-lift floor),
**raw**.

Design: **GPU compute in `sbatch` jobs**; **internet I/O (env build, HF pull, and the login-node
fallback for uploads / `wandb sync`) on the login node** — compute nodes have no usable outbound
internet, and the HF **xet** backend is disabled everywhere (`HF_HUB_DISABLE_XET=1`) so uploads use
plain HTTP.

## 0. One-time setup (login node)

```bash
ssh <bgu_user>@slurm.bgu.ac.il
git clone <this-repo> ~/lwm-competition-2025 && cd ~/lwm-competition-2025
git checkout feat/spectograms
cp cluster/secrets.env.example cluster/secrets.env    # edit: HF_TOKEN=hf_... and WANDB_API_KEY=...
bash cluster/setup_env.sh                             # uv sync (from uv.lock) + mamba-ssm + causal-conv1d
mkdir -p cluster/logs
```

`setup_env.sh` builds the validated env from the committed `uv.lock` at `$REPO_ROOT/.venv`, then builds
`mamba-ssm 2.3.0` **and** `causal-conv1d v1.6.2.post1` with `--no-build-isolation`. Watch for
`causal-conv1d OK: ... (FUSED fast mamba)` — without it mamba runs ~8× slower (unfused). This is the
step that fixes the mamba-ssm dependency issue: `uv.lock` pins torch 2.10+cu128 so the CUDA-extension
build matches (nvcc major 12; use `cuda/12.8`/`12.4`, **never** `cuda/13`).

## 1. Download data (login node)

```bash
bash cluster/download_data.sh
```
Pulls the 85%-user pretrain corpus + 15%-user eval (`lwm-spectro-alluser`) and the held-out-cities eval
(`lwm-spectro-gridstft`, eval only). Idempotent; verifies the three `manifest.json`s.

## 2. Pretrain + downstream + publish (GPU job, per arch)

```bash
sbatch --export=ALL,ARCH=mamba       cluster/pretrain_eval.sbatch
sbatch --export=ALL,ARCH=transformer cluster/pretrain_eval.sbatch
squeue --me
```
Each job: pretrains recon-only → pushes checkpoints to `HF_MODEL_REPO` → runs frozen **LWM** and
**LWM FT** downstream on **both** eval sets → uploads results to `HF_RESULTS_REPO` + W&B. Knobs in
`config.env`: `PER_CLASS_COUNTS`, `DOWNSTREAM_SEEDS`, `RUN_FROZEN`, `RUN_FT`, `FT_*`, `SPECTRO_STEPS`.
LWM FT is backbone-in-loop (slow) → its axis defaults lighter (`FT_PER_CLASS_COUNTS`); set `RUN_FT=0`
to skip.

## 3. Baselines + publish (GPU job)

```bash
sbatch --export=ALL cluster/run_baselines.sbatch      # BASELINES="deepcnn resnet18 random_init raw"
```
Same downstream protocol on both eval sets; uploads to `HF_RESULTS_REPO` + W&B.

## 4. If a job could not upload (compute node offline) — finish on the login node

Uploads are attempted at the end of each job and **degrade gracefully**; if the compute node had no
internet, re-run the exact command the job printed, e.g.:
```bash
# results are already on shared disk under spectro/outputs/submissions/
HF_HUB_DISABLE_XET=1 HF_TOKEN=... uv run python spectro/scripts/upload_results.py \
    --submissions spectro/outputs/submissions/submission_spectro_* \
    --hf-repo tomerraviv95/lwm-spectro-results --private --wandb-project lwm-spectro
uv run wandb sync cluster/logs/wandb/offline-*        # push the offline W&B runs
```

## Files
- `setup_env.sh` — one-time env build (uv + mamba-ssm + causal-conv1d + LLVM for sionna).
- `config.env` — all knobs (HF repos, recipe, downstream axis); sources `secrets.env` (gitignored).
- `download_data.sh` — login-node data pull.
- `pretrain_eval.sbatch` — pretrain + downstream (frozen + FT) + upload, one arch.
- `run_baselines.sbatch` — baselines + upload.

## Notes / gotchas
- **GPU**: jobs request `--gpus=rtx_3090:1` (typed GRES; the bare `--constraint` is ignored on BGU-HPC
  and lands on a GTX 1080 where torch cu128 dies). Add your `--account`/`--partition` at submit time.
- **CUDA module** for the build: `cuda/12.8` (or `12.4`), **never** `cuda/13` (major mismatch fails).
- **W&B logs OFFLINE** by default (`WANDB_MODE=offline`); `wandb sync` on the login node uploads. Set
  `WANDB_MODE=online` only if a node actually has egress.
- **`import sionna` needs LLVM**; `setup_env.sh` installs `libLLVM` and persists `DRJIT_LIBLLVM_PATH`.
- Token: `cluster/secrets.env` (gitignored) **or** a cached `huggingface-cli login`.
```
