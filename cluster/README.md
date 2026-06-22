# Running the spectro pipeline on BGU-HPC

Reproducible Slurm workflow to **pretrain both spectro MoE backbones** (Transformer + Mamba) on
the fixed DeepMIMO-channel spectrogram dataset, then **run the downstream sample-variation sweep**
for both — keeping checkpoints in your Hugging Face account and loss curves in Weights & Biases.

Objective = the LWM-Spectro authors' flagship: **MLM + supervised contrastive on modulation &
mobility** (`loss = 1·MLM + 50·SupCon(mod) + 30·SupCon(mob)`), AdamW(wd=0.05), warmup→cosine.
Deliberate deviation: **magnitude** spectrograms (element_length=16) instead of complex.

Design choice: all **GPU compute runs in jobs**; all **internet I/O (pip install, HF push/pull,
W&B sync) runs on the login node** — robust because compute nodes have no outbound internet.

### How the cluster sees your files (why this works)
- **Home is a shared filesystem mounted on every node.** The cloned repo, the `.venv`, and any
  downloaded data under `$HOME` are written once to networked storage all nodes see. An env built
  once — or a dataset pulled once — is visible to every later job. The env **persists**.
- **Internet is login-node only.** So: build the env, pull the dataset, push to HF, and sync W&B
  **on the login node**; run GPU compute via `sbatch`.

## 0. One-time setup

```bash
ssh <bgu_user>@slurm.bgu.ac.il
git clone <this-repo> ~/lwm-competition-2025 && cd ~/lwm-competition-2025
git checkout feat/spectograms

# tell the pipeline who you are + drop in tokens
cp cluster/secrets.env.example cluster/secrets.env   # edit: HF_TOKEN=hf_... and WANDB_API_KEY=...
# (HF_USER is already set to tomerraviv95 in cluster/config.env)

# build the env ON THE LOGIN NODE (uv needs internet; a GPU is NOT needed to install)
bash cluster/setup_env.sh                             # uv sync + mamba-ssm + causal-conv1d build
mkdir -p cluster/logs
```

`setup_env.sh` bootstraps `uv`, runs `uv sync` (env from `uv.lock` at `$REPO_ROOT/.venv`), then
builds `mamba-ssm` **and `causal-conv1d`** with `--no-build-isolation`. Watch the verify line
`causal-conv1d installed: True (FUSED fast mamba)` — without it mamba runs ~8× slower (unfused).

## 1. Pull the dataset (login node)

```bash
uv run --no-sync python spectro/scripts/hf_sync.py pull-dataset \
    --repo tomerraviv95/lwm-spectro-deepmimo --dir spectro/outputs/spectro_deepmimo
```

## 2. Pretrain both backbones

```bash
sbatch --export=ALL,ARCH=transformer cluster/01_pretrain_spectro.sbatch
sbatch --export=ALL,ARCH=mamba       cluster/01_pretrain_spectro.sbatch
squeue --me
```

- Batch/accum auto-set per arch (transformer 16×16, mamba 64×4 → ~256 effective).
- Per-epoch `train/val` (MLM + both SupCon terms) → `spectro_${ARCH}_weights/{LTE,WiFi,5G}_losses.csv`
  **and** W&B (offline). Epochs/weights from `cluster/config.env` (`SPECTRO_DM_EPOCHS`, `SPECTRO_W_*`).
- Each job pushes its checkpoints to HF at the end (best-effort; canonical push is step 4).

## 3. Downstream sweep (both backbones)

```bash
# one-time: fetch the real demo eval set (login node)
uv run --no-sync python spectro/scripts/download_spectro_hf.py --skip-weights
sbatch cluster/02_downstream_spectro.sbatch          # arms: transformer, transformer_synth, mamba, raw
```
Produces `spectro/outputs/plots/spectro_performance_vs_samples.png`.

## 4. Publish + sync (login node)

```bash
bash cluster/push_to_hf.sh ckpts                     # -> tomerraviv95/wimamba-spectro-ckpts
wandb sync cluster/logs/wandb/offline-run-*          # upload the offline W&B runs
```

## Notes / gotchas
- **GPU**: jobs constrain to `rtx_3090` (Ampere). mamba-ssm will **not** run on `gtx_1080`.
- **`causal-conv1d`** is now built in `setup_env.sh`; the fully-fused `mamba_inner_fn` needs it
  (else ~8× slower per epoch on the bidirectional 12-layer experts). Non-fatal if its build fails.
- **CUDA module = `cuda/12.4`, not `cuda/13`.** Only used to build the kernels; torch's wheel
  bundles its own CUDA 12.8 runtime. 12.4 shares torch's major (12) so the build just warns.
- **W&B logs OFFLINE on compute nodes** (no internet); `wandb sync` on the login node uploads.
- Submit jobs with your conda env **deactivated** (`conda deactivate`).
- **`import sionna` needs an LLVM backend.** `setup_env.sh` installs `libLLVM` (conda-forge
  `llvmdev`) and persists `DRJIT_LIBLLVM_PATH`; symptom if missing: `the LLVM backend is inactive
  ... libLLVM.so could not be found`. Re-run `setup_env.sh` if you see it.
- Token: `cluster/secrets.env` (gitignored) **or** `huggingface-cli login` once (cached in shared home).
