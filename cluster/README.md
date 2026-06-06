# Running the spectro pipeline on BGU-HPC

Reproducible Slurm workflow to **generate the synthetic corpus** and **pretrain the Mamba MoE**
on the cluster, keeping both the data and the checkpoints in your Hugging Face account.

Design choice: all **GPU compute runs in jobs**; all **HF network I/O runs on the login node**
(`push_to_hf.sh`) — robust whether or not compute nodes have outbound internet.

## 0. One-time setup

```bash
ssh <bgu_user>@slurm.bgu.ac.il
git clone <this-repo> ~/lwm-competition-2025 && cd ~/lwm-competition-2025
git checkout feat/spectograms

# tell the pipeline who you are on HF + drop in a write token
cp cluster/secrets.env.example cluster/secrets.env   # then edit: HF_TOKEN=hf_...
# (HF_USER is already set to tomerraviv95 in cluster/config.env)

# build the env via a batch job (NO interactive session needed)
mkdir -p cluster/logs
sbatch cluster/00_setup_env.sbatch                   # GPU job; builds the conda env
# wait for it, then check it finished cleanly:
less cluster/logs/setup-*.out                        # should end with "... -> OK"
```

> Everything runs through `sbatch` — no `sinteractive` required. (If you *prefer*
> interactive, `sinteractive --gpus=1 --constraint=rtx_3090` then `bash cluster/setup_env.sh`
> works too, but it's optional.)

Pinned stack (validated locally): torch 2.10 cu128, sionna 2.0.1, mamba-ssm 2.3.0, py3.12.
`causal-conv1d` is intentionally omitted (mamba-ssm runs without it).

## 1. Generate → pretrain (chained)

```bash
mkdir -p cluster/logs
gid=$(sbatch --parsable cluster/01_gen_data.sbatch)      # ~157k samples (per-combo=500)
sbatch --dependency=afterok:$gid cluster/02_pretrain.sbatch
squeue --me
```

Tune sizes/epochs in `cluster/config.env` (`PER_COMBO`, `N_LAYERS`, `PRETRAIN_EPOCHS`, …).

## 2. Publish to Hugging Face (login node)

```bash
bash cluster/push_to_hf.sh dataset    # -> <user>/lwm-spectro-synthetic   (private)
bash cluster/push_to_hf.sh ckpts      # -> <user>/wimamba-spectro-ckpts   (private)
```

## 3. (Optional) relevance experiment

```bash
sbatch cluster/03_relevance.sbatch    # before/after plot proving the data helps
```

## Fetch results anywhere

```bash
python spectro/scripts/hf_sync.py pull-dataset --repo <user>/lwm-spectro-synthetic --dir spectro/outputs/synthetic
python spectro/scripts/hf_sync.py pull-ckpts   --repo <user>/wimamba-spectro-ckpts  --dir spectro/outputs/pretrained_models
```

## Notes / gotchas
- **GPU**: jobs constrain to `rtx_3090` (Ampere). mamba-ssm will **not** run on `gtx_1080`.
- Submit jobs with your conda env **deactivated** (`conda deactivate`).
- If a compute node *does* have internet you can skip `push_to_hf.sh` and add a pull/push call
  inside the sbatch scripts — but the login-node split is the safe default.
- Large corpus is ~5 GB; it lives under `spectro/outputs/synthetic` (gitignored). For heavy I/O
  you can point `DATA_DIR` at local `--tmp` scratch and copy back (see the BGU cheat sheet).
- Token: `cluster/secrets.env` (gitignored) **or** `huggingface-cli login` once (cached in shared home).
