#!/usr/bin/env bash
# SLURM submit wrapper for the transformer data-scaling ablation (10k/20k pretrain + sweep).
# The driver needs a GPU, so it must run on a compute node — NOT the login node.
#
# Fill in your cluster's partition/account below (ask your admin or check `sinfo`), then:
#     sbatch cluster/datascale_transformer.sbatch
#     squeue -u $USER            # watch the queue
#     tail -f cluster/logs/datascale_transformer.log
#
#SBATCH --job-name=datascale-tf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/datascale_transformer.slurm.%j.out
### --- SET THESE FOR YOUR CLUSTER (uncomment + fill) ---
### #SBATCH --partition=gpu
### #SBATCH --account=your_account
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/mamba-lwm-project}"
echo "[sbatch] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash cluster/run_datascale_transformer.sh
