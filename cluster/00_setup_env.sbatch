#!/bin/bash
################################################################################
# Phase 0 (one-time): build the conda env on a GPU node via Slurm — no interactive
# session needed. mamba-ssm may compile CUDA kernels, so this MUST run on a GPU node;
# running it as a batch job satisfies that. The env lands in shared ~/.conda and
# persists for all later jobs.
#
# Submit from the LOGIN node (conda DEACTIVATED):
#     sbatch cluster/00_setup_env.sbatch
#     less cluster/logs/setup-<JOBID>.out      # check it ends with "-> OK"
################################################################################
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name spectro-setup
#SBATCH --output cluster/logs/setup-%J.out
#SBATCH --gpus=1
#SBATCH --constraint=rtx_3090
#SBATCH --mail-type=END,FAIL

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

module load anaconda
module load cuda/12.4
export PYTHONUNBUFFERED=TRUE

echo "Building env on $(hostname):"; nvidia-smi -L
bash "$HERE/setup_env.sh"
