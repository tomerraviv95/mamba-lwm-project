# CLAUDE.md

## Project Overview

Solution repo for the LWM 2025 Competition. Contains Transformer and Mamba-based Large Wireless Models for five downstream wireless tasks.

## Repository Layout

- `scripts/` - All Python scripts (training, evaluation, benchmarks, plotting)
- `outputs/` - Generated artifacts (plots, submissions, pretrained model weights)
- `task_1/` through `task_5/` - Competition task data
- `data/`, `DeepMIMO/`, `deepmimo_scenarios/` - Raw data and ray-tracing framework

## Key Scripts

- `scripts/train_heads.py` - Main entry point: loads pretrained LWM, fine-tunes with task heads, evaluates, creates submission
- `scripts/train_heads_config.py` - Task head definitions (classification, regression, reconstruction) and training hyperparameters
- `scripts/pretrained_model.py` - Transformer LWM architecture
- `scripts/mamba_model.py` - Mamba (SSM) LWM architecture
- `scripts/utils.py` - Shared utilities (data loading via DeepMIMO, tokenization, patching, scoring)
- `scripts/train_lwm.py` - Pretraining pipeline

## Path Conventions

All scripts use `_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')` to resolve paths relative to the repo root. This allows scripts to work regardless of the current working directory.

## Development Notes

- Python environment: `pip install -r requirements.txt` + `pip install -e ./DeepMIMO`
- GPU required for training (CUDA)
- The `wimae` branch preserves WiMAE/ContraWiMAE code that was removed from master
- `.pth` weight files are gitignored by default; the two used checkpoints in `outputs/pretrained_models/` were force-added
- `DeepMIMO/` is a separate git repo (not a submodule)
