# WiMamba: Linear-Scale Wireless Foundation Model

Official code for the paper **"WiMamba: Linear-Scale Wireless Foundation Model"**.

This repository contains our solution for the [Large Wireless Models (LWM) 2025 Challenge](https://lwm-wireless.net/challenge), which aims to improve performance across five wireless downstream tasks by optimizing a baseline LWM foundation model.

## Tasks

1. **LoS/NLoS Classification** - Binary classification of line-of-sight vs non-line-of-sight
2. **Beam Prediction** - Sub-6 GHz channel to mmWave beam index prediction (64 beams)
3. **Channel Interpolation** - Reconstruct missing channel values from partial observations
4. **Channel Estimation** - Full channel reconstruction from noisy measurements
5. **User Localization** - 2D position estimation from channel embeddings

## Approach

We extend the baseline LWM (Transformer) with:

- **Mamba architecture** - A bidirectional selective state-space model as an alternative backbone, offering linear-time sequence processing
- **Multi-resolution patching** - Support for multiple patch sizes (4x4, 6x6, 8x8) with resolution-specific embeddings, allowing the same backbone to handle different tokenization granularities
- **Continued pretraining** - Adapt pretrained models to new patch sizes by freezing the backbone and retraining only the embedding layer

## Repository Structure

```
lwm-competition-2025/
├── scripts/                           # All Python scripts
│   ├── train_heads.py                 # Main: fine-tune LWM + train task heads
│   ├── train_heads_config.py          # Task head architectures + training configs
│   ├── train_lwm.py                   # LWM pretraining pipeline
│   ├── continue_pretrain.py           # Continued pretraining with new patch sizes
│   ├── pretrained_model.py            # Transformer-based LWM architecture
│   ├── mamba_model.py                 # Mamba-based LWM architecture
│   ├── utils.py                       # Utilities (data loading, tokenization, scoring)
│   ├── benchmark_latency.py           # Mamba vs Transformer inference latency
│   ├── benchmark_patch_sizes.py       # RAM/latency vs patch size benchmark
│   ├── plot_sample_variation.py       # Performance vs training data percentage plots
│   ├── plot_task_scores.py            # Task score comparison across architectures
│   └── run_sample_variation_multi_patch.py  # Wrapper for multi-patch experiments
├── outputs/                           # Generated artifacts
│   ├── plots/                         # Benchmark and result plots
│   ├── submissions/                   # Competition submission directories
│   └── pretrained_models/             # Model checkpoints
│       ├── lwm_weights/               # Transformer LWM checkpoint
│       └── mamba_weights/             # Mamba LWM checkpoint
├── task_1/ ... task_5/                # Competition task data (train/val/test splits)
├── data/                              # Raw channel data
├── DeepMIMO/                          # DeepMIMO ray-tracing framework
├── deepmimo_scenarios/                # DeepMIMO scenario configs
├── config.json                        # Global configuration
├── model_comparison.csv               # Architecture comparison results
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
pip install -e ./DeepMIMO
```

## Usage

### Train and evaluate on all 5 tasks
```bash
python scripts/train_heads.py
```

This loads a pretrained LWM, fine-tunes it jointly with task-specific heads, evaluates on test sets, and produces a submission ZIP.

### Configure model type
Edit `MODEL_TYPE` in `scripts/train_heads.py`:
- `"transformer"` - Transformer LWM (default)
- `"mamba"` - Mamba LWM
- `"raw"` - Raw patch features (no pretrained model)

### Pretrain LWM from scratch
```bash
python scripts/train_lwm.py
```

### Run benchmarks
```bash
python scripts/benchmark_latency.py       # Mamba vs Transformer latency
python scripts/benchmark_patch_sizes.py    # Memory/latency vs patch size
```

## Pretrained Weights

| Model | Checkpoint | Parameters |
|-------|-----------|------------|
| Transformer LWM | `outputs/pretrained_models/lwm_weights/` | ~1.2M |
| Mamba LWM | `outputs/pretrained_models/mamba_weights/` | ~1.2M |

Both models use `d_model=128`, `n_layers=12`, and support multi-resolution patching with patch sizes [4, 6, 8].

## Citation

Based on the LWM foundation model:

```bibtex
@misc{alikhani2025largewirelessmodellwm,
      title={Large Wireless Model (LWM): A Foundation Model for Wireless Channels},
      author={Sadjad Alikhani and Gouranga Charan and Ahmed Alkhateeb},
      year={2025},
      eprint={2411.08872},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2411.08872},
}
```
