---
tags:
  - wireless
  - foundation-model
  - competition
  - data-limited
  - transformer
  - channel-estimation
  - localization
  - beam-prediction
  - channel-interpolation
  - los-nlos-classification
  - pytorch
  - self-supervised-learning
datasets:
  - DeepMIMO
metrics:
  - f1
  - mse
  - normalized-mse
  - localization-error
model-index:
  - name: lwm-v1.1
    results:
      - task:
          type: classification
          name: LoS/NLoS Classification
        dataset:
          name: Wireless Channel Dataset
          type: custom
        metrics:
          - type: f1
            value: baseline
      - task:
          type: regression
          name: Channel Estimation
        dataset:
          name: Wireless Channel Dataset
          type: custom
        metrics:
          - type: normalized-mse
            value: baseline
      - task:
          type: regression
          name: Channel Interpolation
        dataset:
          name: Wireless Channel Dataset
          type: custom
        metrics:
          - type: normalized-mse
            value: baseline
      - task:
          type: classification
          name: Beam Prediction
        dataset:
          name: Wireless Channel Dataset
          type: custom
        metrics:
          - type: f1
            value: baseline
      - task:
          type: regression
          name: Localization
        dataset:
          name: Wireless Channel Dataset
          type: custom
        metrics:
          - type: normalized-localization-error
            value: baseline
pipeline_tag: feature-extraction
library_name: pytorch
---

<div align="center">

# ⚡ LARGE WIRELESS MODELS (LWMs) 2025 CHALLENGE

**The goal is to improve performance across five wireless downstream tasks by optimizing a baseline LWM and/or designing new downstream models**

[![Model Hub](https://img.shields.io/badge/🤗%20HuggingFace-Model%20Hub-orange?style=flat-square)](https://huggingface.co/wi-lab/lwm-v1.1)
[![Tutorials](https://img.shields.io/badge/🎓%20Tutorials-Available-brightgreen?style=flat-square)](https://lwm-wireless.net/tutorials)
[![Website](https://img.shields.io/badge/🌐%20Website-lwm--wireless.net-blue?style=flat-square)](https://lwm-wireless.net/challenge)
[![Contact](https://img.shields.io/badge/📬%20Contact-lwmwireless@gmail.com-red?style=flat-square)](mailto:lwmwireless@gmail.com)

<p align="center">
  <a href="#-challenge-overview">Challenge Overview</a> •
  <a href="#-provided-materials">Provided Materials</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-submission-process">Submission Process</a> •
  <a href="#-tutorials">Tutorials</a> •
  <a href="#-citation">Citation</a> •
  <a href="#-community--support">Community & Support</a> •
  <a href="#-team">Team</a>
</p>

&nbsp;

<a target="_blank" href="https://huggingface.co/spaces/wi-lab/lwm-interactive-demo">
  <img src="https://img.shields.io/badge/▶️%20Try%20Interactive%20Demo-HuggingFace%20Spaces-yellow?style=for-the-badge" height="36px" alt="Try Interactive Demo"/>
</a>

&nbsp;

</div>

# 📡 Large Wireless Model (LWM) Challenge

Welcome to the official repository of the **LWM 2025 Challenge**, a competition designed to advance the state of foundation models in wireless communications and sensing. Participants are invited to optimize a provided baseline Large Wireless Model (LWM) and design downstream models to tackle five core wireless tasks with limited labeled data.

---

## 🧠 About LWM

**Large Wireless Model (LWM) 1.1** is a Transformer-based foundation model pre-trained using self-supervised learning on over 1 million unlabeled wireless channel samples. It generates rich, task-agnostic embeddings that significantly outperform raw channel representations on downstream tasks—especially when data is scarce or noisy or downstream models need to be simple.

---

## 🏁 Challenge Overview

Participants are given:

- A pre-trained LWM 1.1 checkpoint
- Baseline downstream task models
- Training, validation, and public test sets for each task
- Helper functions and templates

Your goal is to improve the **Composite Generalization Score (CG-Score)** across these five tasks:

1. **LoS/NLoS Classification** – F1-score
2. **Sub-6 GHz Channel to mmWave Beam Prediction** – Top-1 Beam F1-score
3. **Channel Interpolation** – Normalized MSE
4. **Channel Estimation** – Normalized MSE
5. **Localization** – Normalized Localization Error

Final rankings are based on hidden test sets evaluated by the organizers.

---

## 📦 Provided Materials

This repository contains:

- `pretrained_model.py` — Loads the baseline or your refined LWM model
- `train_heads.py` — The main script for training and evaluating all task-specific models. **This file must not be modified.** It is provided as a standardized template to ensure fairness and consistency across all teams. Participants must design their submissions to align with this script. The organizers will use an equivalent version of `train_heads.py` for final evaluation, and any deviation from the expected structure will result in automatic disqualification.
- `train_heads_config.py` — Contains training configs and model head definitions
- `train_lwm.py` — Contains LWM 1.1 pre-training and dataset reproducibility script
- `utils.py` — Helper functions (training, scoring, data handling)
- `task_{t}/` — Contains the training, validation, and public test sets for each downstream task. These datasets are used for jointly fine-tuning your refined LWM and training the corresponding task-specific models. While downstream training is restricted to the provided datasets, you are free to use any dataset for LWM pre-training. Participants are granted early access to the **DeepMIMO v4** dataset, which offers new, large-scale scenarios suitable for extended LWM refinement.
- `requirements.yml` — Conda environment file for dependency setup

---

## 🚀 Getting Started

### 📥 Clone the repo

```bash
git clone https://huggingface.co/wi-lab/lwm-competition-2025
cd lwm-competition-2025
```

### 🛠️ Set up the environment
```bash
conda env create -f requirements.yml
conda activate lwm_env
```

### 🧪 Run baseline pipeline
```bash
python train_heads.py
```
This jointly finetunes LWM and trains downstream heads, evaluates on public test sets, and creates a submission ZIP file.


### 🧩 Submission Process
1. Refine your LWM or downstream heads
2. Update `pretrained_model.py`, `train_heads_config.py`, and `utils.py`.
3. Run:
```bash
python train_heads.py
```
4. Submit the generated ZIP file to the competition portal

🛑 **Do not modify `train_heads.py`.** While you may adapt it for local development or experimentation, your final submission **must be fully compatible with the original, unmodified version** provided. The evaluation script used by the organizers assumes this exact structure—any deviation may result in disqualification.

---

## 📚 Tutorials

Visit the official tutorials page:

👉 [https://lwm-wireless.net/tutorials](https://lwm-wireless.net/tutorials)

---

## 🧪 Citation

If you use the LWM model or its components, please cite:

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

---

## 👥 Community & Support

* 💬 [Discussion Forum](https://huggingface.co/wi-lab/lwm-v1.1/discussions)
* 📨 Contact: [lwmwireless@gmail.com](mailto:lwmwireless@gmail.com)

---

## 👨‍🔬 Team

Developed by the [Wireless Intelligence Lab](https://wi-lab.net) at Arizona State University.

<p align="center">
  <a href="https://scholar.google.com/citations?user=PKjnTR4AAAAJ&hl=en" target="_blank">
    <img src="https://img.shields.io/badge/👤 Sadjad Alikhani-Click to view Scholar profile-blue?style=for-the-badge" alt="Sadjad Alikhani">
  </a>
  &nbsp;
  <a href="https://scholar.google.com/citations?user=MHKcvFMAAAAJ&hl=en" target="_blank">
    <img src="https://img.shields.io/badge/👤 Gouranga Charan-Click to view Scholar profile-green?style=for-the-badge" alt="Gouranga Charan">
  </a>
  &nbsp;
  <a href="https://scholar.google.com/citations?user=dLHw2qcAAAAJ&hl=en" target="_blank">
    <img src="https://img.shields.io/badge/👤 Ahmed Alkhateeb-Click to view Scholar profile-orange?style=for-the-badge" alt="Ahmed Alkhateeb">
  </a>
</p>
