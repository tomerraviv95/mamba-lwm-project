"""Domain-agnostic building blocks shared across the channel and spectro pipelines.

- ``mamba_layers``: bidirectional Mamba SSM primitives (``MambaBlock``, ``MambaLayer``).
- ``finetune``: a generic frozen-backbone probing / fine-tuning harness
  (``FineTuningWrapper``, ``finetune``, ``subsample_training_data``).
"""
