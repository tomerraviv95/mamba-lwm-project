"""Load and split the LWM-Spectro demo dataset for downstream sample-variation probing.

Loads ``spectro/hf_cache/demo_data.pt`` (10,500 dicts) once, exposes:
- the raw spectrogram tensor (N,128,128) and per-protocol index groups,
- integer label vectors for the downstream tasks (modulation / snr / mobility),
- the precomputed baseline embeddings (``moe_embedding``, ``tech_embedding``),
- a single deterministic train/val/test split stratified by protocol, shared across tasks
  and arms so the "number of training samples" axis is comparable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_DEMO_PATH = os.path.join(_REPO_ROOT, 'spectro', 'hf_cache', 'demo_data.pt')

PROTOCOLS = ['LTE', 'WiFi', '5G']

# Downstream classification tasks — the LWM-Spectro PAPER set:
#   Task 1: modulation classification (5-class).
#   Task 2: JOINT SNR/Doppler — every (SNR, mobility) pair is one class (paper Sec. V.B).
#   Task 3: multi-protocol classification (LTE/WiFi/5G) — the MoE-router capability (paper Table III).
# Each task lists the raw sample field(s) whose combined value defines the class; '__protocol__'
# is the protocol index resolved from `tech`.
TASKS = {
    'modulation':  {'fields': ('mod',),        'name': 'Modulation'},
    'snr_doppler': {'fields': ('snr', 'mob'),  'name': 'SNR/Doppler'},
    'protocol':    {'fields': ('__protocol__',), 'name': 'Protocol'},
}
# Legacy single-field tasks (SNR / mobility split), available but not in the default paper sweep.
EXTRA_TASKS = {
    'snr':      {'fields': ('snr',), 'name': 'SNR'},
    'mobility': {'fields': ('mob',), 'name': 'Mobility'},
}


def _class_sort_key(field: str, value: str):
    """Deterministic ordering within a task's classes: SNR numeric, protocol by PROTOCOLS order."""
    if field == 'snr':
        return _snr_key(value)
    if field == '__protocol__':
        return PROTOCOLS.index(value) if value in PROTOCOLS else 1e9
    return value


def _build_labels(samples, protocol: np.ndarray, tasks: Dict = TASKS):
    """Build integer label vectors + ordered class-name lists for every task in ``tasks``.

    A task's class is the tuple of its fields' raw string values across the sample (e.g. the joint
    ``snr_doppler`` class is (SNR, mobility)). Classes are sorted per-field for stable indices.
    """
    labels, names = {}, {}
    for task, cfg in tasks.items():
        fields = cfg['fields']
        raw = []
        for i in range(len(protocol)):
            parts = tuple(PROTOCOLS[protocol[i]] if f == '__protocol__' else _field_str(samples[i], f)
                          for f in fields)
            raw.append(parts)
        classes = sorted(set(raw),
                         key=lambda t: tuple(_class_sort_key(f, v) for f, v in zip(fields, t)))
        to_idx = {c: i for i, c in enumerate(classes)}
        labels[task] = np.array([to_idx[r] for r in raw], dtype=np.int64)
        names[task] = [' | '.join(c) for c in classes]
    return labels, names


def _field_str(sample, field) -> str:
    v = sample[field]
    if isinstance(v, np.ndarray):
        return str(v.reshape(-1)[0])
    return str(v)


@dataclass
class SpectroData:
    spectrograms: torch.Tensor          # (N, 128, 128) float32
    protocol: np.ndarray                # (N,) int, index into PROTOCOLS
    labels: Dict[str, np.ndarray]       # task -> (N,) int labels
    label_names: Dict[str, List[str]]   # task -> ordered class names
    moe_embedding: torch.Tensor         # (N, 128) baseline features (None for synthetic data)
    tech_embedding: torch.Tensor        # (N, 128) oracle-routed features (None for synthetic)
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def n_classes(self, task: str) -> int:
        return len(self.label_names[task])


def load_synthetic_data(out_dir: str, seed: int = 42, val_frac: float = 0.15,
                        test_frac: float = 0.15) -> "SpectroData":
    """Load a synthetic corpus produced by ``spectro/datagen/generate.py``.

    Reads the shards listed in ``manifest.json`` into a ``SpectroData`` with the same task
    labels + protocol-stratified split, but **no precomputed embeddings** (those only exist for
    the real demo data). Intended for pretraining the Mamba MoE on a larger corpus. Downstream
    finetuning passes ``val_frac=0.10, test_frac=0.20`` (a 70/10/20 split); the default 0.15/0.15
    keeps the pretraining in-corpus probe unchanged.
    """
    import json
    with open(os.path.join(out_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    samples = []
    for shard in manifest['shards']:
        sp = os.path.join(out_dir, shard)
        try:
            samples.extend(torch.load(sp, weights_only=False))
        except Exception as e:                       # truncated/empty shard from an interrupted download
            sz = os.path.getsize(sp) if os.path.isfile(sp) else -1
            raise RuntimeError(
                f"failed to load shard {sp} ({sz} bytes): {type(e).__name__}: {e}. "
                f"The download is likely incomplete — re-fetch with "
                f"`FORCE=1 bash cluster/download_data.sh` (or hf_download_gridstft.py --force).") from e

    # Keep the corpus in FLOAT16 (the source `data` is already float16) — a large corpus (132k x
    # 2ch x 128x128) is ~17 GB in float32 but ~8.7 GB in float16, and downstream casts to float() per
    # batch anyway. Then free the per-sample dict list (another ~corpus-sized chunk) so the pretrain
    # host doesn't OOM. Costs a little accuracy in the z-score stats (negligible).
    specs = torch.stack([s['data'].squeeze(0).to(torch.float16) for s in samples])
    proto_to_idx = {p: i for i, p in enumerate(PROTOCOLS)}
    protocol = np.array([proto_to_idx[_field_str(s, 'tech')] for s in samples], dtype=np.int64)

    # build TASKS + EXTRA_TASKS so 'mobility'/'snr' are available (e.g. the SupCon-mobility pretraining term
    # needs a standalone 'mobility' label; extra keys are ignored by downstream arms that don't request them).
    labels, label_names = _build_labels(samples, protocol, {**TASKS, **EXTRA_TASKS})
    del samples                                         # release the raw dict list (~corpus-sized)

    train_idx, val_idx, test_idx = _stratified_split(protocol, seed=seed,
                                                     val_frac=val_frac, test_frac=test_frac)
    return SpectroData(
        spectrograms=specs, protocol=protocol, labels=labels, label_names=label_names,
        moe_embedding=None, tech_embedding=None,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )


def _stratified_split(strata: np.ndarray, *, val_frac=0.15, test_frac=0.15, seed=42):
    """Per-stratum shuffled split -> (train_idx, val_idx, test_idx), sorted."""
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(round(val_frac * n))
        n_test = int(round(test_frac * n))
        val.extend(idx[:n_val])
        test.extend(idx[n_val:n_val + n_test])
        train.extend(idx[n_val + n_test:])
    return (np.sort(np.array(train, dtype=np.int64)),
            np.sort(np.array(val, dtype=np.int64)),
            np.sort(np.array(test, dtype=np.int64)))


def load_spectro_data(demo_path: str = _DEMO_PATH, seed: int = 42, val_frac: float = 0.15,
                      test_frac: float = 0.15) -> SpectroData:
    """Load the demo dataset and build label vectors + a protocol-stratified split."""
    samples = torch.load(demo_path, weights_only=False)
    n = len(samples)

    specs = torch.stack([s['data'].squeeze(0).float() for s in samples])  # (N,128,128)

    proto_to_idx = {p: i for i, p in enumerate(PROTOCOLS)}
    protocol = np.array([proto_to_idx[_field_str(s, 'tech')] for s in samples], dtype=np.int64)

    labels, label_names = _build_labels(samples, protocol)

    moe = torch.stack([s['moe_embedding'].float() for s in samples])      # (N,128)
    tech = torch.stack([torch.as_tensor(s['tech_embedding']).float() for s in samples])

    train_idx, val_idx, test_idx = _stratified_split(protocol, seed=seed,
                                                     val_frac=val_frac, test_frac=test_frac)

    return SpectroData(
        spectrograms=specs, protocol=protocol, labels=labels, label_names=label_names,
        moe_embedding=moe, tech_embedding=tech,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )


def _snr_key(v: str) -> float:
    """Sort 'SNR-5dB','SNR0dB',... numerically."""
    digits = v.replace('SNR', '').replace('dB', '')
    try:
        return float(digits)
    except ValueError:
        return 0.0


if __name__ == '__main__':
    d = load_spectro_data()
    print(f"N={len(d.spectrograms)} spectrograms {tuple(d.spectrograms.shape)}")
    print(f"split: train={len(d.train_idx)} val={len(d.val_idx)} test={len(d.test_idx)}")
    for t in TASKS:
        print(f"  task {t}: {d.n_classes(t)} classes -> {d.label_names[t]}")
    print(f"moe_embedding {tuple(d.moe_embedding.shape)} tech_embedding {tuple(d.tech_embedding.shape)}")
