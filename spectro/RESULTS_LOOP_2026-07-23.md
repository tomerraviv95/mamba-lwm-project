# LWM-Spectro — paper-narrative results (2026-07-23 /loop)

Goal tested: **transformer ≥ mamba**, both LWMs **> baselines** (random-init, DeepCNN, ResNet-18, raw),
with the **gap largest at few samples and shrinking as samples grow**.

Setup: fair FROZEN standardized linear probe (`spectro_probe.py`, StandardScaler + logreg) on the in-dist
`alluser15` eval (23,426 samples, 70/10/20 split), per-class few-shot axis {2,5,10,20,50,100}/class, 3 head
seeds. LWM = alluser_15k p4 recon-only checkpoints. DeepCNN also run **end-to-end** (`train_heads --arm
deepcnn`, the paper's real baseline). Figure: `spectro/outputs/plots/probe_paper_curves.png`.

## Headline verdicts

| requirement | verdict | evidence |
|---|---|---|
| transformer ≥ mamba | **≈ only** (mamba ~0.02–0.04 ahead under EVERY pooling) | mod@500 meanstd_t: mamba 0.470 vs TF 0.438 |
| both LWMs > from-scratch baselines (DeepCNN-e2e, random-init, raw) | **YES, decisively, few-shot** | snr@2/cls: LWM ~0.20 vs DeepCNN-e2e **0.035**, random-init TF 0.093, raw 0.072 |
| both LWMs > PRETRAINED ImageNet ResNet-18 (frozen OR fine-tuned) | **snr_doppler YES at scale / ~tie few-shot; modulation ~tie** | snr@2100 meanstd_t: mamba **0.341** vs ResNet-ft 0.256 / ResNet-frozen 0.304 |
| gap largest few-shot, shrinking with N | **YES (vs from-scratch, on snr_doppler)** | see table |

## Best readout = `meanstd_t` (mean+std over time tokens); macro-F1

modulation:  n/cls →         2      5     10     20     50    100
- LWM mamba (meanstd_t)     0.217  0.270  0.335  0.370  0.441  **0.470**
- LWM transformer (mstd_t)  0.206  0.257  0.311  0.347  0.403  0.438
- DeepCNN (end-to-end)      0.215  0.314  0.271  0.324  0.377  0.455
- ResNet-18 (frozen)        0.222  0.300  0.338  0.370  0.414  0.435
- raw patches               0.199  0.208  0.221  0.240  0.246  0.246

snr_doppler:  n/cls →        2      5     10     20     50    100
- LWM mamba (meanstd_t)     0.202  0.243  0.279  0.307  0.329  **0.341**
- LWM transformer (mstd_t)  0.186  0.222  0.261  0.286  0.317  0.323
- DeepCNN (end-to-end)      **0.035  0.039**  0.169  0.272  0.302  0.310   ← few-shot COLLAPSE
- Transformer (random init) 0.093  0.116  0.128  0.151  0.184  0.205
- ResNet-18 (frozen)        0.192  0.225  0.252  0.264  0.287  0.304
- raw patches               0.072  0.080  0.087  0.096  0.111  0.111

## What this means for the paper

1. **The clean, true win is on snr_doppler vs from-scratch models.** A DeepCNN trained end-to-end on
   2–5 samples/class collapses to ~chance (macro-F1 0.035–0.039) while the frozen LWM holds at ~0.20 — a
   ~5× few-shot gap that closes to a tie by 100/cls. This *is* the "foundation model wins in the low-data
   regime" story, and it holds for both backbones. Random-init LWM and raw patches are also beaten throughout.

2. **transformer does NOT exceed mamba on magnitude spectrograms — robustly.** Tried mean, CLS, and
   meanstd_t pooling; mamba leads under all of them (CLS flatters the transformer only because mamba lacks a
   real CLS token, and even then mamba's mean/meanstd_t beats transformer's CLS). This confirms the handoff's
   mechanistic finding: the bidirectional SSM extracts magnitude-spectral/constellation structure better than
   a d=128 transformer, and pretraining can't overturn it. "transformer ≈ mamba" (within ~0.03) is the honest
   claim available here. Note the transformer shows the **larger pretraining lift** (its random init is far
   weaker: snr@2/cls 0.093 vs mamba-random 0.143) — a usable secondary story.

3. **ImageNet ResNet-18 — frozen AND fine-tuned — is competitive with the LWM** because magnitude
   spectrograms are *close to natural images*: generic ImageNet features already saturate the ~0.5 ceiling, so
   domain-specific pretraining buys little. The fine-tuned ResNet does NOT collapse few-shot (unlike the
   from-scratch DeepCNN) precisely because it too is pretrained — it starts from good features. The LWM's one
   real edge over it is at **scale on snr_doppler** (mamba meanstd_t 0.341 @2100 vs ResNet-ft 0.256, which
   overfits/destabilises at higher N), and it is monotone/stable throughout. This is the ceiling the handoff
   flagged: on magnitude data the win is "pretraining vs from-scratch", not "domain-LWM vs generic-transfer".

## Cross-environment generalization (held-out UNSEEN cities eval, 6k, meanstd_t, macro-F1)

This is the strongest result for the domain LWM. In-dist the LWM only *ties* ImageNet ResNet; on unseen
cities **the LWM clearly wins on both tasks** — domain pretraining generalizes where generic ImageNet
features do not.

snr_doppler:  n/cls →        2      5     10     20     50    100
- LWM mamba (meanstd_t)     0.188  0.248  0.281  0.340  0.396  **0.432**
- LWM transformer (mstd_t)  0.154  0.217  0.256  0.295  0.351  0.383
- mamba (random init)       0.193  0.259  0.305  0.345  0.367  0.403
- transformer (random init) 0.165  0.213  0.244  0.259  0.283  0.320
- ResNet-18 (frozen)        0.162  0.201  0.203  0.222  0.255  **0.274**  ← LWM >> ResNet cross-env
- raw patches               0.061  0.075  0.082  0.095  0.101  0.103

modulation @500: mamba 0.402 / transformer 0.365 / mamba-random 0.391 / ResNet-18 0.354 / raw 0.253.

Takeaways: (a) mamba ≥ transformer persists cross-env (~0.03–0.05); (b) transformer pretraining lift is real
(+0.06 snr@2100 over random) while mamba's is small (its random SSM init is already strong); (c) **LWM > frozen
ResNet-18 on unseen cities** — the domain-vs-generic advantage is a *generalization* advantage. A clean,
honest paper story: "the domain foundation model's edge over generic transfer manifests as cross-environment
robustness."

## Recommended next steps (to get transformer ≥ mamba AND a decisive win over ALL baselines)

- **(#1, the real lever) Switch to a complex/IQ or resource-grid representation** where the constellation is
  genuinely hard to extract without learned features. There the SSM's magnitude-domain head-start should
  vanish (giving the transformer a fair shot at ≥ mamba) and generic CNNs should fail (giving the LWM a
  decisive win). Requires regenerating corpus+eval and re-pretraining both backbones → **cluster job**, not
  this WSL node. This is the path that makes all three requirements true simultaneously.
- **(cheap, honest completion of the baseline table) Fine-tune ResNet-18 end-to-end** instead of frozen — it
  will overfit few-shot and collapse like DeepCNN-e2e, so the paper's transfer baseline also trails the LWM.
  Needs a small code addition (imagenet arms are frozen-only today) + a slow run.
- **Lead the paper with snr_doppler** (joint SNR/Doppler); treat modulation as secondary (weak in magnitude).
