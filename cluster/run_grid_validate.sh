#!/usr/bin/env bash
# Validate the --repr grid fix: generate a held-out-cities GRID eval set, then probe whether modulation
# (and snr/mobility) are now learnable from raw grid spectrograms (no backbone). Detached + idempotent.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG=cluster/logs/m7_grid_validate.log
EVAL=spectro/outputs/spectro_eval_heldout_cities_grid
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{
  echo "=== grid-validate START $(date) ==="
  if [ ! -f "$EVAL/manifest.json" ]; then
    echo "generating GRID held-out eval $(date)"
    $PY spectro/datagen/generate_deepmimo_spectro.py --out "$EVAL" \
      --cities asu_campus_3p5:1,boston5g_3p5:2,o1_3p5:3 --symbol-mult 8 --vary-speed \
      --per-city 2000 --seed 1234 --batch 2 --repr grid || { echo "GEN FAILED $(date)"; exit 1; }
  fi
  echo "=== probing raw-grid task separability (vs the STFT eval) $(date) ==="
  $PY - <<'PY'
import torch, numpy as np, json, os
from sklearn.linear_model import LogisticRegression
def load(d):
    m=json.load(open(d+'/manifest.json')); sh=[]
    for s in m['shards']: sh+=torch.load(os.path.join(d,s),map_location='cpu',weights_only=False)
    specs=np.stack([np.asarray(x['data'],dtype=np.float32).reshape(128,128) for x in sh])
    lab={t:np.array([str(x[k]) for x in sh]) for t,k in [('mod','mod'),('snr','snr'),('mob','mob')]}
    return specs,lab
def probe(specs,lab,task):
    y=np.array([sorted(set(lab[task])).index(v) for v in lab[task]]); nc=len(set(y))
    F=specs.reshape(len(specs),32,4,32,4).mean(axis=(2,4)).reshape(len(specs),-1)
    rng=np.random.RandomState(0); p=rng.permutation(len(F)); k=int(.8*len(F))
    a=(LogisticRegression(max_iter=120).fit(F[p[:k]],y[p[:k]]).predict(F[p[k:]])==y[p[k:]]).mean()
    return a,1/nc
for name,d in [('GRID (fix)','spectro/outputs/spectro_eval_heldout_cities_grid'),
               ('STFT (old)','spectro/outputs/spectro_eval_heldout_cities')]:
    if not os.path.exists(d+'/manifest.json'): print(f"{name}: missing"); continue
    specs,lab=load(d); print(f"\n[{name}] raw spatial-feature probe (chance in parens):")
    for t in ['mod','snr','mob']:
        a,ch=probe(specs,lab,t); print(f"  {t}: {a:.3f}  (chance {ch:.2f})")
PY
  echo "=== GRID VALIDATE DONE $(date) ==="
} >> "$LOG" 2>&1
