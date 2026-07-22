"""Debug #3: is the pretraining lift BIGGER on cross-environment (held-out cities) than in-distribution?
If yes, pretraining's value is generalization and the in-dist eval merely saturates from random features.
Frozen mean-pool readout (RAM-safe), both arches, both eval sets."""
import os, sys, numpy as np, torch
sys.path.insert(0, 'spectro/scripts'); sys.path.insert(0, '.')
from huggingface_hub import hf_hub_download
from spectro_data import load_synthetic_data, PROTOCOLS
from spectro_moe import SpectroMoE
from spectro_patchify import patch_geometry
from sklearn.linear_model import LogisticRegression

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
REPO = 'tomerraviv95/wimamba-spectro-ckpts'; CK = '/tmp/claude-1000/ckpts'; N = 6000
EVALS = {'in-dist(alluser15)': 'spectro/outputs/spectro_eval_alluser15_gridstft',
         'cross-env(heldout-cities)': 'spectro/outputs/spectro_eval_heldout_cities_gridstft'}

def load_moe(arch, patch=4, random=False, suffix='alluser_15k'):
    d = f'spectro_{arch}_p{patch}_{suffix}_weights'
    for f in ['LTE_expert.pth', 'WiFi_expert.pth', '5G_expert.pth', 'router.pth']:
        hf_hub_download(REPO, f'{d}/{f}', repo_type='model', local_dir=CK)
    wdir = os.path.join(CK, d); s = torch.load(os.path.join(wdir, 'LTE_expert.pth'), map_location='cpu', weights_only=False)
    el = s.get('element_length', patch * patch); ml = s.get('max_len', patch_geometry(patch)['max_len'])
    if random:
        torch.manual_seed(42)
    moe = SpectroMoE(PROTOCOLS, d_model=s.get('d_model', 128), arch=arch, n_layers=s.get('n_layers', 12),
                     pool='mean', patch=patch, element_length=el, max_len=ml, in_channels=max(1, el // (patch * patch)))
    if not random:
        for p in PROTOCOLS:
            moe.load_expert(p, torch.load(os.path.join(wdir, f'{p}_expert.pth'), map_location='cpu', weights_only=False)['state_dict'])
    return moe

# cache the 4 backbones once
MOES = {(a, r): load_moe(a, random=(r == 'random')) for a in ['transformer', 'mamba'] for r in ['pretrained', 'random']}

@torch.no_grad()
def extract(moe, specs, proto):
    moe = moe.to(dev).eval(); X = moe.extract_embeddings(specs, routing='oracle', protocol_idx=proto, device=dev).numpy()
    moe.to('cpu'); torch.cuda.empty_cache(); return X

summary = {}
for ename, edir in EVALS.items():
    print(f'\n########## EVAL: {ename} ##########', flush=True)
    data = load_synthetic_data(edir, seed=42); tot = data.spectrograms.shape[0]
    idx = np.sort(np.random.RandomState(0).permutation(tot)[:min(N, tot)])
    specs = data.spectrograms[torch.as_tensor(idx)]; proto = data.protocol[idx]
    lab = {t: data.labels[t][idx].astype(int) for t in ['modulation', 'snr_doppler']}
    del data
    n = specs.shape[0]; perm = np.random.RandomState(1).permutation(n); tr, te = perm[:int(.7 * n)], perm[int(.7 * n):]
    print(f'  N={n} train={len(tr)} test={len(te)}', flush=True)
    def sweep(X, y):
        out = []
        for k in [50, 100, 200, 400, 600]:
            sub = tr[:k]
            out.append(float((LogisticRegression(max_iter=300).fit(X[sub], y[sub]).predict(X[te]) == y[te]).mean())
                       if len(set(y[sub].tolist())) > 1 else float('nan'))
        return out
    for a in ['transformer', 'mamba']:
        Xp = extract(MOES[(a, 'pretrained')], specs, proto); Xr = extract(MOES[(a, 'random')], specs, proto)
        for t in ['modulation', 'snr_doppler']:
            p = np.array(sweep(Xp, lab[t])); r = np.array(sweep(Xr, lab[t]))
            summary[(ename, a, t)] = (p, r)
            print(f'  {a:11s} {t:12s}: pre {" ".join(f"{v:.2f}" for v in p)} | rnd {" ".join(f"{v:.2f}" for v in r)} | LIFT {" ".join(f"{v:+.2f}" for v in p-r)}', flush=True)
    del specs

print('\n=== PRETRAINING LIFT @600 (pretrained - random): in-dist vs cross-env ===', flush=True)
for a in ['transformer', 'mamba']:
    for t in ['modulation', 'snr_doppler']:
        ind = summary[('in-dist(alluser15)', a, t)]; xe = summary[('cross-env(heldout-cities)', a, t)]
        print(f'  {a:11s} {t:12s}: in-dist {ind[0][-1]-ind[1][-1]:+.2f}   cross-env {xe[0][-1]-xe[1][-1]:+.2f}', flush=True)
print('DONE', flush=True)
