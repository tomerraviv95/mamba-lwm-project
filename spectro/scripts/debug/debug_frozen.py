"""Debug: frozen PRETRAINED vs frozen RANDOM backbone, same mean-pool readout, on the downstream eval.
Isolates the pretraining contribution from the finetuning confound. RAM-safe: mean-pool features only
(N x 128), subsampled eval, no per-token sequence tensors."""
import os, sys, numpy as np, torch
sys.path.insert(0, 'spectro/scripts'); sys.path.insert(0, '.')
from huggingface_hub import hf_hub_download
from spectro_data import load_synthetic_data, PROTOCOLS
from spectro_moe import SpectroMoE
from spectro_patchify import patch_geometry
from sklearn.linear_model import LogisticRegression

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL = 'spectro/outputs/spectro_eval_alluser15_gridstft'
REPO = 'tomerraviv95/wimamba-spectro-ckpts'
CK = '/tmp/claude-1000/ckpts'
N = 6000

def load_moe(arch, suffix='alluser_15k', patch=4, random=False):
    d = f'spectro_{arch}_p{patch}_{suffix}_weights'
    for f in ['LTE_expert.pth', 'WiFi_expert.pth', '5G_expert.pth', 'router.pth']:
        hf_hub_download(REPO, f'{d}/{f}', repo_type='model', local_dir=CK)
    wdir = os.path.join(CK, d)
    s = torch.load(os.path.join(wdir, 'LTE_expert.pth'), map_location='cpu', weights_only=False)
    el = s.get('element_length', patch * patch); ml = s.get('max_len', patch_geometry(patch)['max_len'])
    in_ch = max(1, el // (patch * patch))
    if random:
        torch.manual_seed(42)
    moe = SpectroMoE(PROTOCOLS, d_model=s.get('d_model', 128), arch=arch, n_layers=s.get('n_layers', 12),
                     pool='mean', patch=patch, element_length=el, max_len=ml, in_channels=in_ch)
    if not random:
        for p in PROTOCOLS:
            moe.load_expert(p, torch.load(os.path.join(wdir, f'{p}_expert.pth'), map_location='cpu',
                                          weights_only=False)['state_dict'])
    return moe

print('loading eval ...', flush=True)
data = load_synthetic_data(EVAL, seed=42, val_frac=0.10, test_frac=0.20)
tot = data.spectrograms.shape[0]
idx = np.sort(np.random.RandomState(0).permutation(tot)[:min(N, tot)])
specs = data.spectrograms[torch.as_tensor(idx)]
proto = data.protocol[idx]
lab = {t: data.labels[t][idx].astype(int) for t in ['modulation', 'snr_doppler']}
del data
n = specs.shape[0]
perm = np.random.RandomState(1).permutation(n); ntr = int(0.7 * n); tr, te = perm[:ntr], perm[ntr:]
print(f'eval subset N={n}  train={len(tr)} test={len(te)}', flush=True)

@torch.no_grad()
def extract(moe):
    moe = moe.to(dev).eval()
    X = moe.extract_embeddings(specs, routing='oracle', protocol_idx=proto, device=dev).numpy()
    moe.to('cpu'); torch.cuda.empty_cache()
    return X

def sweep(X, name):
    print(f'--- {name} ---', flush=True)
    out = {}
    for task in ['modulation', 'snr_doppler']:
        y = lab[task]; accs = []
        for k in [50, 100, 200, 400, 600]:
            sub = tr[:k]
            if len(set(y[sub].tolist())) < 2:
                accs.append(float('nan')); continue
            clf = LogisticRegression(max_iter=300).fit(X[sub], y[sub])
            accs.append(float((clf.predict(X[te]) == y[te]).mean()))
        out[task] = accs
        print('  %-12s: %s' % (task, ' '.join(f'{k}:{a:.2f}' for k, a in zip([50,100,200,400,600], accs))), flush=True)
    return out

res = {}
for arch in ['transformer', 'mamba']:
    res[(arch, 'pretrained')] = sweep(extract(load_moe(arch, random=False)), f'{arch} PRETRAINED (frozen mean-pool)')
    res[(arch, 'random')] = sweep(extract(load_moe(arch, random=True)), f'{arch} RANDOM-INIT (frozen mean-pool)')

print('\n=== PRETRAINING LIFT (pretrained - random), frozen mean-pool ===', flush=True)
for arch in ['transformer', 'mamba']:
    for task in ['modulation', 'snr_doppler']:
        p = np.array(res[(arch, 'pretrained')][task]); r = np.array(res[(arch, 'random')][task])
        d = p - r
        print(f'  {arch:11s} {task:12s}: ' + ' '.join(f'{k}:{v:+.2f}' for k, v in zip([50,100,200,400,600], d)), flush=True)
print('DONE', flush=True)
