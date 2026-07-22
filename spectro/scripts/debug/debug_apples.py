"""Apples-to-apples: ALL feature extractors FROZEN + the SAME linear probe. Adds the MISSING fair baseline
= frozen RANDOM DeepCNN (so DeepCNN is not the only end-to-end arm). Also tests the z-score-normalization
hypothesis for the LWM. If frozen-random-CNN << ResNet/LWM, DeepCNN's downstream edge was end-to-end
training, not representation. RAM-safe: mean/natural-pooled features, N=6000 subset."""
import os, sys, types, numpy as np, torch
sys.path.insert(0, 'spectro/scripts'); sys.path.insert(0, '.')
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from spectro_data import load_synthetic_data, PROTOCOLS
from spectro_moe import SpectroMoE
from spectro_patchify import patch_geometry, spectrogram_patchify
from spectro_train_heads import DeepCNN, _imagenet_features

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL = 'spectro/outputs/spectro_eval_alluser15_gridstft'; REPO = 'tomerraviv95/wimamba-spectro-ckpts'; N = 6000

def lwm(arch, random=False, patch=4, normalize=True):
    d = f'spectro_{arch}_p{patch}_alluser_15k_weights'
    for f in ['LTE_expert.pth', 'WiFi_expert.pth', '5G_expert.pth', 'router.pth']:
        hf_hub_download(REPO, f'{d}/{f}', repo_type='model', local_dir='spectro/outputs/pretrained_models')
    wdir = f'spectro/outputs/pretrained_models/{d}'; s = torch.load(f'{wdir}/LTE_expert.pth', map_location='cpu', weights_only=False)
    el = s.get('element_length', patch * patch * 2); ml = s.get('max_len', patch_geometry(patch)['max_len'])
    if random:
        torch.manual_seed(42)
    m = SpectroMoE(PROTOCOLS, d_model=128, arch=arch, n_layers=12, pool='mean', patch=patch,
                   element_length=el, max_len=ml, in_channels=max(1, el // (patch * patch)))
    if not random:
        for p in PROTOCOLS:
            m.load_expert(p, torch.load(f'{wdir}/{p}_expert.pth', map_location='cpu', weights_only=False)['state_dict'])
    return m

print('loading eval ...', flush=True)
data = load_synthetic_data(EVAL, seed=42, val_frac=0.10, test_frac=0.20); tot = data.spectrograms.shape[0]
idx = np.sort(np.random.RandomState(0).permutation(tot)[:min(N, tot)])
specs = data.spectrograms[torch.as_tensor(idx)]; proto = data.protocol[idx]
lab = {t: data.labels[t][idx].astype(int) for t in ['modulation', 'snr_doppler']}
del data
n = specs.shape[0]; perm = np.random.RandomState(1).permutation(n); tr, te = perm[:int(.7 * n)], perm[int(.7 * n):]
print(f'N={n} train={len(tr)} test={len(te)}', flush=True)

@torch.no_grad()
def lwm_feats(m):
    m = m.to(dev).eval(); X = m.extract_embeddings(specs, routing='oracle', protocol_idx=proto, device=dev).numpy()
    m.to('cpu'); torch.cuda.empty_cache(); return X

@torch.no_grad()
def cnn_random_feats():
    torch.manual_seed(7); net = DeepCNN(in_channels=specs.shape[1] if specs.ndim == 4 else 1).to(dev).eval()
    out = []
    for s in range(0, n, 128):
        out.append(net(specs[s:s + 128].to(dev).float()).cpu())
    torch.cuda.empty_cache(); return torch.cat(out).numpy()

def raw_feats():
    P = spectrogram_patchify(specs, patch=4, normalize=True); return P.mean(axis=1)

feats = {}
feats['LWM mamba pretrained'] = lwm_feats(lwm('mamba', random=False))
feats['LWM mamba RANDOM'] = lwm_feats(lwm('mamba', random=True))
feats['LWM transf pretrained'] = lwm_feats(lwm('transformer', random=False))
feats['LWM transf RANDOM'] = lwm_feats(lwm('transformer', random=True))
feats['DeepCNN RANDOM (frozen)'] = cnn_random_feats()
sh = types.SimpleNamespace(spectrograms=specs)
feats['ResNet18 (frozen ImageNet)'] = _imagenet_features(sh, 'resnet18', dev).numpy()
feats['raw patches'] = raw_feats()

print('\n=== FROZEN feature quality — linear probe test accuracy @ sample counts ===', flush=True)
for name, X in feats.items():
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    line = []
    for task in ['modulation', 'snr_doppler']:
        y = lab[task]; accs = []
        for k in [50, 100, 200, 400, 600]:
            sub = tr[:k]
            accs.append((LogisticRegression(max_iter=300).fit(Xs[sub], y[sub]).predict(Xs[te]) == y[te]).mean()
                        if len(set(y[sub].tolist())) > 1 else float('nan'))
        line.append(f"{task[:3]} " + "/".join(f"{a:.2f}" for a in accs))
    print(f'  {name:28s} dim={X.shape[1]:4d} | {"  |  ".join(line)}', flush=True)
print('DONE', flush=True)
