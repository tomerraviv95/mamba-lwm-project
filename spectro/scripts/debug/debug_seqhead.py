"""Debug #2: frozen backbone + Conv1d-SEQUENCE head (no finetune), pretrained vs random, both arches.
Completes the 2x2 {pretrained,random} x {mean-pool, conv1d-seq}. Tells us whether the powerful seq head
EQUALIZES random vs pretrained (task saturable from random features) or whether FINETUNING was the
destroyer (frozen-pretrained-seq would then beat the finetuned downstream numbers). RAM-safe: N subsampled,
sequences fp16, one arch/init at a time then freed."""
import os, sys, copy, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'spectro/scripts'); sys.path.insert(0, '.')
from huggingface_hub import hf_hub_download
from spectro_data import load_synthetic_data, PROTOCOLS
from spectro_moe import SpectroMoE
from spectro_patchify import patch_geometry
from spectro_train_heads_config import Conv1dHead

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL = 'spectro/outputs/spectro_eval_alluser15_gridstft'; REPO = 'tomerraviv95/wimamba-spectro-ckpts'
CK = '/tmp/claude-1000/ckpts'; N = 3200

def load_moe(arch, patch=4, random=False, suffix='alluser_15k'):
    d = f'spectro_{arch}_p{patch}_{suffix}_weights'
    for f in ['LTE_expert.pth', 'WiFi_expert.pth', '5G_expert.pth', 'router.pth']:
        hf_hub_download(REPO, f'{d}/{f}', repo_type='model', local_dir=CK)
    wdir = os.path.join(CK, d); s = torch.load(os.path.join(wdir, 'LTE_expert.pth'), map_location='cpu', weights_only=False)
    el = s.get('element_length', patch * patch); ml = s.get('max_len', patch_geometry(patch)['max_len'])
    if random:
        torch.manual_seed(42)
    moe = SpectroMoE(PROTOCOLS, d_model=s.get('d_model', 128), arch=arch, n_layers=s.get('n_layers', 12),
                     pool='seq', patch=patch, element_length=el, max_len=ml, in_channels=max(1, el // (patch * patch)))
    if not random:
        for p in PROTOCOLS:
            moe.load_expert(p, torch.load(os.path.join(wdir, f'{p}_expert.pth'), map_location='cpu', weights_only=False)['state_dict'])
    return moe

print('loading eval ...', flush=True)
data = load_synthetic_data(EVAL, seed=42); tot = data.spectrograms.shape[0]
idx = np.sort(np.random.RandomState(0).permutation(tot)[:min(N, tot)])
specs = data.spectrograms[torch.as_tensor(idx)]; proto = data.protocol[idx]
lab = {t: torch.as_tensor(data.labels[t][idx], dtype=torch.long) for t in ['modulation', 'snr_doppler']}
ncls = {t: int(lab[t].max()) + 1 for t in lab}
del data
n = specs.shape[0]; perm = np.random.RandomState(1).permutation(n)
tr_all, va, te = perm[:int(.7 * n)], perm[int(.7 * n):int(.85 * n)], perm[int(.85 * n):]
print(f'N={n} train_pool={len(tr_all)} val={len(va)} test={len(te)}', flush=True)

@torch.no_grad()
def extract_seq(moe):
    moe = moe.to(dev).eval()
    S = moe.extract_sequences(specs, routing='oracle', protocol_idx=proto, device=dev)  # (N,T,d) fp16 on cpu
    moe.to('cpu'); torch.cuda.empty_cache(); return S

def train_head(S, task, k, seed=0):
    torch.manual_seed(seed)
    y = lab[task]; head = Conv1dHead(S.shape[2], ncls[task]).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(); sub = tr_all[:k]
    def acc(idxs):
        head.eval()
        with torch.no_grad():
            out = []
            for s in range(0, len(idxs), 512):
                b = idxs[s:s + 512]; out.append(head(S[torch.as_tensor(b)].to(dev)).cpu())
            p = torch.cat(out).argmax(1)
        return float((p == y[torch.as_tensor(idxs)]).float().mean())
    best, best_state, ctr = -1, None, 0
    for ep in range(60):
        head.train(); order = np.random.RandomState(ep).permutation(sub)
        for s in range(0, len(order), 64):
            b = order[s:s + 64]; opt.zero_grad()
            loss = ce(head(S[torch.as_tensor(b)].to(dev)), y[torch.as_tensor(b)].to(dev)); loss.backward(); opt.step()
        v = acc(va)
        if v > best + 1e-4:
            best, best_state, ctr = v, copy.deepcopy(head.state_dict()), 0
        else:
            ctr += 1
            if ctr >= 12:
                break
    head.load_state_dict(best_state); return acc(te)

res = {}
for arch in ['transformer', 'mamba']:
    for init in ['pretrained', 'random']:
        S = extract_seq(load_moe(arch, random=(init == 'random')))
        print(f'--- {arch} {init} (frozen, Conv1d-seq head) ---', flush=True)
        for task in ['modulation', 'snr_doppler']:
            accs = [train_head(S, task, k) for k in [50, 100, 200, 400, 600]]
            res[(arch, init, task)] = accs
            print('  %-12s: %s' % (task, ' '.join(f'{k}:{a:.2f}' for k, a in zip([50,100,200,400,600], accs))), flush=True)
        del S; torch.cuda.empty_cache()

print('\n=== frozen Conv1d-seq: pretraining lift (pretrained - random) ===', flush=True)
for arch in ['transformer', 'mamba']:
    for task in ['modulation', 'snr_doppler']:
        p = np.array(res[(arch, 'pretrained', task)]); r = np.array(res[(arch, 'random', task)])
        print(f'  {arch:11s} {task:12s}: ' + ' '.join(f'{k}:{v:+.2f}' for k, v in zip([50,100,200,400,600], p - r)), flush=True)
print('DONE', flush=True)
