"""LWM FT — LoRA fine-tuning of a pretrained spectro MoE on the downstream task.

Unlike ``spectro_train_heads.py`` (frozen encoder -> head), this adapts the routed expert backbones with
LoRA (adapter params capped at ``--lora-budget`` of the backbone; the head is trainable on top) and trains
them together with the residual 1-D CNN head under the joint objective

    L = CE(head)  +  w_recon * L_recon(MLM)  +  w_cont * L_cont(SupCon on the task label)

LoRA is applied as a WEIGHT parametrization (``shared/lora.py``) so it also adapts the Mamba SSM
projections that the fused kernel reads via raw ``.weight`` (the finetune Mamba is built with
``use_fast_path=False``). Data uses a 70/10/20 train/val/test split; the few-shot axis is TOTAL train
samples drawn from the 70% pool; early-stops on validation macro-F1; oracle routing. Reports macro-F1
(primary) + accuracy in the sweep schema. Loads ``spectro_{arch}_p{patch}_{suffix}_weights``.

Example (all-user in-distribution, mamba, patch 4):
    CUDA_VISIBLE_DEVICES=0 python spectro/scripts/spectro_finetune.py --arch mamba --patch 4 \
        --weights-suffix alluser_b128 --synth-dir spectro/outputs/spectro_eval_alluser15_gridstft \
        --sample-counts 50 100 200 400 600 --seeds 42 43 44 --lora-budget 0.05
"""
from __future__ import annotations
import argparse, copy, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datagen'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from spectro_data import PROTOCOLS, TASKS, load_synthetic_data, load_spectro_data  # noqa: E402
from spectro_moe import SpectroMoE  # noqa: E402
from spectro_patchify import spectrogram_patchify, patch_geometry, build_masked_tensors  # noqa: E402
from spectro_train_heads_config import Conv1dHead  # noqa: E402
from spectro_sweep import _macro_f1, _accuracy, _subsample_per_class, _subsample_count  # noqa: E402
from spectro_pretrain import weights_dir  # noqa: E402
from contrastive import ProjectionHead, supervised_contrastive_loss  # noqa: E402
from shared.lora import (TARGETS, apply_lora, pick_rank, backbone_param_count,  # noqa: E402
                         lora_params_at_rank, trainable_param_count)

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_SUBMISSIONS = os.path.join(_REPO_ROOT, 'spectro', 'outputs', 'submissions')


def _load_moe(arch, patch, suffix, device, use_fast_path=False):
    wdir = weights_dir(arch, patch, suffix)
    router_ckpt = torch.load(os.path.join(wdir, 'router.pth'), map_location='cpu', weights_only=False)
    sample = torch.load(os.path.join(wdir, f'{PROTOCOLS[0]}_expert.pth'), map_location='cpu', weights_only=False)
    d_model, n_layers = sample.get('d_model', 128), sample.get('n_layers', 12)
    element_length = sample.get('element_length', patch * patch)
    max_len = sample.get('max_len', patch_geometry(patch)['max_len'])
    in_ch = max(1, element_length // (patch * patch))
    # LoRA mode needs use_fast_path=False (slow path exposes the Mamba SSM weights to the weight-
    # parametrization). Partial finetuning trains real weights, so the fused fast path is fine.
    moe = SpectroMoE(PROTOCOLS, d_model=d_model, arch=arch, n_layers=n_layers, pool='seq', patch=patch,
                     element_length=element_length, max_len=max_len, in_channels=in_ch,
                     use_fast_path=use_fast_path)
    for p in PROTOCOLS:
        moe.load_expert(p, torch.load(os.path.join(wdir, f'{p}_expert.pth'),
                                      map_location='cpu', weights_only=False)['state_dict'])
    moe.router.load_state_dict(router_ckpt['state_dict'])
    return moe.to(device), d_model, patch


def _unfreeze_last_layers(expert, n) -> int:
    """Freeze the expert, then unfreeze the last ``n`` backbone blocks (partial finetuning, à la the
    channel-pipeline ``FineTuningWrapper``'s named-layer unfreeze — NOT LoRA). Both backbones store their
    block stack as ``.layers`` (the transformer under ``.net.layers``). Returns #trainable backbone params.
    """
    for p in expert.parameters():
        p.requires_grad_(False)
    stack = getattr(expert, 'layers', None) or getattr(getattr(expert, 'net', None), 'layers', None)
    if stack is None:                                   # unknown structure -> unfreeze all (safe fallback)
        for p in expert.parameters():
            p.requires_grad_(True)
    else:
        n = max(0, min(n, len(stack)))
        for blk in list(stack)[len(stack) - n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
    return sum(p.numel() for p in expert.parameters() if p.requires_grad)


def _input_ids(specs, patch, device):
    """(N,[C,]128,128) -> (N, n_patches+1, element_length) clean tokens with CLS, on device."""
    P = spectrogram_patchify(specs, patch=patch, normalize=True)
    cls = np.full((P.shape[0], 1, P.shape[2]), 0.2, dtype=np.float32)
    return torch.tensor(np.concatenate([cls, P], axis=1), dtype=torch.float32, device=device)


def _routed(moe, ids, proto, device, masked_pos=None):
    """Grad-enabled oracle-routed expert forward. masked_pos=None -> token sequence (B,T,d);
    else masked-recon logits (B, n_masks, E). Each protocol's expert runs on its own samples."""
    out = None
    for e_idx, p in enumerate(moe.protocols):
        mask = torch.as_tensor(proto == e_idx, device=device)
        if not mask.any():
            continue
        if masked_pos is None:
            o = moe.experts[p](ids[mask])                     # (m, T, d)
        else:
            o, _ = moe.experts[p](ids[mask], masked_pos[mask])  # (m, n_masks, E)
        if out is None:
            out = torch.zeros(ids.shape[0], o.shape[1], o.shape[2], device=device, dtype=o.dtype)
        out[mask] = o
    return out


def _finetune_once(moe0, head, proj, ids_all, specs, proto, y, tr, va, te, args, device):
    """One FT run from the pretrained state: returns (test_f1, test_acc). Mutates fresh head/proj/moe."""
    moe = moe0                                     # already reloaded to pretrained state by caller
    # backbone group = ONLY the trainable LoRA adapter params (base weights are frozen); head+proj full.
    lora_params = [p for e in moe.experts.values() for p in e.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {'params': lora_params, 'lr': args.lr_backbone},
        {'params': list(head.parameters()) + list(proj.parameters()), 'lr': args.lr_head},
    ], weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _eval(idx):
        moe.eval(); head.eval()
        outs, ys = [], []
        for s in range(0, len(idx), args.eval_batch):
            b = idx[s:s + args.eval_batch]
            seq = _routed(moe, ids_all[torch.as_tensor(b)], proto[b], device)
            outs.append(head(seq).cpu()); ys.append(y[torch.as_tensor(b)])
        logits, yy = torch.cat(outs), torch.cat(ys)
        return _macro_f1(logits, yy), _accuracy(logits, yy)

    best_f1, best_state, ctr = -1.0, None, 0
    rng = np.random.RandomState(args.seed)
    for ep in range(args.epochs):
        moe.train(); head.train(); proj.train(); order = rng.permutation(tr)
        for s in range(0, len(order), args.batch_size):
            b = order[s:s + args.batch_size]
            bt = torch.as_tensor(b)
            opt.zero_grad()
            seq = _routed(moe, ids_all[bt], proto[b], device)          # clean (B,T,d)
            logits = head(seq)
            loss = ce(logits, y[bt].to(device))
            if args.w_cont > 0:
                loss = loss + args.w_cont * supervised_contrastive_loss(
                    proj(seq), y[bt].to(device), temperature=args.temperature,
                    base_temperature=args.temperature)
            if args.w_recon > 0:                                       # masked-recon (MLM) on the same batch
                m_ids, m_toks, m_pos = build_masked_tensors(specs[bt], mask_percent=args.mask_percent,
                                                            seed=args.seed + s, patch=args.patch)
                m_ids, m_toks, m_pos = m_ids.to(device).float(), m_toks.to(device).float(), m_pos.to(device)
                rec = _routed(moe, m_ids, proto[b], device, masked_pos=m_pos)
                loss = loss + args.w_recon * nn.functional.mse_loss(rec, m_toks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
            opt.step()
        va_f1, _ = _eval(va)
        if va_f1 > best_f1 + 1e-6:
            best_f1, ctr = va_f1, 0
            best_state = (copy.deepcopy(moe.state_dict()), copy.deepcopy(head.state_dict()))
        else:
            ctr += 1
            if ctr >= args.patience:
                break
    if best_state is not None:
        moe.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    return _eval(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['mamba', 'transformer'], default='mamba')
    ap.add_argument('--patch', type=int, default=4, choices=[4, 6, 8])
    ap.add_argument('--weights-suffix', default='alluser')
    ap.add_argument('--synth-dir', default='spectro/outputs/spectro_eval_alluser15_gridstft',
                    help="in-domain eval corpus (train/val/test split); omit to use the demo set")
    ap.add_argument('--tasks', nargs='+', default=list(TASKS.keys()))
    ap.add_argument('--per-class-counts', type=int, nargs='+', default=None,
                    help="few-shot axis: #train samples PER CLASS. Overrides --sample-counts.")
    ap.add_argument('--sample-counts', type=int, nargs='+', default=None,
                    help="few-shot axis: TOTAL #train samples drawn from the 70%% train pool "
                         "(default 50 100 200 400 600).")
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--val-frac', type=float, default=0.10, help='downstream val fraction (70/10/20 split).')
    ap.add_argument('--test-frac', type=float, default=0.20, help='downstream test fraction (70/10/20 split).')
    ap.add_argument('--lora-budget', type=float, default=0.05,
                    help='cap LoRA ADAPTER params at this fraction of the backbone (head is trainable, '
                         'not counted). The rank is auto-picked to fit unless --lora-rank is given.')
    ap.add_argument('--lora-rank', type=int, default=None,
                    help='override the LoRA rank (else auto-picked from --lora-budget).')
    ap.add_argument('--lora-alpha', type=float, default=None,
                    help='LoRA scaling alpha (default = rank, i.e. scaling 1.0).')
    ap.add_argument('--lora-max-rank', type=int, default=16, help='cap for the auto-picked rank.')
    ap.add_argument('--tune-mode', choices=['partial', 'lora'], default='partial',
                    help="how to adapt the backbone: 'partial' unfreezes the last --tune-last-n blocks "
                         "(CE fine-tuning of part of the network, like scripts/train_heads.py); 'lora' uses "
                         "the ≤--lora-budget weight-parametrization adapters.")
    ap.add_argument('--tune-last-n', type=int, default=2,
                    help="partial mode: number of top backbone blocks to unfreeze (of n_layers).")
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--eval-batch', type=int, default=128)
    ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--lr-backbone', type=float, default=2e-4)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--w-recon', type=float, default=0.0,
                    help='lambda_recon (MLM aux loss) during fine-tuning. Default 0 = CE only (set 1.0 for '
                         'the paper eq.22 joint loss).')
    ap.add_argument('--w-cont', type=float, default=0.0,
                    help='lambda_cont (SupCon aux loss) during fine-tuning. Default 0 = CE only (set 0.3 for paper).')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--mask-percent', type=float, default=0.7)
    ap.add_argument('--proj-pool', choices=['mean', 'cls', 'meanstd_t'], default='mean')
    ap.add_argument('--seed', type=int, default=42, help='data-split seed (per-run seeds come from --seeds).')
    ap.add_argument('--run-tag', default='',
                    help="extra suffix on the submission dir to disambiguate eval sets.")
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = (load_synthetic_data(args.synth_dir, seed=args.seed,
                                val_frac=args.val_frac, test_frac=args.test_frac) if args.synth_dir
            else load_spectro_data(seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac))
    # partial finetuning trains real weights (fused fast path OK); LoRA needs the slow path.
    moe, d_model, _ = _load_moe(args.arch, args.patch, args.weights_suffix, device,
                                use_fast_path=(args.tune_mode != 'lora'))

    # --- adapt the backbone: partial (unfreeze top blocks, CE) or LoRA (<=budget adapters) ---
    one_expert = next(iter(moe.experts.values()))
    bb = backbone_param_count(one_expert)
    rank_meta = None
    if args.tune_mode == 'lora':
        rules = TARGETS[args.arch]
        rank_meta = args.lora_rank or pick_rank(one_expert, rules, budget_frac=args.lora_budget,
                                                max_rank=args.lora_max_rank)
        for e in moe.experts.values():
            apply_lora(e, rules, rank_meta, alpha=args.lora_alpha)
        tunable = lora_params_at_rank(one_expert, rules, rank_meta)
        how = f"LoRA rank={rank_meta} adapters"
    else:  # partial: unfreeze the last N backbone blocks on every expert
        for e in moe.experts.values():
            _unfreeze_last_layers(e, args.tune_last_n)
        tunable = trainable_param_count(one_expert)
        how = f"partial FT last {args.tune_last_n} blocks"
    aux = ''.join(([f' +{args.w_recon}*MLM'] if args.w_recon > 0 else [])
                  + ([f' +{args.w_cont}*SupCon'] if args.w_cont > 0 else []))
    print(f"FT[{args.tune_mode}] arch={args.arch}: {how} = {tunable:,}/{bb:,} = {tunable / bb:.2%} of "
          f"backbone trainable; head trainable on top; loss=CE{aux or ' only'}")
    # capture the reset point AFTER freeze/wrap so per-run `moe.load_state_dict(pretrained_state)` restores
    # the pretrained backbone (LoRA adapters start as a no-op: B=0; the requires_grad config persists).
    pretrained_state = copy.deepcopy(moe.state_dict())
    ids_all = _input_ids(data.spectrograms, args.patch, device)
    specs = data.spectrograms if torch.is_tensor(data.spectrograms) else torch.as_tensor(np.asarray(data.spectrograms))
    proto = data.protocol
    tr, va, te = data.train_idx, data.val_idx, data.test_idx

    if args.per_class_counts is not None:
        mode, x_points = 'per_class', list(args.per_class_counts)
    elif args.sample_counts is not None:
        mode, x_points = 'counts', list(args.sample_counts)
    else:
        mode, x_points = 'counts', [50, 100, 200, 400, 600]

    arm = ('transformer_synth' if args.arch == 'transformer' else 'mamba') + '_ft'
    print(f"LWM-FT {arm} p{args.patch} [{args.weights_suffix}] tasks={args.tasks} mode={mode} "
          f"x={x_points} seeds={args.seeds} eval={os.path.basename((args.synth_dir or 'demo').rstrip('/'))}")

    y_train_full = {}
    results_by_task, results_by_x = {}, {}
    for task in args.tasks:
        y = torch.as_tensor(data.labels[task], dtype=torch.long)
        n_classes = data.n_classes(task)
        y_train_full[task] = y[tr].numpy()
        task_results = {}
        for x in x_points:
            f1s, accs, n_samples = [], [], None
            for sd in args.seeds:
                if mode == 'per_class':
                    sub = _subsample_per_class(y_train_full[task], x, seed=sd)
                else:
                    sub = _subsample_count(len(tr), x, seed=sd)
                sel = tr[sub]; n_samples = len(sel)
                torch.manual_seed(sd); np.random.seed(sd)
                moe.load_state_dict(pretrained_state)                 # fresh start from the pretrained ckpt
                head = Conv1dHead(d_model, n_classes).to(device)
                proj = ProjectionHead(d_model, 128, pool=args.proj_pool).to(device)
                run_args = argparse.Namespace(**{**vars(args), 'seed': sd})
                f1, acc = _finetune_once(moe, head, proj, ids_all, specs, proto, y, sel, va, te,
                                         run_args, device)
                f1s.append(f1); accs.append(acc)
            key = f"{x}pc" if mode == 'per_class' else str(n_samples)
            task_results[key] = {'score': float(np.mean(f1s)), 'score_std': float(np.std(f1s)),
                                 'scores': f1s, 'accuracy': float(np.mean(accs)),
                                 'accuracy_std': float(np.std(accs)), 'accuracies': accs,
                                 'n_samples': n_samples, 'per_class': x if mode == 'per_class' else None}
            results_by_x.setdefault(key, {})[f'task_{task}'] = float(np.mean(f1s))
            print(f"  [{arm}] {task:12s} n={n_samples:5d}{f' ({x}/cls)' if mode=='per_class' else ''}  "
                  f"F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}  acc={np.mean(accs):.4f}", flush=True)
        results_by_task[f'task_{task}'] = {'name': TASKS[task]['name'], 'results': task_results}

    aggregated = {
        'experiment_config': {
            'arm': arm, 'tasks': args.tasks, 'feature_dim': f'{args.tune_mode}-ft', 'seeds': args.seeds,
            'metric': 'macro_f1 (primary) + accuracy', 'head': f'conv1d_{args.tune_mode}_ft',
            'tune_mode': args.tune_mode, 'tune_last_n': args.tune_last_n, 'lora_rank': rank_meta,
            'trainable_backbone_frac': tunable / bb,
            'lambda_recon': args.w_recon, 'lambda_cont': args.w_cont, 'temperature': args.temperature,
            'lr_backbone': args.lr_backbone, 'lr_head': args.lr_head,
            'split': f'{int((1-args.val_frac-args.test_frac)*100)}/'
                     f'{int(args.val_frac*100)}/{int(args.test_frac*100)}',
            'x_axis': 'per_class_counts' if mode == 'per_class' else 'sample_counts',
            'per_class_counts': x_points if mode == 'per_class' else None,
            'sample_counts': x_points if mode == 'counts' else None,
        },
        'results_by_task': results_by_task,
        'results_by_percentage': {
            k: {**v, 'composite_score': float(np.mean(list(v.values())))} for k, v in results_by_x.items()
        },
    }
    tag = (('_heldout' if args.synth_dir else '') + (f'_{args.weights_suffix}' if args.weights_suffix else '')
           + (f'_{args.run_tag}' if args.run_tag else ''))
    out_dir = os.path.join(_SUBMISSIONS, f'submission_spectro_{arm}_p{args.patch}{tag}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nDone. LWM-FT results -> {out_dir}/aggregated_results.json")


if __name__ == '__main__':
    main()
