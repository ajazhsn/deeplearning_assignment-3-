"""
ablations.py — W&B Report Experiments (Sections 2.1–2.5)
DA6401 Assignment 3

Run all experiments:
    python ablations.py --exp all

Run individually:
    python ablations.py --exp noam_vs_fixed
    python ablations.py --exp scaling_factor
    python ablations.py --exp attention_rollout
    python ablations.py --exp learned_pe
    python ablations.py --exp label_smoothing
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import (
    Transformer, MultiHeadAttention, PositionalEncoding,
    make_src_mask, make_tgt_mask,
)
from dataset import get_dataloaders, PAD_IDX
from lr_scheduler import NoamScheduler, get_lr_history
from train import (
    run_epoch, LabelSmoothingLoss, evaluate_bleu,
    save_checkpoint, greedy_decode,
)


# ══════════════════════════════════════════════════════════════════════
#  SHARED CONFIG  (15 epochs ≈ 25–30 min per run on T4)
# ══════════════════════════════════════════════════════════════════════

BASE_CONFIG = {
    'd_model':         256,
    'N':               3,
    'num_heads':       8,
    'd_ff':            512,
    'dropout':         0.1,
    'batch_size':      128,
    'num_epochs':      15,
    'warmup_steps':    4000,
    'label_smoothing': 0.1,
    'min_freq':        2,
    'max_len':         128,
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def build_model(src_vocab_size, tgt_vocab_size, cfg, use_learned_pe=False):
    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model   = cfg['d_model'],
        N         = cfg['N'],
        num_heads = cfg['num_heads'],
        d_ff      = cfg['d_ff'],
        dropout   = cfg['dropout'],
        pad_idx   = PAD_IDX,
    ).to(DEVICE)
    if use_learned_pe:
        model.pos_enc = LearnedPositionalEncoding(
            cfg['d_model'], cfg['dropout'], cfg['max_len'] + 10
        ).to(DEVICE)
    return model


def get_data(cfg):
    return get_dataloaders(
        batch_size = cfg['batch_size'],
        min_freq   = cfg['min_freq'],
        max_len    = cfg['max_len'],
    )


def make_opt_sched(model, cfg):
    opt   = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = NoamScheduler(opt, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps'])
    return opt, sched


def train_model(model, train_loader, val_loader, cfg,
                optimizer, scheduler, loss_fn,
                step_hook=None, epoch_extra=None):
    """
    Full training loop.
    step_hook(model, global_step) -> dict  : logged every step (grad norms etc.)
    epoch_extra(model, val_loader) -> dict : logged every epoch
    """
    best_val, global_step = float('inf'), 0

    for epoch in range(cfg['num_epochs']):
        model.train()
        total_loss = total_tokens = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{cfg['num_epochs']}")
        for src, tgt in pbar:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            logits = model(src, tgt_in,
                           make_src_mask(src, PAD_IDX),
                           make_tgt_mask(tgt_in, PAD_IDX))
            loss = loss_fn(logits.contiguous().view(-1, logits.size(-1)),
                           tgt_out.contiguous().view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if step_hook:
                hook_dict = step_hook(model, global_step)
                if hook_dict:
                    wandb.log({**hook_dict, 'global_step': global_step})

            optimizer.step()
            if scheduler:
                scheduler.step()

            n = (tgt_out != PAD_IDX).sum().item()
            total_loss   += loss.item() * n
            total_tokens += n
            global_step  += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss = total_loss / max(total_tokens, 1)
        val_loss   = run_epoch(val_loader, model, loss_fn, None, None,
                               epoch_num=epoch+1, is_train=False, device=DEVICE)
        lr = optimizer.param_groups[0]['lr']

        log = {'epoch': epoch+1, 'train_loss': train_loss,
               'val_loss': val_loss, 'lr': lr}
        if epoch_extra:
            log.update(epoch_extra(model, val_loader))
        wandb.log(log, step=epoch+1)
        print(f"  Epoch {epoch+1}: train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch+1, 'best_ablation.pt')

    return best_val


# ══════════════════════════════════════════════════════════════════════
#  2.1 — NOAM SCHEDULER vs FIXED LR
# ══════════════════════════════════════════════════════════════════════

def exp_noam_vs_fixed():
    print("\n=== 2.1  Noam vs Fixed LR ===")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_data(BASE_CONFIG)
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, BASE_CONFIG['label_smoothing'])

    # Plot LR schedule
    lrs = get_lr_history(BASE_CONFIG['d_model'], BASE_CONFIG['warmup_steps'], 20_000)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lrs)
    ax.axvline(BASE_CONFIG['warmup_steps'], color='red', linestyle='--',
               label=f"warmup={BASE_CONFIG['warmup_steps']}")
    ax.set(xlabel='Step', ylabel='LR', title='Noam LR Schedule')
    ax.legend(); plt.tight_layout()

    # Run 1: Noam
    wandb.init(project="da6401-a3", name="2.1_noam",
               config={**BASE_CONFIG, 'lr_type': 'noam'})
    wandb.log({'noam_lr_curve': wandb.Image(fig)}); plt.close(fig)
    model = build_model(len(src_vocab), len(tgt_vocab), BASE_CONFIG)
    opt, sched = make_opt_sched(model, BASE_CONFIG)
    train_model(model, train_loader, val_loader, BASE_CONFIG, opt, sched, loss_fn)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=DEVICE)
    wandb.log({'test_bleu': bleu}); wandb.finish()
    print(f"  Noam BLEU: {bleu:.2f}")

    # Run 2: Fixed LR
    wandb.init(project="da6401-a3", name="2.1_fixed_lr_1e-4",
               config={**BASE_CONFIG, 'lr_type': 'fixed', 'fixed_lr': 1e-4})
    model = build_model(len(src_vocab), len(tgt_vocab), BASE_CONFIG)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    train_model(model, train_loader, val_loader, BASE_CONFIG,
                opt, scheduler=None, loss_fn=loss_fn)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=DEVICE)
    wandb.log({'test_bleu': bleu}); wandb.finish()
    print(f"  Fixed LR BLEU: {bleu:.2f}")


# ══════════════════════════════════════════════════════════════════════
#  2.2 — SCALING FACTOR  (with vs without 1/√dk)
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttentionNoScale(MultiHeadAttention):
    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q, K, V = self.W_q(query), self.W_k(key), self.W_v(value)

        def sh(x):
            return x.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        Q, K, V = sh(Q), sh(K), sh(V)

        scores = torch.matmul(Q, K.transpose(-2, -1))   # NO /sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_w = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        self.attn_weights = attn_w

        out = torch.matmul(attn_w, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


def _patch_no_scale(model):
    for layer in list(model.encoder.layers) + list(model.decoder.layers):
        for attr in ['self_attn', 'cross_attn']:
            if not hasattr(layer, attr):
                continue
            old = getattr(layer, attr)
            new = MultiHeadAttentionNoScale(old.d_model, old.num_heads, old.dropout.p)
            new.load_state_dict(old.state_dict())
            setattr(layer, attr, new.to(DEVICE))
    return model


def _grad_hook(model, step, log_every=25):
    if step > 1000 or step % log_every != 0:
        return {}
    q_norms, k_norms = [], []
    for m in model.modules():
        if isinstance(m, (MultiHeadAttention, MultiHeadAttentionNoScale)):
            if m.W_q.weight.grad is not None:
                q_norms.append(m.W_q.weight.grad.norm().item())
            if m.W_k.weight.grad is not None:
                k_norms.append(m.W_k.weight.grad.norm().item())
    return ({'Wq_grad_norm': float(np.mean(q_norms)),
             'Wk_grad_norm': float(np.mean(k_norms))} if q_norms else {})


def exp_scaling_factor():
    print("\n=== 2.2  Scaling Factor Ablation ===")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_data(BASE_CONFIG)
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, BASE_CONFIG['label_smoothing'])

    for use_scale in [True, False]:
        name = "2.2_with_scale" if use_scale else "2.2_no_scale"
        wandb.init(project="da6401-a3", name=name,
                   config={**BASE_CONFIG, 'sqrt_dk_scaling': use_scale})
        model = build_model(len(src_vocab), len(tgt_vocab), BASE_CONFIG)
        if not use_scale:
            model = _patch_no_scale(model)
        opt, sched = make_opt_sched(model, BASE_CONFIG)
        train_model(model, train_loader, val_loader, BASE_CONFIG, opt, sched, loss_fn,
                    step_hook=lambda m, s: _grad_hook(m, s))
        bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=DEVICE)
        wandb.log({'test_bleu': bleu}); wandb.finish()
        print(f"  {'With' if use_scale else 'No'} scale — BLEU: {bleu:.2f}")


# ══════════════════════════════════════════════════════════════════════
#  2.3 — ATTENTION HEAD HEATMAPS
# ══════════════════════════════════════════════════════════════════════

def exp_attention_rollout():
    print("\n=== 2.3  Attention Head Visualization ===")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_data(BASE_CONFIG)
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, BASE_CONFIG['label_smoothing'])

    wandb.init(project="da6401-a3", name="2.3_attention_heads",
               config={**BASE_CONFIG, 'exp': '2.3'})

    model = build_model(len(src_vocab), len(tgt_vocab), BASE_CONFIG)
    opt, sched = make_opt_sched(model, BASE_CONFIG)
    train_model(model, train_loader, val_loader, BASE_CONFIG, opt, sched, loss_fn)

    # Extract attention from last encoder layer on one validation sentence
    model.eval()
    src, _ = next(iter(val_loader))
    src = src[:1].to(DEVICE)
    with torch.no_grad():
        _ = model.encode(src, make_src_mask(src, PAD_IDX))

    attn_w    = model.encoder.layers[-1].self_attn.attn_weights  # [1, H, S, S]
    num_heads = attn_w.size(1)
    seq_len   = attn_w.size(2)
    tokens    = [src_vocab.itos.get(idx.item(), '<unk>') for idx in src[0, :seq_len]]

    ncols = 4
    nrows = math.ceil(num_heads / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        w  = attn_w[0, h].cpu().numpy()[:len(tokens), :len(tokens)]
        im = ax.imshow(w, cmap='viridis', vmin=0, vmax=w.max(), aspect='auto')
        ax.set_title(f'Head {h+1}', fontsize=10)
        ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=6)
        ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046)

    for h in range(num_heads, len(axes)):
        axes[h].set_visible(False)

    plt.suptitle('Last Encoder Layer — Attention Heads', fontsize=13)
    plt.tight_layout()
    wandb.log({'attention_heads_heatmap': wandb.Image(fig)}); plt.close(fig)

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=DEVICE)
    wandb.log({'test_bleu': bleu}); wandb.finish()
    print(f"  Attention viz logged. BLEU: {bleu:.2f}")


# ══════════════════════════════════════════════════════════════════════
#  2.4 — SINUSOIDAL PE vs LEARNED PE
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(pos))


def exp_learned_pe():
    print("\n=== 2.4  Sinusoidal PE vs Learned PE ===")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_data(BASE_CONFIG)
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, BASE_CONFIG['label_smoothing'])

    for use_learned in [False, True]:
        name = "2.4_learned_pe" if use_learned else "2.4_sinusoidal_pe"
        wandb.init(project="da6401-a3", name=name,
                   config={**BASE_CONFIG, 'pe_type': 'learned' if use_learned else 'sinusoidal'})
        model = build_model(len(src_vocab), len(tgt_vocab), BASE_CONFIG,
                            use_learned_pe=use_learned)
        opt, sched = make_opt_sched(model, BASE_CONFIG)
        train_model(model, train_loader, val_loader, BASE_CONFIG, opt, sched, loss_fn)
        bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=DEVICE)
        wandb.log({'test_bleu': bleu}); wandb.finish()
        print(f"  {'Learned' if use_learned else 'Sinusoidal'} PE — BLEU: {bleu:.2f}")


# ══════════════════════════════════════════════════════════════════════
#  2.5 — LABEL SMOOTHING  ε=0.1 vs ε=0.0
# ══════════════════════════════════════════════════════════════════════

def _confidence(model, val_loader):
    model.eval()
    confs = []
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in,
                           make_src_mask(src, PAD_IDX),
                           make_tgt_mask(tgt_in, PAD_IDX))
            probs   = F.softmax(logits, dim=-1)
            correct = probs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
            mask    = tgt_out != PAD_IDX
            confs.extend(correct[mask].cpu().tolist())
            if len(confs) > 5000:
                break
    return {'pred_confidence': float(np.mean(confs))}


def exp_label_smoothing():
    print("\n=== 2.5  Label Smoothing ===")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_data(BASE_CONFIG)

    for eps in [0.0, 0.1]:
        name = f"2.5_smoothing_{eps}"
        cfg  = {**BASE_CONFIG, 'label_smoothing': eps}
        wandb.init(project="da6401-a3", name=name,
                   config={**cfg, 'loss_type': 'CE' if eps == 0 else 'label_smooth'})
        model   = build_model(len(src_vocab), len(tgt_vocab), cfg)
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=eps)
        opt, sched = make_opt_sched(model, cfg)
        train_model(model, train_loader, val_loader, cfg, opt, sched, loss_fn,
                    epoch_extra=lambda m, vl: _confidence(m, vl))
        bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=DEVICE)
        wandb.log({'test_bleu': bleu}); wandb.finish()
        print(f"  ε={eps} — BLEU: {bleu:.2f}")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    'noam_vs_fixed':     exp_noam_vs_fixed,
    'scaling_factor':    exp_scaling_factor,
    'attention_rollout': exp_attention_rollout,
    'learned_pe':        exp_learned_pe,
    'label_smoothing':   exp_label_smoothing,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True,
                        choices=list(EXPERIMENTS.keys()) + ['all'])
    args = parser.parse_args()

    if args.exp == 'all':
        for name, fn in EXPERIMENTS.items():
            print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
            fn()
    else:
        EXPERIMENTS[args.exp]()
