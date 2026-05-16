"""
ablations.py — W&B Report Experiments (Sections 2.1–2.5)
DA6401 Assignment 3

Run each experiment separately:
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
    make_src_mask, make_tgt_mask, scaled_dot_product_attention
)
from dataset import get_dataloaders, PAD_IDX
from lr_scheduler import NoamScheduler, get_lr_history
from train import run_epoch, LabelSmoothingLoss, evaluate_bleu, save_checkpoint, load_checkpoint


# ─── Shared config ────────────────────────────────────────────────────
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


def build_model(src_vocab_size, tgt_vocab_size, cfg, use_learned_pe=False):
    """Build Transformer, optionally replacing sinusoidal PE with learned."""
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg['d_model'],
        N=cfg['N'],
        num_heads=cfg['num_heads'],
        d_ff=cfg['d_ff'],
        dropout=cfg['dropout'],
        pad_idx=PAD_IDX,
    ).to(DEVICE)

    if use_learned_pe:
        # Replace sinusoidal PE with learned embeddings
        max_len = cfg['max_len'] + 10
        learned_pe = LearnedPositionalEncoding(cfg['d_model'], cfg['dropout'], max_len)
        model.pos_enc = learned_pe.to(DEVICE)

    return model


def train_and_log(model, train_loader, val_loader, cfg, extra_log=None):
    """Train for cfg['num_epochs'], log to W&B, return best val_loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps'])

    # Rebuild tgt vocab size from model
    tgt_vocab_size = model.tgt_embed.num_embeddings
    loss_fn = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, cfg['label_smoothing'])

    best_val_loss = float('inf')
    for epoch in range(cfg['num_epochs']):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                               epoch_num=epoch, is_train=True, device=DEVICE)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                             epoch_num=epoch, is_train=False, device=DEVICE)
        lr = optimizer.param_groups[0]['lr']

        log_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': lr,
        }
        if extra_log:
            log_dict.update(extra_log(model, epoch))

        wandb.log(log_dict)
        print(f"  Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, lr={lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, 'best_ablation.pt')

    return best_val_loss


# ══════════════════════════════════════════════════════════════════════
# Experiment 2.1 — Noam vs Fixed LR
# ══════════════════════════════════════════════════════════════════════

def exp_noam_vs_fixed():
    """Train with Noam scheduler vs fixed LR=1e-4."""
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'],
        min_freq=BASE_CONFIG['min_freq'],
        max_len=BASE_CONFIG['max_len'],
    )

    # Also visualize the LR schedule
    lrs = get_lr_history(BASE_CONFIG['d_model'], BASE_CONFIG['warmup_steps'], 20000)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lrs)
    ax.axvline(BASE_CONFIG['warmup_steps'], color='red', linestyle='--',
               label=f"warmup={BASE_CONFIG['warmup_steps']}")
    ax.set(xlabel='Step', ylabel='LR', title='Noam LR Schedule')
    ax.legend()
    wandb.log({'noam_lr_curve': wandb.Image(fig)})
    plt.close(fig)

    # --- Noam run ---
    wandb.init(project="da6401-a3", name="noam_scheduler", config={**BASE_CONFIG, 'exp': '2.1_noam'})
    model = build_model(len(src_vocab), len(tgt_vocab), BASE_CONFIG)
    train_and_log(model, train_loader, val_loader, BASE_CONFIG)
    wandb.finish()

    # --- Fixed LR run ---
    wandb.init(project="da6401-a3", name="fixed_lr_1e4", config={**BASE_CONFIG, 'exp': '2.1_fixed'})
    model = build_model(len(src_vocab), len(tgt_vocab), BASE_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    tgt_vocab_size = len(tgt_vocab)
    loss_fn = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, BASE_CONFIG['label_smoothing'])
    for epoch in range(BASE_CONFIG['num_epochs']):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, None,
                               epoch_num=epoch, is_train=True, device=DEVICE)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                             epoch_num=epoch, is_train=False, device=DEVICE)
        wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss,
                   'learning_rate': 1e-4})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# Experiment 2.2 — Ablation: √(1/dk) scaling factor
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttentionNoScale(MultiHeadAttention):
    """MHA without the 1/√dk scaling."""

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        def split_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # NO scaling: scores = Q @ K^T (not divided by sqrt(d_k))
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_w = F.softmax(scores, dim=-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        self.attn_weights = attn_w

        out = torch.matmul(attn_w, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out)


def log_grad_norms(model, epoch):
    """Log mean gradient norm of Q and K weight matrices."""
    q_norms, k_norms = [], []
    for module in model.modules():
        if isinstance(module, (MultiHeadAttention, MultiHeadAttentionNoScale)):
            if module.W_q.weight.grad is not None:
                q_norms.append(module.W_q.weight.grad.norm().item())
            if module.W_k.weight.grad is not None:
                k_norms.append(module.W_k.weight.grad.norm().item())
    return {
        'mean_q_grad_norm': float(np.mean(q_norms)) if q_norms else 0.0,
        'mean_k_grad_norm': float(np.mean(k_norms)) if k_norms else 0.0,
    }


def exp_scaling_factor():
    train_loader, val_loader, _, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'], min_freq=BASE_CONFIG['min_freq'], max_len=BASE_CONFIG['max_len']
    )

    for use_scale in [True, False]:
        name = "with_scale" if use_scale else "no_scale"
        wandb.init(project="da6401-a3", name=name, config={**BASE_CONFIG, 'exp': f'2.2_{name}'})

        model = build_model(len(src_vocab), len(tgt_vocab), BASE_CONFIG)

        if not use_scale:
            # Patch all MHA modules in encoder/decoder
            for layer in list(model.encoder.layers) + list(model.decoder.layers):
                for attr in ['self_attn', 'cross_attn'] if hasattr(layer, 'cross_attn') else ['self_attn']:
                    old = getattr(layer, attr)
                    new = MultiHeadAttentionNoScale(old.d_model, old.num_heads)
                    new.load_state_dict(old.state_dict())
                    setattr(layer, attr, new.to(DEVICE))

        train_and_log(model, train_loader, val_loader, BASE_CONFIG,
                      extra_log=lambda m, e: log_grad_norms(m, e))
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# Experiment 2.3 — Attention Rollout & Head Specialization
# ══════════════════════════════════════════════════════════════════════

def exp_attention_rollout():
    from train import greedy_decode
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'], min_freq=BASE_CONFIG['min_freq'], max_len=BASE_CONFIG['max_len']
    )

    wandb.init(project="da6401-a3", name="attention_rollout", config={**BASE_CONFIG, 'exp': '2.3'})
    model = build_model(len(src_vocab), len(tgt_vocab), BASE_CONFIG)

    # Train first
    train_and_log(model, train_loader, val_loader, BASE_CONFIG)

    # Pick one validation sentence
    model.eval()
    for src, tgt in val_loader:
        src = src[:1].to(DEVICE)
        tgt = tgt[:1].to(DEVICE)
        break

    src_mask = make_src_mask(src, PAD_IDX)
    with torch.no_grad():
        _ = model.encode(src, src_mask)

    # Get attention weights from last encoder layer
    last_enc_layer = model.encoder.layers[-1]
    attn_weights = last_enc_layer.self_attn.attn_weights  # [1, heads, seq, seq]

    num_heads = attn_weights.size(1)
    seq_len = attn_weights.size(2)

    # Token labels
    itos = src_vocab.itos
    src_tokens = [itos.get(idx.item(), '<unk>') for idx in src[0]][:seq_len]

    # Plot one heatmap per head
    fig, axes = plt.subplots(2, num_heads // 2, figsize=(4 * num_heads // 2, 8))
    axes = axes.flatten()
    for h in range(num_heads):
        ax = axes[h]
        w = attn_weights[0, h].cpu().numpy()[:len(src_tokens), :len(src_tokens)]
        im = ax.imshow(w, cmap='viridis', aspect='auto')
        ax.set_title(f'Head {h+1}')
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=6)
        ax.set_yticks(range(len(src_tokens)))
        ax.set_yticklabels(src_tokens, fontsize=6)
        plt.colorbar(im, ax=ax)

    plt.suptitle('Last Encoder Layer — Attention Heads')
    plt.tight_layout()
    wandb.log({'attention_heads': wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# Experiment 2.4 — Sinusoidal PE vs Learned PE
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings (alternative to sinusoidal)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(positions))


def exp_learned_pe():
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'], min_freq=BASE_CONFIG['min_freq'], max_len=BASE_CONFIG['max_len']
    )

    for use_learned in [False, True]:
        name = "learned_pe" if use_learned else "sinusoidal_pe"
        wandb.init(project="da6401-a3", name=name, config={**BASE_CONFIG, 'exp': f'2.4_{name}'})

        model = build_model(len(src_vocab), len(tgt_vocab), BASE_CONFIG, use_learned_pe=use_learned)
        train_and_log(model, train_loader, val_loader, BASE_CONFIG)

        # Compute BLEU on test set
        bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=DEVICE)
        wandb.log({'test_bleu': bleu})
        print(f"  {name} BLEU: {bleu:.2f}")
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# Experiment 2.5 — Label Smoothing ε=0.1 vs ε=0.0
# ══════════════════════════════════════════════════════════════════════

def log_prediction_confidence(model, val_loader, loss_fn):
    """Compute average softmax probability of the correct token (confidence)."""
    model.eval()
    confidences = []
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_in, PAD_IDX)
            logits = model(src, tgt_in, src_mask, tgt_mask)
            probs = F.softmax(logits, dim=-1)
            # Gather prob of correct token
            correct_probs = probs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
            # Only non-pad
            mask = (tgt_out != PAD_IDX)
            confidences.extend(correct_probs[mask].cpu().tolist())
            if len(confidences) > 5000:
                break
    return float(np.mean(confidences))


def exp_label_smoothing():
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=BASE_CONFIG['batch_size'], min_freq=BASE_CONFIG['min_freq'], max_len=BASE_CONFIG['max_len']
    )

    for eps in [0.0, 0.1]:
        name = f"smoothing_{eps}"
        cfg = {**BASE_CONFIG, 'label_smoothing': eps}
        wandb.init(project="da6401-a3", name=name, config={**cfg, 'exp': f'2.5_{name}'})

        model = build_model(len(src_vocab), len(tgt_vocab), cfg)
        tgt_vocab_size = len(tgt_vocab)
        loss_fn = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=eps)

        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps'])

        for epoch in range(cfg['num_epochs']):
            train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                                   epoch_num=epoch, is_train=True, device=DEVICE)
            val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                                 epoch_num=epoch, is_train=False, device=DEVICE)
            confidence = log_prediction_confidence(model, val_loader, loss_fn)
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'pred_confidence': confidence,
                'learning_rate': optimizer.param_groups[0]['lr'],
            })

        bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=DEVICE)
        wandb.log({'test_bleu': bleu})
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True,
                        choices=['noam_vs_fixed', 'scaling_factor', 'attention_rollout',
                                 'learned_pe', 'label_smoothing', 'all'])
    args = parser.parse_args()

    experiments = {
        'noam_vs_fixed':    exp_noam_vs_fixed,
        'scaling_factor':   exp_scaling_factor,
        'attention_rollout': exp_attention_rollout,
        'learned_pe':       exp_learned_pe,
        'label_smoothing':  exp_label_smoothing,
    }

    if args.exp == 'all':
        for name, fn in experiments.items():
            print(f"\n{'='*60}")
            print(f"Running experiment: {name}")
            fn()
    else:
        experiments[args.exp]()
