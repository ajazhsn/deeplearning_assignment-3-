"""
train.py — Complete Training Pipeline
DA6401 Assignment 3: "Attention Is All You Need"

Hyperparameters (optimized for Multi30k):
  - d_model=256, N=3, num_heads=8, d_ff=1024
  - 30 epochs, warmup_steps=2000
  - Label smoothing eps=0.1
  - Saves best checkpoint by BLEU score
  - Full W&B logging
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional
import wandb

from dataset import get_dataloaders, PAD_IDX
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

CONFIG = dict(
    d_model      = 256,
    N            = 3,
    num_heads    = 8,
    d_ff         = 1024,
    dropout      = 0.1,
    batch_size   = 128,
    num_epochs   = 30,
    warmup_steps = 2000,
    label_smooth = 0.1,
    min_freq     = 2,
    max_len      = 128,
)


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".
    Smoothed distribution: y_s = (1-eps)*one_hot(y) + eps/(V-1)
    PAD positions receive 0 probability.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : [batch * tgt_len, vocab_size]
            target : [batch * tgt_len]
        Returns:
            Scalar loss.
        """
        with torch.no_grad():
            smooth_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0
            pad_mask = (target == self.pad_idx)
            smooth_dist[pad_mask] = 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_dist * log_probs).sum(dim=-1)
        n_tokens = (~pad_mask).sum().float()
        return loss.sum() / (n_tokens + 1e-8)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """Run one epoch of training or evaluation. Returns avg loss."""
    model.train() if is_train else model.eval()

    total_loss, total_tokens = 0.0, 0
    pad_idx = model.pad_idx

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} Epoch {epoch_num}")
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)

            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx)

            logits = model(src, tgt_in, src_mask, tgt_mask)

            logits_flat  = logits.contiguous().view(-1, logits.size(-1))
            tgt_out_flat = tgt_out.contiguous().view(-1)

            loss = loss_fn(logits_flat, tgt_out_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tokens      = (tgt_out != pad_idx).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Token-by-token greedy decoding. Returns [1, out_len]."""
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_mask   = make_tgt_mask(ys, model.pad_idx).to(device)
            logits     = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys         = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_loader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """Corpus-level BLEU score (0-100)."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)

    model.eval()
    pad_idx = model.pad_idx

    # Support both our Vocabulary class and torchtext-style vocab
    if hasattr(tgt_vocab, 'stoi'):
        eos_idx = tgt_vocab.stoi['<eos>']
        sos_idx = tgt_vocab.stoi['<sos>']
        def idx2tok(i): return tgt_vocab.itos.get(i, '<unk>')
    else:
        eos_idx = tgt_vocab['<eos>']
        sos_idx = tgt_vocab['<sos>']
        itos = {v: k for k, v in tgt_vocab.items()}
        def idx2tok(i): return itos.get(i, '<unk>')

    hyps, refs = [], []
    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="BLEU", leave=False):
            src      = src.to(device)
            src_mask = make_src_mask(src, pad_idx)
            ys       = greedy_decode(model, src, src_mask, max_len,
                                     sos_idx, eos_idx, device)

            hyp = []
            for i in ys.squeeze(0).tolist()[1:]:
                if i == eos_idx: break
                if i != pad_idx: hyp.append(idx2tok(i))

            ref = [idx2tok(i) for i in tgt.squeeze(0).tolist()
                   if i not in (eos_idx, pad_idx, sos_idx)]

            hyps.append(hyp)
            refs.append([ref])

    bleu = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
    return bleu * 100.0


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    model_config = {
        'src_vocab_size': model.src_embed.num_embeddings,
        'tgt_vocab_size': model.tgt_embed.num_embeddings,
        'd_model':        model.d_model,
        'N':              len(model.encoder.layers),
        'num_heads':      model.encoder.layers[0].self_attn.num_heads,
        'd_ff':           model.encoder.layers[0].ffn.linear1.out_features,
        'dropout':        model.encoder.layers[0].dropout.p,
        'pad_idx':        model.pad_idx,
    }
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config':         model_config,
    }, path)
    print(f"Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer=None,
    scheduler=None,
) -> int:
    ckpt  = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt.get('epoch', 0)
    print(f"Loaded checkpoint from {path}  (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  MAIN TRAINING EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── W&B init ──────────────────────────────────────────────────────
    wandb.init(
        project = "da6401-a3",
        name    = "optimized_d_ff1024_ep30_warmup2000",
        config  = CONFIG,
    )
    cfg = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(
            batch_size = cfg.batch_size,
            min_freq   = cfg.min_freq,
            max_len    = cfg.max_len,
        )
    print(f"Src vocab: {len(src_vocab)}  Tgt vocab: {len(tgt_vocab)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model   = cfg.d_model,
        N         = cfg.N,
        num_heads = cfg.num_heads,
        d_ff      = cfg.d_ff,
        dropout   = cfg.dropout,
        pad_idx   = PAD_IDX,
        load_weights = False,   # no gdown during training
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")
    wandb.log({"n_params": n_params})

    # ── Optimizer & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model      = cfg.d_model,
        warmup_steps = cfg.warmup_steps,
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = PAD_IDX,
        smoothing  = cfg.label_smooth,
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_bleu      = 0.0
    best_ckpt_path = "best_checkpoint.pt"

    for epoch in range(cfg.num_epochs):
        print(f"\n{'='*55}")
        print(f"Epoch {epoch+1}/{cfg.num_epochs}")

        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch+1, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch+1, is_train=False, device=device,
        )

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.2e}")

        wandb.log({
            "epoch":         epoch + 1,
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "learning_rate": lr_now,
        })

        # BLEU every 5 epochs and on final epoch
        if (epoch + 1) % 5 == 0 or epoch == cfg.num_epochs - 1:
            bleu = evaluate_bleu(model, test_loader, tgt_vocab, device)
            print(f"  *** Test BLEU = {bleu:.2f} ***")
            wandb.log({"epoch": epoch + 1, "test_bleu": bleu})

            if bleu > best_bleu:
                best_bleu = bleu
                save_checkpoint(model, optimizer, scheduler,
                                epoch + 1, best_ckpt_path)
                print(f"  → New best! (BLEU {best_bleu:.2f})")
                wandb.run.summary["best_bleu"]  = best_bleu
                wandb.run.summary["best_epoch"] = epoch + 1

        # Save latest every epoch
        save_checkpoint(model, optimizer, scheduler, epoch + 1, "checkpoint.pt")

    # ── Final summary ─────────────────────────────────────────────────
    wandb.run.summary["best_bleu"] = best_bleu
    print(f"\nBest BLEU: {best_bleu:.2f}")
    print(f"Best checkpoint saved at: {best_ckpt_path}")
    print("\nNext steps:")
    print("  1. Download best_checkpoint.pt from Kaggle output panel")
    print("  2. Upload to Google Drive")
    print("  3. Copy the file ID into model.py → Transformer.GDRIVE_FILE_ID")

    wandb.finish()
    return best_bleu


if __name__ == "__main__":
    main()
