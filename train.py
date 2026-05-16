"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device) → Tensor
  evaluate_bleu(model, test_dataloader, tgt_vocab, device)                       → float
  save_checkpoint(model, optimizer, scheduler, epoch, path)                      → None
  load_checkpoint(path, model, optimizer, scheduler)                             → int
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import math
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".
    Smoothed distribution: y_s = (1-ε)*one_hot(y) + ε/(V-1)
    PAD positions receive 0 probability (excluded from loss).

    Args:
        vocab_size : Number of output classes.
        pad_idx    : Index of <pad> token — excluded from loss.
        smoothing  : Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
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
        # Build smoothed target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # Zero out pad positions in the target distribution
            smooth_dist[:, self.pad_idx] = 0.0
            # Mask rows where the target itself is pad
            pad_mask = (target == self.pad_idx)
            smooth_dist[pad_mask] = 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        # KL divergence: sum(-y_smooth * log_probs)
        loss = -(smooth_dist * log_probs).sum(dim=-1)

        # Average over non-pad tokens
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
    """
    Run one epoch of training or evaluation.

    Returns:
        avg_loss : Average loss over the epoch.
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    pad_idx = model.pad_idx

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} Epoch {epoch_num}")
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing: decoder input is tgt[:-1], target is tgt[1:]
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            # logits: [batch, tgt_len-1, vocab_size]

            # Flatten for loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat  = logits.contiguous().view(-1, vocab_size)
            tgt_out_flat = tgt_out.contiguous().view(-1)

            loss = loss_fn(logits_flat, tgt_out_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (important for transformer stability)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Count non-pad tokens for logging
            n_tokens = (tgt_out != pad_idx).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


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
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : [1, src_len]
        src_mask     : [1, 1, 1, src_len]
        max_len      : Max tokens to generate.
        start_symbol : <sos> index.
        end_symbol   : <eos> index.
        device       : device string.

    Returns:
        ys : [1, out_len] — includes start_symbol, stops at end_symbol.
    """
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

    # Start with <sos>
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, model.pad_idx).to(device)

        with torch.no_grad():
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            # logits: [1, cur_len, vocab_size]

        # Take the last timestep, pick argmax
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        ys = torch.cat([ys, next_token], dim=1)

        if next_token.item() == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    except ImportError:
        raise ImportError("Run: pip install nltk")

    model.eval()
    pad_idx = model.pad_idx

    # Determine vocab lookup method
    if hasattr(tgt_vocab, 'itos'):
        # torchtext-style
        def idx2tok(idx):
            return tgt_vocab.itos[idx]
        eos_idx = tgt_vocab.stoi.get('<eos>', 3)
        sos_idx = tgt_vocab.stoi.get('<sos>', 2)
    elif hasattr(tgt_vocab, 'lookup_token'):
        def idx2tok(idx):
            return tgt_vocab.lookup_token(idx)
        eos_idx = tgt_vocab['<eos>']
        sos_idx = tgt_vocab['<sos>']
    elif hasattr(tgt_vocab, 'stoi'):
        # Our Vocabulary class
        def idx2tok(idx):
            return tgt_vocab.itos.get(idx, '<unk>')
        eos_idx = tgt_vocab.stoi.get('<eos>', 3)
        sos_idx = tgt_vocab.stoi.get('<sos>', 2)
    else:
        # Assume dict
        itos = {v: k for k, v in tgt_vocab.items()}
        def idx2tok(idx):
            return itos.get(idx, '<unk>')
        eos_idx = tgt_vocab.get('<eos>', 3)
        sos_idx = tgt_vocab.get('<sos>', 2)

    hypotheses = []
    references = []

    for src, tgt in tqdm(test_dataloader, desc="BLEU eval"):
        src = src.to(device)
        tgt = tgt.to(device)

        src_mask = make_src_mask(src, pad_idx)

        ys = greedy_decode(
            model, src, src_mask,
            max_len=max_len,
            start_symbol=sos_idx,
            end_symbol=eos_idx,
            device=device,
        )

        # Decode hypothesis: skip <sos>, stop at <eos>
        hyp_tokens = []
        for idx in ys.squeeze(0).tolist()[1:]:  # skip <sos>
            if idx == eos_idx:
                break
            if idx != pad_idx:
                hyp_tokens.append(idx2tok(idx))

        # Decode reference: skip <sos> and <eos>
        ref_tokens = []
        for idx in tgt.squeeze(0).tolist():
            if idx in (sos_idx, pad_idx):
                continue
            if idx == eos_idx:
                break
            ref_tokens.append(idx2tok(idx))

        hypotheses.append(hyp_tokens)
        references.append([ref_tokens])

    smoother = SmoothingFunction().method1
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoother)
    return bleu * 100.0  # return 0–100


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
    """Save model + optimizer + scheduler state to disk."""
    # Collect model config for reconstruction
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
    print(f"Checkpoint saved to {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """Restore model (and optionally optimizer/scheduler) from disk."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """Full training experiment with W&B logging."""
    import wandb
    from dataset import get_dataloaders, PAD_IDX
    from lr_scheduler import NoamScheduler

    # ── Hyperparameters ──────────────────────────────────────────────
    config = {
        'd_model':      256,      # smaller than paper's 512 for Multi30k
        'N':            3,        # 3 encoder/decoder layers
        'num_heads':    8,
        'd_ff':         512,
        'dropout':      0.1,
        'batch_size':   128,
        'num_epochs':   20,
        'warmup_steps': 4000,
        'label_smoothing': 0.1,
        'min_freq':     2,
        'max_len':      128,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    # ── Data ─────────────────────────────────────────────────────────
    print("Loading data...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=cfg.batch_size,
        min_freq=cfg.min_freq,
        max_len=cfg.max_len,
    )

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"Src vocab: {src_vocab_size}, Tgt vocab: {tgt_vocab_size}")

    # ── Model ────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pad_idx=PAD_IDX,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    wandb.log({'n_params': n_params})

    # ── Optimizer & Scheduler ────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    # ── Loss ─────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=PAD_IDX,
        smoothing=cfg.label_smoothing,
    )

    # ── Training loop ────────────────────────────────────────────────
    best_val_loss = float('inf')

    for epoch in range(cfg.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{cfg.num_epochs}")

        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        wandb.log({
            'epoch':      epoch + 1,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'learning_rate': current_lr,
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, 'best_checkpoint.pt')

        # Save latest checkpoint every epoch
        save_checkpoint(model, optimizer, scheduler, epoch, 'checkpoint.pt')

    # ── Final BLEU evaluation ────────────────────────────────────────
    print("\nEvaluating BLEU on test set...")
    # Load best checkpoint
    load_checkpoint('best_checkpoint.pt', model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({'test_bleu': bleu})

    wandb.finish()
    return bleu


if __name__ == "__main__":
    run_training_experiment()
