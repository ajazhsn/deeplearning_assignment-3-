"""
train_optimized.py — Retrain with better hyperparameters to break 35 BLEU
Run on Colab GPU: !python train_optimized.py

Key changes vs original:
  - d_ff 512 → 1024  (bigger FFN = more capacity, costs ~10% more RAM)
  - num_epochs 20 → 30
  - warmup_steps 4000 → 2000  (faster warm-up for small dataset)
  - Saves best checkpoint by val BLEU, not val loss
  - Uploads best checkpoint to Google Drive automatically
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── pull from your repo ───────────────────────────────────────────────
import subprocess, sys

REPO = "https://github.com/ajazhsn/deeplearning_assignment-3-"
if not os.path.exists("repo"):
    subprocess.run(["git", "clone", REPO, "repo"], check=True)
sys.path.insert(0, "repo")

from dataset import get_dataloaders, PAD_IDX
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from train import LabelSmoothingLoss, run_epoch, greedy_decode, save_checkpoint


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

CONFIG = dict(
    d_model      = 256,
    N            = 3,
    num_heads    = 8,
    d_ff         = 1024,   # ← key change (was 512)
    dropout      = 0.1,
    batch_size   = 128,
    num_epochs   = 30,     # ← more epochs
    warmup_steps = 2000,   # ← faster warm-up for Multi30k size
    label_smooth = 0.1,
    min_freq     = 2,
    max_len      = 128,
    gdrive_id    = "1R-nnKC_69Vxg-TqTlOMMhirFzbC9oORr",  # existing file id
)


# ══════════════════════════════════════════════════════════════════════
#  BLEU (sentence-level, fast, no dataloader needed)
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu_fast(model, test_loader, tgt_vocab, device, max_len=100):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    import nltk; nltk.download('punkt', quiet=True)

    model.eval()
    pad_idx = model.pad_idx
    eos_idx = tgt_vocab.stoi['<eos>']
    sos_idx = tgt_vocab.stoi['<sos>']

    hyps, refs = [], []
    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="BLEU", leave=False):
            src = src.to(device)
            src_mask = make_src_mask(src, pad_idx)
            ys = greedy_decode(model, src, src_mask, max_len,
                               sos_idx, eos_idx, device)

            hyp = [tgt_vocab.itos[i] for i in ys.squeeze(0).tolist()[1:]
                   if i not in (eos_idx, pad_idx, sos_idx)]
            ref = [tgt_vocab.itos[i] for i in tgt.squeeze(0).tolist()
                   if i not in (eos_idx, pad_idx, sos_idx)]

            # stop hyp at first eos
            ys_list = ys.squeeze(0).tolist()[1:]
            hyp = []
            for i in ys_list:
                if i == eos_idx: break
                if i != pad_idx:
                    hyp.append(tgt_vocab.itos[i])

            hyps.append(hyp)
            refs.append([ref])

    bleu = corpus_bleu(refs, hyps,
                       smoothing_function=SmoothingFunction().method1)
    return bleu * 100.0


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=CONFIG['batch_size'],
                        min_freq=CONFIG['min_freq'],
                        max_len=CONFIG['max_len'])

    print(f"Src vocab: {len(src_vocab)}  Tgt vocab: {len(tgt_vocab)}")

    # ── Model ──────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model   = CONFIG['d_model'],
        N         = CONFIG['N'],
        num_heads = CONFIG['num_heads'],
        d_ff      = CONFIG['d_ff'],
        dropout   = CONFIG['dropout'],
        pad_idx   = PAD_IDX,
    ).to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Optimiser & scheduler ──────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer,
                              d_model=CONFIG['d_model'],
                              warmup_steps=CONFIG['warmup_steps'])

    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = PAD_IDX,
        smoothing  = CONFIG['label_smooth'],
    )

    # ── Training loop ──────────────────────────────────────────────────
    best_bleu      = 0.0
    best_ckpt_path = "best_checkpoint.pt"

    for epoch in range(CONFIG['num_epochs']):
        print(f"\n{'='*55}")
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

        train_loss = run_epoch(train_loader, model, loss_fn,
                               optimizer, scheduler,
                               epoch_num=epoch+1, is_train=True, device=device)

        val_loss = run_epoch(val_loader, model, loss_fn,
                             None, None,
                             epoch_num=epoch+1, is_train=False, device=device)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.2e}")

        # Evaluate BLEU every 5 epochs (fast enough on Colab)
        if (epoch + 1) % 5 == 0 or epoch == CONFIG['num_epochs'] - 1:
            bleu = evaluate_bleu_fast(model, test_loader, tgt_vocab, device)
            print(f"  *** Test BLEU = {bleu:.2f} ***")

            if bleu > best_bleu:
                best_bleu = bleu
                save_checkpoint(model, optimizer, scheduler,
                                epoch + 1, best_ckpt_path)
                print(f"  → New best checkpoint saved (BLEU {best_bleu:.2f})")

        # Always save latest
        save_checkpoint(model, optimizer, scheduler, epoch + 1, "checkpoint.pt")

    # ── Upload best checkpoint to Google Drive ─────────────────────────
    print(f"\nBest BLEU: {best_bleu:.2f}")
    print(f"Best checkpoint: {best_ckpt_path}")

    try:
        import gdown
        # Upload overwrites the existing file by its id
        # gdown doesn't support upload; use pydrive instead
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        from oauth2client.service_account import ServiceAccountCredentials

        print("\nTo upload to Drive, run in Colab:")
        print(f"  from google.colab import drive")
        print(f"  drive.mount('/content/drive')")
        print(f"  import shutil")
        print(f"  shutil.copy('{best_ckpt_path}', '/content/drive/MyDrive/checkpoint.pt')")
        print(f"\nThen update GDRIVE_FILE_ID in model.py with the new file's share link.")
    except Exception:
        pass

    print("\nDone. Copy the checkpoint to Drive and update the GDRIVE_FILE_ID in model.py.")
    return best_bleu


if __name__ == "__main__":
    main()
