"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
    Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : shape (..., seq_q, d_k)
        K    : shape (..., seq_k, d_k)
        V    : shape (..., seq_k, d_v)
        mask : BoolTensor broadcastable to (..., seq_q, seq_k).
               True positions are masked out (set to -inf).

    Returns:
        output : shape (..., seq_q, d_v)
        attn_w : shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    # Scaled scores: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    # Handle all-masked rows (nan → 0)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for encoder.
    Returns BoolTensor [batch, 1, 1, src_len]; True = PAD (masked out).
    """
    # src: [batch, src_len]
    src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]
    return src_mask


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal mask for decoder.
    Returns BoolTensor [batch, 1, tgt_len, tgt_len]; True = masked out.
    """
    batch_size, tgt_len = tgt.size()
    # Causal mask: upper triangle (excluding diagonal) is True
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool), diagonal=1
    )  # [tgt_len, tgt_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]

    # Padding mask: True where pad
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]

    # Combine: mask out if either is True
    tgt_mask = causal_mask | pad_mask  # broadcasts to [batch, 1, tgt_len, tgt_len]
    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
    Does NOT use torch.nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Store last attention weights for visualization
        self.attn_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : [batch, seq_q, d_model]
            key   : [batch, seq_k, d_model]
            value : [batch, seq_k, d_model]
            mask  : BoolTensor broadcastable to [batch, num_heads, seq_q, seq_k]

        Returns:
            output : [batch, seq_q, d_model]
        """
        batch_size = query.size(0)

        # 1. Linear projections
        Q = self.W_q(query)  # [batch, seq_q, d_model]
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into heads: [batch, num_heads, seq, d_k]
        def split_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # 3. Scaled dot-product attention
        attn_out, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        self.attn_weights = attn_w  # store for visualization

        # 4. Concatenate heads: [batch, seq_q, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. Final linear
        output = self.W_o(attn_out)
        return output


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Stored as a non-trainable buffer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build [max_len, d_model] table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]
    Uses Post-LayerNorm (as in the original paper).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-layer with residual + layer norm
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN sub-layer with residual + layer norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    x → [Masked Self-Attn → Add & Norm]
      → [Cross-Attn(memory) → Add & Norm]
      → [FFN → Add & Norm]
    Uses Post-LayerNorm.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Masked self-attention
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # 2. Cross-attention over encoder memory
        attn2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # 3. FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        pad_idx:   int   = 1,
        checkpoint_path: str = "checkpoint.pt",
    ) -> None:
        super().__init__()

        import spacy
        from datasets import load_dataset
        from collections import Counter

        # ============================================================
        # 1. LOAD TOKENIZERS
        # ============================================================

       try:
          self.de_nlp = spacy.load("de_core_news_sm")
      except:
          self.de_nlp = spacy.blank("de")
      
      try:
          self.en_nlp = spacy.load("en_core_web_sm")
      except:
          self.en_nlp = spacy.blank("en")

        # ============================================================
        # 2. SPECIAL TOKENS
        # ============================================================

        PAD_IDX = 1
        UNK_IDX = 0
        SOS_IDX = 2
        EOS_IDX = 3

        SPECIAL = ['<unk>', '<pad>', '<sos>', '<eos>']

        self.pad_idx = PAD_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX

        # ============================================================
        # 3. TOKENIZERS
        # ============================================================

        def tokenize_de(text):
            return [tok.text.lower() for tok in self.de_nlp.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text.lower() for tok in self.en_nlp.tokenizer(text)]

        # ============================================================
        # 4. BUILD VOCAB FROM DATASET
        # ============================================================

        dataset = load_dataset("bentrevett/multi30k", split="train")

        de_tokens = [
            tok
            for ex in dataset
            for tok in tokenize_de(ex["de"])
        ]

        en_tokens = [
            tok
            for ex in dataset
            for tok in tokenize_en(ex["en"])
        ]

        def build_vocab(tokens, min_freq=2):
            counter = Counter(tokens)

            stoi = {tok: i for i, tok in enumerate(SPECIAL)}
            itos = {i: tok for i, tok in enumerate(SPECIAL)}

            idx = len(SPECIAL)

            for tok, cnt in sorted(counter.items()):
                if cnt >= min_freq and tok not in stoi:
                    stoi[tok] = idx
                    itos[idx] = tok
                    idx += 1

            return stoi, itos

        self.src_stoi, self.src_itos = build_vocab(de_tokens)
        self.tgt_stoi, self.tgt_itos = build_vocab(en_tokens)

        src_vocab_size = len(self.src_stoi)
        tgt_vocab_size = len(self.tgt_stoi)

        self.d_model = d_model

        # ============================================================
        # 5. MODEL ARCHITECTURE
        # ============================================================

        self.src_embed = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        self.tgt_embed = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        self.pos_enc = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # ============================================================
        # 6. INITIALIZE WEIGHTS
        # ============================================================

        self._init_weights()

        # ============================================================
        # 7. DOWNLOAD + LOAD CHECKPOINT
        # ============================================================

        GDRIVE_FILE_ID = "1R-nnKC_69Vxg-TqTlOMMhirFzbC9oORr"

        if (
            checkpoint_path is not None
            and GDRIVE_FILE_ID != "YOUR_FILE_ID_HERE"
        ):
            try:
                import gdown

                if not os.path.exists(checkpoint_path):
                    print("Downloading checkpoint...")
                    gdown.download(
                        id=GDRIVE_FILE_ID,
                        output=checkpoint_path,
                        quiet=False
                    )

                ckpt = torch.load(
                    checkpoint_path,
                    map_location="cpu"
                )

                self.load_state_dict(
                    ckpt["model_state_dict"]
                )

                print("Weights loaded successfully.")

            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")

    # ================================================================
    # WEIGHT INITIALIZATION
    # ================================================================

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ================================================================
    # AUTOGRADER HOOKS
    # ================================================================

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor
    ) -> torch.Tensor:

        x = self.pos_enc(
            self.src_embed(src) * math.sqrt(self.d_model)
        )

        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        x = self.pos_enc(
            self.tgt_embed(tgt) * math.sqrt(self.d_model)
        )

        x = self.decoder(
            x,
            memory,
            src_mask,
            tgt_mask
        )

        return self.output_proj(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        memory = self.encode(src, src_mask)

        return self.decode(
            memory,
            src_mask,
            tgt,
            tgt_mask
        )

    # ================================================================
    # INFERENCE
    # ================================================================

    def infer(self, src_sentence: str) -> str:

        self.eval()

        device = next(self.parameters()).device

        # Tokenize German sentence
        tokens = [
            tok.text.lower()
            for tok in self.de_nlp.tokenizer(src_sentence)
        ]

        # Convert to indices
        src_indices = (
            [self.sos_idx]
            + [
                self.src_stoi.get(tok, 0)
                for tok in tokens
            ]
            + [self.eos_idx]
        )

        src = torch.tensor(
            src_indices,
            dtype=torch.long
        ).unsqueeze(0).to(device)

        src_mask = make_src_mask(src, self.pad_idx)

        ys = torch.tensor(
            [[self.sos_idx]],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            memory = self.encode(src, src_mask)

            for _ in range(100):

                tgt_mask = make_tgt_mask(
                    ys,
                    self.pad_idx
                ).to(device)

                logits = self.decode(
                    memory,
                    src_mask,
                    ys,
                    tgt_mask
                )

                next_tok = logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True
                )

                ys = torch.cat([ys, next_tok], dim=1)

                if next_tok.item() == self.eos_idx:
                    break

        result = []

        for idx in ys.squeeze(0).tolist()[1:]:

            if idx == self.eos_idx:
                break

            if idx != self.pad_idx:
                result.append(
                    self.tgt_itos.get(idx, "<unk>")
                )

        return " ".join(result)
