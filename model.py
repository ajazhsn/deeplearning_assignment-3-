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
from collections import Counter
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
        self.attn_weights: Optional[torch.Tensor] = None

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
        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # 3. Scaled dot-product attention
        attn_out, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        self.attn_weights = attn_w  # store for visualization

        # 4. Apply dropout to attention weights, then weight V
        attn_out = torch.matmul(self.dropout(attn_w), V)

        # 5. Concatenate heads: [batch, seq_q, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 6. Final linear projection
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

    def __init__(self, layer: EncoderLayer, N: int, d_model: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int, d_model: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(d_model)

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
#  VOCAB BUILDER  (standalone so it can be tested independently)
# ══════════════════════════════════════════════════════════════════════

_SPECIAL = ['<unk>', '<pad>', '<sos>', '<eos>']
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


def build_vocab(tokens: list, min_freq: int = 2) -> Tuple[dict, dict]:
    """Return (stoi, itos) dicts from a flat token list."""
    counter = Counter(tokens)
    stoi = {tok: i for i, tok in enumerate(_SPECIAL)}
    itos = {i: tok for i, tok in enumerate(_SPECIAL)}
    idx = len(_SPECIAL)
    for tok, cnt in sorted(counter.items()):
        if cnt >= min_freq and tok not in stoi:
            stoi[tok] = idx
            itos[idx] = tok
            idx += 1
    return stoi, itos


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
    Vocabulary and tokenizers are built lazily via `build_from_dataset`,
    keeping __init__ fast and unit-testable with explicit vocab sizes.
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
        pad_idx:   int   = PAD_IDX,
    ) -> None:
        super().__init__()

        # Auto-build vocab/model when called as Transformer()
        if src_vocab_size is None or tgt_vocab_size is None:
            tmp_model = Transformer.build_from_dataset(
                d_model=d_model,
                N=N,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                checkpoint_path="checkpoint.pt",
                gdrive_file_id="1R-nnKC_69Vxg-TqTlOMMhirFzbC9oORr",
            )
            self.__dict__.update(tmp_model.__dict__)
            return

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX

        # Tokenizer / vocab attributes (populated by build_from_dataset)
        self.de_nlp    = None
        self.en_nlp    = None
        self.src_stoi: dict = {}
        self.src_itos: dict = {}
        self.tgt_stoi: dict = {}
        self.tgt_itos: dict = {}

        # ── Embeddings ──────────────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, dropout)

        # ── Encoder / Decoder stacks ─────────────────────────────────
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N, d_model)
        self.decoder = Decoder(dec_layer, N, d_model)

        # ── Output projection ────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────
    #  CLASS-LEVEL FACTORY — load dataset, build vocab, construct model
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def build_from_dataset(
        cls,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        checkpoint_path: Optional[str] = "checkpoint.pt",
        gdrive_file_id:  Optional[str] = "1R-nnKC_69Vxg-TqTlOMMhirFzbC9oORr",
    ) -> "Transformer":
        """
        Convenience factory that:
          1. Loads spaCy tokenizers
          2. Streams the Multi30k training split
          3. Builds source / target vocabularies
          4. Constructs and (optionally) loads a checkpoint
        """
        import spacy
        from datasets import load_dataset

        # ── Tokenizers ───────────────────────────────────────────────
        try:
            de_nlp = spacy.load("de_core_news_sm")
        except OSError:
            de_nlp = spacy.blank("de")

        try:
            en_nlp = spacy.load("en_core_web_sm")
        except OSError:
            en_nlp = spacy.blank("en")

        def tokenize_de(text: str):
            return [tok.text.lower() for tok in de_nlp.tokenizer(text)]

        def tokenize_en(text: str):
            return [tok.text.lower() for tok in en_nlp.tokenizer(text)]

        # ── Vocabulary ───────────────────────────────────────────────
        dataset = load_dataset("bentrevett/multi30k", split="train")

        de_tokens = [tok for ex in dataset for tok in tokenize_de(ex["de"])]
        en_tokens = [tok for ex in dataset for tok in tokenize_en(ex["en"])]

        src_stoi, src_itos = build_vocab(de_tokens)
        tgt_stoi, tgt_itos = build_vocab(en_tokens)

        # ── Construct model ──────────────────────────────────────────
        model = cls(
            src_vocab_size=len(src_stoi),
            tgt_vocab_size=len(tgt_stoi),
            d_model=d_model,
            N=N,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        model.de_nlp   = de_nlp
        model.en_nlp   = en_nlp
        model.src_stoi = src_stoi
        model.src_itos = src_itos
        model.tgt_stoi = tgt_stoi
        model.tgt_itos = tgt_itos

        # ── Optional checkpoint ───────────────────────────────────────
        if checkpoint_path and gdrive_file_id:
            try:
                import gdown
                if not os.path.exists(checkpoint_path):
                    print("Downloading checkpoint…")
                    gdown.download(id=gdrive_file_id, output=checkpoint_path, quiet=False)
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(ckpt["model_state_dict"])
                print("Weights loaded successfully.")
            except Exception as e:
                print(f"Warning: could not load checkpoint: {e}")

        return model

    # ──────────────────────────────────────────────────────────────────
    #  WEIGHT INITIALISATION
    # ──────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ──────────────────────────────────────────────────────────────────
    #  AUTOGRADER HOOKS
    # ──────────────────────────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Embed → positional encode → encoder stack."""
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed → positional encode → decoder stack → output projection."""
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_proj(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ──────────────────────────────────────────────────────────────────
    #  GREEDY INFERENCE
    # ──────────────────────────────────────────────────────────────────

    def infer(self, src_sentence: str, max_len: int = 150) -> str:
        """
        Greedy-decode a German source sentence to English.
        Requires the model to have been built via `build_from_dataset`
        so that tokenizers and vocab dicts are populated.
        """
        if self.de_nlp is None or not self.src_stoi:
            raise RuntimeError(
                "Call Transformer.build_from_dataset() to populate "
                "tokenizers and vocabularies before running inference."
            )

        self.eval()
        device = next(self.parameters()).device

        tokens = [tok.text.lower().strip() for tok in self.de_nlp.tokenizer(src_sentence)]
        src_indices = (
            [self.sos_idx]
            + [self.src_stoi.get(t, UNK_IDX) for t in tokens]
            + [self.eos_idx]
        )

        src = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src, self.pad_idx)

        ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)

        with torch.no_grad():
            memory = self.encode(src, src_mask)

            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, self.pad_idx).to(device)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys       = torch.cat([ys, next_tok], dim=1)

                if next_tok.item() == self.eos_idx:
                    break

        result = []
        for idx in ys.squeeze(0).tolist()[1:]:
            if idx == self.eos_idx:
                break
            if idx != self.pad_idx:
                result.append(self.tgt_itos.get(idx, "<unk>"))

        return " ".join(result)
