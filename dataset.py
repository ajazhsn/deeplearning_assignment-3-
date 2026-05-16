"""
dataset.py — Multi30k Dataset Loading and Preprocessing
DA6401 Assignment 3: "Attention Is All You Need"
"""

from datasets import load_dataset
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import Dict, List, Tuple, Optional


# Special tokens
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_IDX   = 1
UNK_IDX   = 0
SOS_IDX   = 2
EOS_IDX   = 3

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]


class Vocabulary:
    """Simple vocab with stoi (string→int) and itos (int→string)."""

    def __init__(self, tokens: List[str], min_freq: int = 2):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._build(tokens, min_freq)

    def _build(self, tokens: List[str], min_freq: int):
        # Special tokens first
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.stoi[tok] = i
            self.itos[i] = tok

        # Count and add tokens above min_freq
        counter = Counter(tokens)
        idx = len(SPECIAL_TOKENS)
        for tok, cnt in sorted(counter.items()):
            if cnt >= min_freq and tok not in self.stoi:
                self.stoi[tok] = idx
                self.itos[idx] = tok
                idx += 1

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def get(self, token: str, default: int = UNK_IDX) -> int:
        return self.stoi.get(token, default)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, UNK_TOKEN)


class Multi30kDataset(Dataset):
    """
    Multi30k dataset for German→English translation.
    Downloads from HuggingFace, tokenizes with spaCy, builds vocabularies.
    """

    def __init__(
        self,
        split: str = 'train',
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        min_freq: int = 2,
        max_len: int = 256,
    ):
        self.split   = split
        self.max_len = max_len

        # Load spacy tokenizers
        print("Loading spaCy tokenizers...")
        try:
            self.de_nlp = spacy.load('de_core_news_sm')
        except OSError:
            raise OSError(
                "German spaCy model not found. Run:\n"
                "  python -m spacy download de_core_news_sm"
            )
        try:
            self.en_nlp = spacy.load('en_core_web_sm')
        except OSError:
            raise OSError(
                "English spaCy model not found. Run:\n"
                "  python -m spacy download en_core_web_sm"
            )

        # Load HuggingFace dataset
        print(f"Loading Multi30k {split} split...")
        dataset = load_dataset('bentrevett/multi30k', split=split)
        self.raw_de = [ex['de'] for ex in dataset]
        self.raw_en = [ex['en'] for ex in dataset]

        # Tokenize all sentences
        print("Tokenizing...")
        self.tok_de = [self._tokenize_de(s) for s in self.raw_de]
        self.tok_en = [self._tokenize_en(s) for s in self.raw_en]

        # Build or reuse vocab (only build from training split)
        if src_vocab is None or tgt_vocab is None:
            print("Building vocabularies...")
            all_de_tokens = [tok for sent in self.tok_de for tok in sent]
            all_en_tokens = [tok for sent in self.tok_en for tok in sent]
            self.src_vocab = Vocabulary(all_de_tokens, min_freq)
            self.tgt_vocab = Vocabulary(all_en_tokens, min_freq)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # Numericalize
        print("Numericalizing...")
        self.src_data = [self._numericalize(toks, self.src_vocab) for toks in self.tok_de]
        self.tgt_data = [self._numericalize(toks, self.tgt_vocab) for toks in self.tok_en]

        print(f"  {split}: {len(self.src_data)} sentence pairs")
        print(f"  src vocab size: {len(self.src_vocab)}, tgt vocab size: {len(self.tgt_vocab)}")

    def _tokenize_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.de_nlp.tokenizer(text)]

    def _tokenize_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.en_nlp.tokenizer(text)]

    def _numericalize(self, tokens: List[str], vocab: Vocabulary) -> List[int]:
        return (
            [SOS_IDX]
            + [vocab[tok] for tok in tokens[:self.max_len]]
            + [EOS_IDX]
        )

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.src_data[idx], dtype=torch.long),
            torch.tensor(self.tgt_data[idx], dtype=torch.long),
        )

    def build_vocab(self):
        """Return (src_vocab, tgt_vocab) — for compatibility."""
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """Return list of (src_indices, tgt_indices) tuples."""
        return list(zip(self.src_data, self.tgt_data))


def collate_fn(batch, pad_idx: int = PAD_IDX):
    """Pad sequences in a batch to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=pad_idx
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=pad_idx
    )
    return src_padded, tgt_padded


def get_dataloaders(
    batch_size: int = 128,
    min_freq:   int = 2,
    max_len:    int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Build train / val / test DataLoaders and vocabularies.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    from functools import partial

    # Build vocab from train only
    train_ds = Multi30kDataset('train', min_freq=min_freq, max_len=max_len)
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    # val / test reuse train vocab
    val_ds  = Multi30kDataset('validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab, max_len=max_len)
    test_ds = Multi30kDataset('test',       src_vocab=src_vocab, tgt_vocab=tgt_vocab, max_len=max_len)

    _collate = partial(collate_fn, pad_idx=PAD_IDX)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=1,          shuffle=False,
                              collate_fn=_collate, num_workers=num_workers)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
