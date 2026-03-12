"""
Dataset classes and data-loading utilities for Sentimind.
Provides both a vocabulary builder and PyTorch Dataset wrappers.
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Simple word-level vocabulary built from training data."""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self._counter: Counter = Counter()
        self._built = False

    # ------------------------------------------------------------------
    def fit(self, texts: List[str]) -> "Vocabulary":
        """Build vocabulary from a list of cleaned texts (space-tokenised)."""
        for text in texts:
            self._counter.update(text.split())
        for word, freq in self._counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        self._built = True
        logger.info("Vocabulary built: %d tokens (min_freq=%d).", len(self.word2idx), self.min_freq)
        return self

    def encode(self, text: str, max_len: int) -> List[int]:
        """Encode text to padded/truncated list of token ids."""
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, self.word2idx[UNK_TOKEN]) for t in tokens]
        # Pad
        ids += [self.word2idx[PAD_TOKEN]] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.word2idx)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "min_freq": self.min_freq}, f, ensure_ascii=False, indent=2)
        logger.info("Vocabulary saved to %s (%d tokens).", path, len(self.word2idx))

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        vocab = cls(min_freq=obj.get("min_freq", 2))
        vocab.word2idx = obj["word2idx"]
        vocab.idx2word = {int(i): w for w, i in vocab.word2idx.items()}
        vocab._built = True
        logger.info("Vocabulary loaded from %s (%d tokens).", path, len(vocab.word2idx))
        return vocab


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """PyTorch Dataset for mental-health sentiment classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        max_len: int = 128,
        text_col: str = "text",
        label_col: str = "label_id",
    ):
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Data loaders factory
# ---------------------------------------------------------------------------

def build_vocab_and_loaders(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    vocab_path: str | Path,
    max_len: int = 128,
    vocab_min_freq: int = 2,
    batch_size: int = 64,
    text_col: str = "text",
    label_col: str = "label_id",
    seed: int = 42,
) -> Tuple[Vocabulary, DataLoader, DataLoader, DataLoader]:
    """Build vocabulary from train set and return dataloaders for all splits."""
    # Fixed seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    vocab_p = Path(vocab_path)
    if vocab_p.exists():
        logger.info("Loading existing vocabulary from %s.", vocab_path)
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = Vocabulary(min_freq=vocab_min_freq)
        vocab.fit(train_df[text_col].tolist())
        vocab.save(vocab_path)

    train_ds = SentimentDataset(train_df, vocab, max_len, text_col, label_col)
    val_ds = SentimentDataset(val_df, vocab, max_len, text_col, label_col)
    test_ds = SentimentDataset(test_df, vocab, max_len, text_col, label_col)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Dataloaders ready — train: %d | val: %d | test: %d",
                len(train_ds), len(val_ds), len(test_ds))
    return vocab, train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Class weight helper (for weighted cross-entropy loss)
# ---------------------------------------------------------------------------

def compute_class_weights(train_path: str | Path, num_classes: int,
                          label_col: str = "label_id") -> torch.Tensor:
    """Inverse-frequency class weights from training labels."""
    df = pd.read_csv(train_path)
    counts = np.zeros(num_classes)
    for label_id, cnt in df[label_col].value_counts().items():
        counts[int(label_id)] = cnt
    # Replace zeros to avoid division by zero
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)
