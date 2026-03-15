"""
Dataset and dataloader utilities for BERTweet / Transformer fine-tuning.

Uses a HuggingFace tokenizer instead of the custom Vocabulary used by BiLSTM,
while reading the same processed CSV format (text + label_id columns).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TransformerSentimentDataset(Dataset):
    """Dataset for transformer-based sentiment classifiers.

    Args:
        texts: list of cleaned text strings.
        labels: list of integer label ids.
        tokenizer: HuggingFace tokenizer instance.
        max_len: maximum token length; sequences are truncated / padded to this.
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        max_len: int = 128,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def build_transformer_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    model_name: str,
    max_len: int = 128,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for a transformer model.

    The tokenizer is downloaded once and shared across all three splits.

    Args:
        train_path: path to processed train CSV (must have 'text' and 'label_id').
        val_path: path to processed val CSV.
        test_path: path to processed test CSV.
        model_name: HuggingFace model identifier used to load the tokenizer.
        max_len: maximum token sequence length.
        batch_size: samples per batch.
        num_workers: DataLoader worker processes (set 0 on Windows or for debugging).
        seed: random seed used for shuffling the training set.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    logger.info("Tokenizer loaded: %s", model_name)

    def _build_loader(path: str, shuffle: bool) -> DataLoader:
        df = pd.read_csv(path)
        texts = df["text"].astype(str).tolist()
        labels = df["label_id"].astype(int).tolist()
        dataset = TransformerSentimentDataset(texts, labels, tokenizer, max_len)
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
        )

    train_loader = _build_loader(train_path, shuffle=True)
    val_loader = _build_loader(val_path, shuffle=False)
    test_loader = _build_loader(test_path, shuffle=False)

    logger.info(
        "DataLoaders built: train=%d, val=%d, test=%d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader
