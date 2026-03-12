"""
BiLSTM classifier for mental-health sentiment analysis.
Architecture: Embedding → BiLSTM → Dropout → Dense → Softmax
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM text classifier.

    Args:
        vocab_size: vocabulary size (including PAD and UNK).
        embedding_dim: word embedding dimension.
        hidden_dim: LSTM hidden state dimension (per direction).
        num_classes: number of output classes.
        num_layers: number of stacked LSTM layers.
        dropout: dropout probability applied between layers and on embeddings.
        pad_idx: index of the <PAD> token (used for embedding padding).
        bidirectional: if True, use bidirectional LSTM.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LongTensor of shape (batch, seq_len).

        Returns:
            logits: FloatTensor of shape (batch, num_classes).
        """
        # Embeddings  (batch, seq_len, emb_dim)
        emb = self.embedding_dropout(self.embedding(x))

        # LSTM  →  (batch, seq_len, hidden*dirs)
        lstm_out, _ = self.lstm(emb)

        # Use the last non-padding hidden state (mean pooling is more robust)
        # Simple approach: mean-pool over the sequence dimension
        pooled = lstm_out.mean(dim=1)   # (batch, hidden*dirs)

        logits = self.classifier(pooled)
        return logits

    # ------------------------------------------------------------------
    # Pretrained embedding loader
    # ------------------------------------------------------------------

    def load_pretrained_embeddings(
        self,
        glove_path: str | Path,
        word2idx: dict,
        freeze: bool = False,
    ) -> None:
        """Load GloVe / word2vec text-format embeddings into the embedding layer.

        Unknown tokens keep their random initialisation (no zeroing — preserves
        gradient signal for OOV words during training).
        """
        glove_path = Path(glove_path)
        logger.info("Loading pretrained embeddings from %s …", glove_path)
        emb_dim = self.embedding.embedding_dim
        embeddings = self.embedding.weight.data.clone()   # start from current init

        loaded, total = 0, 0
        with open(glove_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip().split(" ")
                word = parts[0]
                total += 1
                if word in word2idx:
                    vec = np.array(parts[1:], dtype=np.float32)
                    if len(vec) == emb_dim:
                        embeddings[word2idx[word]] = torch.from_numpy(vec)
                        loaded += 1

        self.embedding.weight = nn.Parameter(embeddings, requires_grad=not freeze)
        logger.info("Loaded %d / %d GloVe vectors (vocab coverage: %.1f%%).",
                    loaded, len(word2idx), 100.0 * loaded / max(len(word2idx), 1))
