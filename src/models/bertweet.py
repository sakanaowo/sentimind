"""
BERTweet / Twitter-RoBERTa classifier.

Wraps HuggingFace AutoModelForSequenceClassification so the rest of the
pipeline (trainer, metrics, eval script) can treat it like any nn.Module that
accepts (input_ids, attention_mask) and returns logits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class BERTweetClassifier(nn.Module):
    """Sequence classifier built on top of a Twitter-adapted Transformer.

    Args:
        model_name: HuggingFace model identifier (default: ``vinai/bertweet-base``).
        num_classes: number of output classes.
        dropout: dropout probability applied to hidden states and attention.
        freeze_base: if True, freeze all transformer layers and only train the
            classification head. Useful for a quick sanity-check run.
    """

    def __init__(
        self,
        model_name: str = "vinai/bertweet-base",
        num_classes: int = 7,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,  # classifier head is always fresh
        )
        self.model_name = model_name
        self.num_classes = num_classes

        if freeze_base:
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "pooler" not in name:
                    param.requires_grad_(False)
            logger.info("Base encoder frozen; only classifier head will be trained.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: LongTensor of shape (batch, seq_len).
            attention_mask: BoolTensor of shape (batch, seq_len).

        Returns:
            logits: FloatTensor of shape (batch, num_classes).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        best_metric: float,
    ) -> None:
        """Save model weights and metadata to a single .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "epoch": epoch,
                "best_metric": best_metric,
            },
            path,
        )
        logger.info(
            "Checkpoint saved to %s (epoch %d, metric=%.4f).", path, epoch, best_metric
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "BERTweetClassifier":
        """Load a BERTweetClassifier from a checkpoint saved by :meth:`save_checkpoint`.

        Args:
            checkpoint_path: path to the .pt file.
            device: target device; defaults to CPU.

        Returns:
            Loaded BERTweetClassifier in eval mode.
        """
        device = device or torch.device("cpu")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = cls(
            model_name=ckpt["model_name"],
            num_classes=ckpt["num_classes"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        logger.info(
            "Loaded checkpoint from %s (epoch %d, best_metric=%.4f).",
            checkpoint_path,
            ckpt.get("epoch", -1),
            ckpt.get("best_metric", float("nan")),
        )
        return model
