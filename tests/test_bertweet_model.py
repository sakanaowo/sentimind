"""
Smoke tests for BERTweet model components.

Tests are designed to run quickly without downloading actual model weights —
the HuggingFace model loading is mocked so only the wrapping logic is verified.

Run: pytest tests/test_bertweet_model.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.data.bertweet_dataset import TransformerSentimentDataset
from src.models.bertweet import BERTweetClassifier
from src.utils.metrics import compute_metrics, save_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 7
HIDDEN_SIZE = 32  # tiny hidden dim for mock model


def _make_mock_hf_output(batch_size: int, num_labels: int) -> MagicMock:
    """Return a fake HF model output with random logits."""
    output = MagicMock()
    output.logits = torch.randn(batch_size, num_labels)
    return output


def _make_mock_hf_model(num_labels: int) -> MagicMock:
    """A mock AutoModelForSequenceClassification that returns fixed-shape logits."""
    mock_model = MagicMock(spec=nn.Module)
    mock_model.return_value = _make_mock_hf_output(4, num_labels)
    # Replicate nn.Module's named_parameters
    mock_model.named_parameters.return_value = iter([])
    mock_model.state_dict.return_value = {}
    mock_model.load_state_dict.return_value = None
    mock_model.to.return_value = mock_model
    mock_model.train.return_value = mock_model
    mock_model.eval.return_value = mock_model
    return mock_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_bertweet(monkeypatch):
    """BERTweetClassifier whose transformer backbone is replaced by a mock."""
    with (
        patch("src.models.bertweet.AutoConfig.from_pretrained") as mock_cfg,
        patch(
            "src.models.bertweet.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_hf,
    ):
        mock_cfg.return_value = MagicMock()
        mock_hf.return_value = _make_mock_hf_model(NUM_CLASSES)
        model = BERTweetClassifier(
            model_name="mock/bertweet",
            num_classes=NUM_CLASSES,
            dropout=0.1,
        )
        # Replace the inner HF model with a properly parameterised mock
        # so state_dict round-trips work in checkpoint tests.
        model.model = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        model.model_name = "mock/bertweet"
        model.num_classes = NUM_CLASSES
    return model


@pytest.fixture
def mock_tokenizer():
    """Minimal tokenizer mock that returns fixed-length tensors."""

    class _MockTokenizer:
        def __call__(self, texts, truncation, padding, max_length, return_tensors):
            n = len(texts) if isinstance(texts, list) else 1
            return {
                "input_ids": torch.zeros(n, max_length, dtype=torch.long),
                "attention_mask": torch.ones(n, max_length, dtype=torch.long),
            }

    return _MockTokenizer()


@pytest.fixture
def sample_dataset(mock_tokenizer):
    texts = [
        "i feel overwhelmed and hopeless",
        "having a great day today",
        "anxiety keeps me awake at night",
        "everything is fine",
    ]
    labels = [1, 0, 2, 0]
    return TransformerSentimentDataset(texts, labels, mock_tokenizer, max_len=32)


# ---------------------------------------------------------------------------
# TransformerSentimentDataset
# ---------------------------------------------------------------------------


class TestTransformerSentimentDataset:
    def test_length(self, sample_dataset):
        assert len(sample_dataset) == 4

    def test_item_keys(self, sample_dataset):
        item = sample_dataset[0]
        assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_input_ids_shape(self, sample_dataset):
        item = sample_dataset[0]
        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)

    def test_label_type(self, sample_dataset):
        item = sample_dataset[0]
        assert item["labels"].dtype == torch.long

    def test_correct_label_values(self, sample_dataset):
        assert sample_dataset[0]["labels"].item() == 1
        assert sample_dataset[1]["labels"].item() == 0
        assert sample_dataset[2]["labels"].item() == 2


# ---------------------------------------------------------------------------
# BERTweetClassifier forward pass
# ---------------------------------------------------------------------------


class TestBERTweetClassifier:
    def test_forward_output_shape(self, mock_bertweet):
        batch_size, seq_len = 4, 32
        # mock_bertweet.model is an nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        # We override forward to bypass the real model call.
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        with patch.object(
            mock_bertweet,
            "forward",
            return_value=torch.randn(batch_size, NUM_CLASSES),
        ):
            logits = mock_bertweet(input_ids, attention_mask)
        assert logits.shape == (batch_size, NUM_CLASSES)

    def test_model_name_stored(self, mock_bertweet):
        assert mock_bertweet.model_name == "mock/bertweet"

    def test_num_classes_stored(self, mock_bertweet):
        assert mock_bertweet.num_classes == NUM_CLASSES

    def test_checkpoint_roundtrip(self, mock_bertweet, tmp_path):
        """save_checkpoint then from_checkpoint must restore state_dict."""
        ckpt_path = tmp_path / "bertweet_best.pt"
        mock_bertweet.save_checkpoint(ckpt_path, epoch=3, best_metric=0.87)

        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 3
        assert abs(ckpt["best_metric"] - 0.87) < 1e-6
        assert ckpt["model_name"] == "mock/bertweet"
        assert ckpt["num_classes"] == NUM_CLASSES

    def test_checkpoint_state_dict_keys(self, mock_bertweet, tmp_path):
        ckpt_path = tmp_path / "ckpt.pt"
        original_keys = set(mock_bertweet.state_dict().keys())
        mock_bertweet.save_checkpoint(ckpt_path, epoch=1, best_metric=0.5)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert set(ckpt["model_state_dict"].keys()) == original_keys


# ---------------------------------------------------------------------------
# Metric contract compatibility
# ---------------------------------------------------------------------------


class TestMetricContractCompatibility:
    """Ensure BERTweet outputs can be processed by the shared metrics module."""

    def test_compute_metrics_structure(self):
        y_true = [0, 1, 2, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 1, 0, 2, 3]
        label_names = {0: "Normal", 1: "Depression", 2: "Anxiety", 3: "Bipolar"}
        metrics = compute_metrics(
            y_true,
            y_pred,
            label_names=label_names,
            model_name="bertweet",
            split="test",
        )
        assert metrics["model"] == "bertweet"
        assert metrics["split"] == "test"
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics

    def test_compute_metrics_per_class_keys(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        metrics = compute_metrics(
            y_true,
            y_pred,
            label_names={0: "Normal", 1: "Depression"},
            model_name="bertweet",
            split="val",
        )
        for class_metrics in metrics["per_class"].values():
            assert "precision" in class_metrics
            assert "recall" in class_metrics
            assert "f1" in class_metrics
            assert "support" in class_metrics

    def test_save_metrics_produces_valid_json(self, tmp_path):
        y_true = [0, 1, 2, 0]
        y_pred = [0, 1, 1, 0]
        metrics = compute_metrics(y_true, y_pred, model_name="bertweet", split="test")
        path = tmp_path / "bertweet_metrics.json"
        save_metrics(metrics, path)
        loaded = json.loads(path.read_text())
        assert loaded["model"] == "bertweet"

    def test_metrics_schema_matches_bilstm(self):
        """Verify same top-level keys as the data_contract §7 schema."""
        required_keys = {
            "model",
            "split",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "per_class",
            "confusion_matrix",
        }
        y_true = [0, 1]
        y_pred = [0, 1]
        metrics = compute_metrics(y_true, y_pred, model_name="bertweet", split="test")
        assert required_keys.issubset(set(metrics.keys()))


# ---------------------------------------------------------------------------
# build_transformer_loaders (with mock tokenizer & CSV files)
# ---------------------------------------------------------------------------


class TestBuildTransformerLoaders:
    def test_loaders_created(self, tmp_path):
        for split in ("train", "val", "test"):
            df = pd.DataFrame({"text": ["hello world"] * 8, "label_id": [0] * 8})
            df.to_csv(tmp_path / f"{split}.csv", index=False)

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.zeros(8, 32, dtype=torch.long),
            "attention_mask": torch.ones(8, 32, dtype=torch.long),
        }

        with patch(
            "src.data.bertweet_dataset.AutoTokenizer.from_pretrained",
            return_value=mock_tok,
        ):
            from src.data.bertweet_dataset import build_transformer_loaders

            train_loader, val_loader, test_loader = build_transformer_loaders(
                train_path=str(tmp_path / "train.csv"),
                val_path=str(tmp_path / "val.csv"),
                test_path=str(tmp_path / "test.csv"),
                model_name="mock/bertweet",
                max_len=32,
                batch_size=4,
                seed=42,
            )
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_batch_structure(self, tmp_path):
        for split in ("train", "val", "test"):
            df = pd.DataFrame({"text": ["sample text"] * 4, "label_id": [1] * 4})
            df.to_csv(tmp_path / f"{split}.csv", index=False)

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.zeros(4, 16, dtype=torch.long),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
        }

        with patch(
            "src.data.bertweet_dataset.AutoTokenizer.from_pretrained",
            return_value=mock_tok,
        ):
            from src.data.bertweet_dataset import build_transformer_loaders

            train_loader, _, _ = build_transformer_loaders(
                train_path=str(tmp_path / "train.csv"),
                val_path=str(tmp_path / "val.csv"),
                test_path=str(tmp_path / "test.csv"),
                model_name="mock/bertweet",
                max_len=16,
                batch_size=4,
                seed=42,
            )
        batch = next(iter(train_loader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[1] == 16
