"""
Smoke tests for BiLSTM model and training components.
Run: pytest tests/test_bilstm_model.py -v
"""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.models.bilstm import BiLSTMClassifier
from src.data.dataset import Vocabulary, SentimentDataset, compute_class_weights
from src.training.trainer import EarlyStopping, set_seed
from src.utils.metrics import compute_metrics, save_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_vocab():
    texts = [
        "i feel very hopeless today",
        "my anxiety is really bad",
        "feeling great and productive",
        "i cannot sleep anymore",
    ]
    vocab = Vocabulary(min_freq=1)
    vocab.fit(texts)
    return vocab


@pytest.fixture
def small_model(small_vocab):
    return BiLSTMClassifier(
        vocab_size=len(small_vocab),
        embedding_dim=32,
        hidden_dim=16,
        num_classes=3,
        num_layers=1,
        dropout=0.0,
        pad_idx=0,
        bidirectional=True,
    )


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_fit_creates_tokens(self, small_vocab):
        assert len(small_vocab) > 2   # at least PAD + UNK + real words

    def test_pad_and_unk_present(self, small_vocab):
        assert "<PAD>" in small_vocab.word2idx
        assert "<UNK>" in small_vocab.word2idx
        assert small_vocab.word2idx["<PAD>"] == 0
        assert small_vocab.word2idx["<UNK>"] == 1

    def test_encode_length(self, small_vocab):
        ids = small_vocab.encode("i feel hopeless", max_len=10)
        assert len(ids) == 10

    def test_encode_truncates(self, small_vocab):
        long_text = " ".join(["word"] * 50)
        ids = small_vocab.encode(long_text, max_len=5)
        assert len(ids) == 5

    def test_encode_pads(self, small_vocab):
        ids = small_vocab.encode("hi", max_len=8)
        assert ids[-1] == small_vocab.word2idx["<PAD>"]

    def test_oov_maps_to_unk(self, small_vocab):
        ids = small_vocab.encode("xyznonexistentword", max_len=5)
        assert ids[0] == small_vocab.word2idx["<UNK>"]

    def test_save_and_load(self, small_vocab, tmp_path):
        path = tmp_path / "vocab.json"
        small_vocab.save(path)
        loaded = Vocabulary.load(path)
        assert len(loaded) == len(small_vocab)
        assert loaded.word2idx == small_vocab.word2idx


# ---------------------------------------------------------------------------
# BiLSTM model
# ---------------------------------------------------------------------------

class TestBiLSTMClassifier:
    def test_forward_shape(self, small_model):
        batch_size, seq_len = 4, 20
        x = torch.randint(0, len(small_model.embedding.weight), (batch_size, seq_len))
        logits = small_model(x)
        assert logits.shape == (batch_size, 3)

    def test_output_is_finite(self, small_model):
        x = torch.randint(0, 10, (2, 15))
        logits = small_model(x)
        assert torch.isfinite(logits).all()

    def test_pad_idx_embedding_zero(self, small_model):
        pad_vec = small_model.embedding.weight[0].detach()
        assert (pad_vec == 0).all()

    def test_parameters_trainable(self, small_model):
        n_trainable = sum(p.numel() for p in small_model.parameters() if p.requires_grad)
        assert n_trainable > 0

    def test_different_batch_sizes(self, small_model):
        for bs in [1, 8, 16]:
            x = torch.randint(0, 10, (bs, 12))
            out = small_model(x)
            assert out.shape[0] == bs


# ---------------------------------------------------------------------------
# SentimentDataset
# ---------------------------------------------------------------------------

class TestSentimentDataset:
    def test_len(self, small_vocab):
        df = pd.DataFrame({"text": ["feel bad", "feel good"], "label_id": [1, 0]})
        ds = SentimentDataset(df, small_vocab, max_len=10)
        assert len(ds) == 2

    def test_item_shapes(self, small_vocab):
        df = pd.DataFrame({"text": ["feel anxious"], "label_id": [2]})
        ds = SentimentDataset(df, small_vocab, max_len=8)
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.ndim == 0


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, mode="max")
        assert not es(0.5, 1)
        assert not es(0.4, 2)
        assert not es(0.4, 3)
        assert es(0.4, 4)   # 3 non-improvements → stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=2, mode="max")
        es(0.5, 1)
        es(0.4, 2)   # counter = 1
        es(0.9, 3)   # improvement → counter = 0
        assert not es(0.8, 4)   # counter = 1, not yet patience

    def test_min_mode(self):
        es = EarlyStopping(patience=2, mode="min")
        assert not es(1.0, 1)
        assert not es(1.5, 2)   # worse, counter = 1
        assert es(1.6, 3)       # worse, counter = 2 → stop


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self):
        m = compute_metrics([0, 1, 2, 0, 1], [0, 1, 2, 0, 1])
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_metric_contract_keys(self):
        m = compute_metrics([0, 1, 2], [0, 1, 1])
        required_keys = {"model", "split", "accuracy", "macro_f1", "weighted_f1",
                         "per_class", "confusion_matrix"}
        assert required_keys.issubset(set(m.keys()))

    def test_per_class_keys(self):
        m = compute_metrics([0, 1], [0, 1])
        for label_metrics in m["per_class"].values():
            assert {"precision", "recall", "f1", "support"} == set(label_metrics.keys())

    def test_confusion_matrix_shape(self):
        y = [0, 1, 2, 0, 1, 2]
        m = compute_metrics(y, y)
        assert len(m["confusion_matrix"]) == 3
        assert all(len(row) == 3 for row in m["confusion_matrix"])

    def test_save_metrics(self, tmp_path):
        m = compute_metrics([0, 1], [0, 1], model_name="bilstm", split="test")
        path = tmp_path / "metrics.json"
        save_metrics(m, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["model"] == "bilstm"
        assert loaded["accuracy"] == 1.0
