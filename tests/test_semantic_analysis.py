"""
Unit tests for the semantic analysis pipeline (no real embeddings — all mocked).

Run: pytest tests/test_semantic_analysis.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import the functions we want to test directly from the script.
# The script uses top-level imports only inside functions, so importing it
# at module level is safe even when optional deps (umap, hdbscan) are absent.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_semantic_analysis import (
    cosine_similarity,
    run_comparison,
    run_sts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_embeddings():
    """4 normalised 8-D embeddings (2 per class)."""
    rng = np.random.RandomState(42)
    emb = rng.randn(8, 8)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / norms).astype(np.float32)


@pytest.fixture
def small_labels():
    return [0, 0, 1, 1, 2, 2, 0, 1]


@pytest.fixture
def label_map():
    return {0: "Normal", 1: "Depression", 2: "Anxiety"}


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_returns_float(self):
        a = np.array([0.6, 0.8])
        b = np.array([0.8, 0.6])
        result = cosine_similarity(a, b)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# run_sts
# ---------------------------------------------------------------------------


class TestRunSts:
    def test_returns_dict(self, small_embeddings, small_labels, label_map):
        result = run_sts(
            small_embeddings, small_labels, label_map, pairs_per_class=5, seed=0
        )
        assert isinstance(result, dict)

    def test_has_within_class_key(self, small_embeddings, small_labels, label_map):
        result = run_sts(
            small_embeddings, small_labels, label_map, pairs_per_class=5, seed=0
        )
        assert "within_class_avg_cosine" in result

    def test_has_cross_class_key(self, small_embeddings, small_labels, label_map):
        result = run_sts(
            small_embeddings, small_labels, label_map, pairs_per_class=5, seed=0
        )
        assert "cross_class_avg_cosine" in result

    def test_within_class_scores_are_floats(
        self, small_embeddings, small_labels, label_map
    ):
        result = run_sts(
            small_embeddings, small_labels, label_map, pairs_per_class=5, seed=0
        )
        for v in result["within_class_avg_cosine"].values():
            assert isinstance(v, float)

    def test_within_class_keys_match_labels(
        self, small_embeddings, small_labels, label_map
    ):
        result = run_sts(
            small_embeddings, small_labels, label_map, pairs_per_class=5, seed=0
        )
        present_names = set(result["within_class_avg_cosine"].keys())
        expected_names = {label_map[lbl] for lbl in set(small_labels)}
        assert present_names == expected_names

    def test_deterministic_with_same_seed(
        self, small_embeddings, small_labels, label_map
    ):
        r1 = run_sts(
            small_embeddings, small_labels, label_map, pairs_per_class=10, seed=7
        )
        r2 = run_sts(
            small_embeddings, small_labels, label_map, pairs_per_class=10, seed=7
        )
        assert r1["within_class_avg_cosine"] == r2["within_class_avg_cosine"]

    def test_single_class_skipped(self):
        emb = np.eye(4, dtype=np.float32)
        labels = [0, 0, 0, 0]
        lmap = {0: "Normal"}
        result = run_sts(emb, labels, lmap, pairs_per_class=5, seed=0)
        assert result["cross_class_avg_cosine"] is None


# ---------------------------------------------------------------------------
# run_comparison
# ---------------------------------------------------------------------------


class TestRunComparison:
    def test_builds_report(self, tmp_path):
        m1 = {
            "model": "bilstm",
            "split": "test",
            "accuracy": 0.77,
            "macro_f1": 0.73,
            "weighted_f1": 0.75,
            "per_class": {},
            "confusion_matrix": [],
        }
        m2 = {
            "model": "bertweet",
            "split": "test",
            "accuracy": 0.88,
            "macro_f1": 0.86,
            "weighted_f1": 0.87,
            "per_class": {},
            "confusion_matrix": [],
        }

        (tmp_path / "bilstm_metrics.json").write_text(json.dumps(m1))
        (tmp_path / "bertweet_metrics.json").write_text(json.dumps(m2))

        comparison_cfg = {
            "bilstm_metrics": str(tmp_path / "bilstm_metrics.json"),
            "bertweet_metrics": str(tmp_path / "bertweet_metrics.json"),
            "llm_metrics": None,
        }
        report = run_comparison(
            comparison_cfg=comparison_cfg,
            artifacts_dir=tmp_path,
            out_name="comparison_report.json",
        )
        assert "bilstm_metrics" in report["models"]
        assert "bertweet_metrics" in report["models"]
        assert "llm_metrics" not in report["models"]  # null → skipped

    def test_ranking_by_macro_f1(self, tmp_path):
        m_low = {
            "accuracy": 0.7,
            "macro_f1": 0.65,
            "weighted_f1": 0.67,
            "per_class": {},
            "confusion_matrix": [],
        }
        m_high = {
            "accuracy": 0.9,
            "macro_f1": 0.88,
            "weighted_f1": 0.89,
            "per_class": {},
            "confusion_matrix": [],
        }
        (tmp_path / "low.json").write_text(json.dumps(m_low))
        (tmp_path / "high.json").write_text(json.dumps(m_high))

        cfg = {
            "model_a": str(tmp_path / "low.json"),
            "model_b": str(tmp_path / "high.json"),
        }
        report = run_comparison(cfg, tmp_path, "cmp.json")
        assert report["ranking_by_macro_f1"][0] == "model_b"

    def test_missing_file_skipped(self, tmp_path):
        cfg = {"bilstm_metrics": str(tmp_path / "nonexistent.json")}
        report = run_comparison(cfg, tmp_path, "cmp.json")
        assert "bilstm_metrics" not in report["models"]

    def test_output_file_created(self, tmp_path):
        m = {
            "accuracy": 0.8,
            "macro_f1": 0.78,
            "weighted_f1": 0.79,
            "per_class": {},
            "confusion_matrix": [],
        }
        (tmp_path / "m.json").write_text(json.dumps(m))
        run_comparison({"model_a": str(tmp_path / "m.json")}, tmp_path, "out.json")
        assert (tmp_path / "out.json").exists()

    def test_output_json_valid(self, tmp_path):
        m = {
            "accuracy": 0.8,
            "macro_f1": 0.78,
            "weighted_f1": 0.79,
            "per_class": {},
            "confusion_matrix": [],
        }
        (tmp_path / "m.json").write_text(json.dumps(m))
        run_comparison({"model_a": str(tmp_path / "m.json")}, tmp_path, "out.json")
        loaded = json.loads((tmp_path / "out.json").read_text())
        assert "models" in loaded
        assert "ranking_by_macro_f1" in loaded


# ---------------------------------------------------------------------------
# generate_embeddings (with mocked sentence-transformers)
# ---------------------------------------------------------------------------


class TestGenerateEmbeddings:
    def test_uses_cache_when_exists(self, tmp_path):
        cached = np.random.rand(5, 8).astype(np.float32)
        cache_path = tmp_path / "emb.npy"
        np.save(cache_path, cached)

        from scripts.run_semantic_analysis import generate_embeddings

        with patch(
            "scripts.run_semantic_analysis.generate_embeddings",
            wraps=generate_embeddings,
        ) as mock_gen:
            result = generate_embeddings(
                texts=["a", "b"],
                model_name="mock",
                cache_path=cache_path,
            )
        assert result.shape == (5, 8)

    def test_saves_cache(self, tmp_path):
        fake_embeddings = np.ones((4, 16), dtype=np.float32)
        # The function does `from sentence_transformers import SentenceTransformer`
        # so mock_st.SentenceTransformer must be the callable.
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value.encode.return_value = fake_embeddings
        cache_path = tmp_path / "new_cache.npy"

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            from scripts.run_semantic_analysis import generate_embeddings

            result = generate_embeddings(
                texts=["x"] * 4,
                model_name="mock",
                cache_path=cache_path,
            )
        assert result.shape == (4, 16)
        assert cache_path.exists()
