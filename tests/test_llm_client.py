"""
Unit tests for LLM prompting client (no real API calls — fully mocked).

Run: pytest tests/test_llm_client.py -v
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.models.llm_client import (
    CostAccumulator,
    LLMClient,
    LLMPrediction,
    _build_few_shot_messages,
    _build_zero_shot_messages,
    _extract_json,
    _normalise_label,
)

# ---------------------------------------------------------------------------
# Label map shared across tests
# ---------------------------------------------------------------------------

LABEL_MAP = {
    0: "Normal",
    1: "Depression",
    2: "Anxiety",
    3: "Bipolar",
    4: "Personality disorder",
    5: "Stress",
    6: "Suicidal",
}


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_clean_json(self):
        raw = '{"label": "Anxiety", "confidence": 0.9, "explanation": "test"}'
        result = _extract_json(raw)
        assert result is not None
        assert result["label"] == "Anxiety"

    def test_json_with_preamble(self):
        raw = 'Sure! Here is my answer:\n{"label": "Normal", "confidence": 0.8, "explanation": "ok"}'
        result = _extract_json(raw)
        assert result is not None
        assert result["label"] == "Normal"

    def test_markdown_fenced_json(self):
        raw = (
            '```json\n{"label": "Stress", "confidence": 0.7, "explanation": "..."}\n```'
        )
        result = _extract_json(raw)
        assert result is not None
        assert result["label"] == "Stress"

    def test_invalid_json_returns_none(self):
        raw = "I think it is Depression."
        result = _extract_json(raw)
        assert result is None

    def test_empty_string_returns_none(self):
        assert _extract_json("") is None


# ---------------------------------------------------------------------------
# _normalise_label
# ---------------------------------------------------------------------------


class TestNormaliseLabel:
    def test_exact_match(self):
        label, lid = _normalise_label("Anxiety", LABEL_MAP)
        assert label == "Anxiety"
        assert lid == 2

    def test_case_insensitive(self):
        label, lid = _normalise_label("depression", LABEL_MAP)
        assert lid == 1

    def test_prefix_match(self):
        # "personality" matches "Personality disorder"
        label, lid = _normalise_label("Personality", LABEL_MAP)
        assert lid == 4

    def test_unknown_label(self):
        label, lid = _normalise_label("Schizophrenia", LABEL_MAP)
        assert label == "unknown"
        assert lid == -1


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


class TestPromptBuilders:
    def test_zero_shot_has_system_and_user(self):
        msgs = _build_zero_shot_messages("I feel hopeless.", list(LABEL_MAP.values()))
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user"]

    def test_zero_shot_user_contains_text(self):
        text = "I feel hopeless today."
        msgs = _build_zero_shot_messages(text, list(LABEL_MAP.values()))
        assert text in msgs[-1]["content"]

    def test_few_shot_has_assistant_turns(self):
        examples = [
            {"text": "example text", "label": "Normal", "explanation": "fine"},
        ]
        msgs = _build_few_shot_messages("new post", examples, list(LABEL_MAP.values()))
        roles = [m["role"] for m in msgs]
        assert "assistant" in roles

    def test_few_shot_last_message_is_user(self):
        examples = [
            {"text": "ex", "label": "Stress", "explanation": "stress"},
        ]
        msgs = _build_few_shot_messages(
            "classify this", examples, list(LABEL_MAP.values())
        )
        assert msgs[-1]["role"] == "user"

    def test_system_prompt_contains_all_labels(self):
        msgs = _build_zero_shot_messages("text", list(LABEL_MAP.values()))
        system_content = msgs[0]["content"]
        for label in LABEL_MAP.values():
            assert label in system_content


# ---------------------------------------------------------------------------
# CostAccumulator
# ---------------------------------------------------------------------------


class TestCostAccumulator:
    def test_initial_cost_is_zero(self):
        acc = CostAccumulator(
            input_price_per_1k=0.00015,
            output_price_per_1k=0.00060,
            budget_cap_usd=5.0,
        )
        assert acc.estimated_cost_usd == 0.0

    def test_cost_accumulation(self):
        acc = CostAccumulator(
            input_price_per_1k=0.001,
            output_price_per_1k=0.002,
            budget_cap_usd=5.0,
        )
        acc.update(1000, 1000)
        # 1000 input @ 0.001/1k = 0.001
        # 1000 output @ 0.002/1k = 0.002
        assert abs(acc.estimated_cost_usd - 0.003) < 1e-9

    def test_budget_not_exceeded_initially(self):
        acc = CostAccumulator(0.001, 0.002, budget_cap_usd=5.0)
        assert not acc.budget_exceeded()

    def test_budget_exceeded(self):
        acc = CostAccumulator(0.001, 0.002, budget_cap_usd=0.001)
        acc.update(10_000, 0)
        assert acc.budget_exceeded()

    def test_to_dict_keys(self):
        acc = CostAccumulator(0.001, 0.002, 5.0)
        acc.update(100, 50)
        d = acc.to_dict()
        assert "total_prompt_tokens" in d
        assert "estimated_cost_usd" in d
        assert "total_requests" in d


# ---------------------------------------------------------------------------
# LLMClient (mocked OpenAI)
# ---------------------------------------------------------------------------


def _make_mock_gemini_response(label: str, confidence: float = 0.9) -> MagicMock:
    """Construct a minimal mock Gemini generate_content response."""
    resp = MagicMock()
    resp.text = json.dumps(
        {
            "label": label,
            "confidence": confidence,
            "explanation": "mock explanation",
        }
    )
    resp.usage_metadata.prompt_token_count = 50
    resp.usage_metadata.candidates_token_count = 30
    return resp


@pytest.fixture
def mock_client(monkeypatch):
    """LLMClient with Gemini constructor mocked and API key in environment."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    with patch("src.models.llm_client.LLMClient.__init__") as mock_init:
        mock_init.return_value = None
        client = LLMClient.__new__(LLMClient)

    # Set up client attributes directly (mirrors __init__ assignments)
    mock_gemini_client = MagicMock()
    client._client = mock_gemini_client
    client._gentypes = MagicMock()
    client.model = "gemini-2.5-flash"
    client.temperature = 0.0
    client.max_tokens = 256
    client.request_timeout = 60
    client.max_retries = 3
    client.cost = CostAccumulator(0.000075, 0.000300, budget_cap_usd=5.0)

    mock_gemini_client.models.generate_content.return_value = _make_mock_gemini_response(
        "Anxiety"
    )
    return client


class TestLLMClient:
    def test_classify_returns_prediction(self, mock_client):
        pred = mock_client.classify("I feel anxious", LABEL_MAP)
        assert isinstance(pred, LLMPrediction)

    def test_classify_correct_label_id(self, mock_client):
        # Mock returns "Anxiety" → label_id=2
        pred = mock_client.classify("test text", LABEL_MAP)
        assert pred.predicted_label_id == 2
        assert pred.predicted_label == "Anxiety"

    def test_classify_updates_cost(self, mock_client):
        before = mock_client.cost.total_prompt_tokens
        mock_client.classify("test", LABEL_MAP)
        assert mock_client.cost.total_prompt_tokens > before

    def test_classify_parse_failure_sets_flag(self, mock_client):
        # Return garbled JSON
        mock_client._client.models.generate_content.return_value.text = (
            "I think it is something."
        )
        mock_client.max_retries = 1
        pred = mock_client.classify("some text", LABEL_MAP)
        assert pred.parse_error is True
        assert pred.predicted_label_id == -1

    def test_classify_budget_exceeded_raises(self, mock_client):
        mock_client.cost.budget_cap_usd = 0.0
        mock_client.cost.update(1, 1)  # triggers exceeded
        with pytest.raises(RuntimeError, match="Budget cap"):
            mock_client.classify("test", LABEL_MAP)

    def test_classify_few_shot_mode(self, mock_client):
        examples = [
            {"text": "ex1", "label": "Normal", "explanation": "fine"},
            {"text": "ex2", "label": "Stress", "explanation": "stressed"},
        ]
        pred = mock_client.classify(
            "new post", LABEL_MAP, mode="few_shot", few_shot_examples=examples
        )
        assert isinstance(pred, LLMPrediction)
        # API should have received a call regardless of mode
        mock_client._client.models.generate_content.assert_called()
