"""
LLM client for zero-shot / few-shot mental-health label classification.

Supports any OpenAI-compatible chat API (OpenAI, Azure OpenAI, local Ollama, etc.).
Structured output: each response must contain a JSON block with keys
  ``label``, ``confidence``, and ``explanation``.

Security note:
  - The API key is read from an environment variable (never hard-coded).
  - User-supplied text is inserted into the *user* turn only, not the system prompt,
    so prompt-injection into the system instruction is not possible.
  - The budget cap prevents runaway API costs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LLMPrediction:
    """Structured output for a single LLM classification call."""

    text: str  # original input text
    raw_response: str  # full text returned by the model
    predicted_label: str  # normalised label string
    predicted_label_id: int  # integer index (-1 if parse failed)
    confidence: float  # LLM-reported confidence proxy (0–1)
    explanation: str  # free-text rationale
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    parse_error: bool = False  # True when JSON extraction failed


@dataclass
class CostAccumulator:
    """Tracks cumulative token usage and estimated cost."""

    input_price_per_1k: float
    output_price_per_1k: float
    budget_cap_usd: float

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    failed_requests: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.total_prompt_tokens / 1000 * self.input_price_per_1k
            + self.total_completion_tokens / 1000 * self.output_price_per_1k
        )

    def update(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1

    def budget_exceeded(self) -> bool:
        return self.estimated_cost_usd >= self.budget_cap_usd

    def to_dict(self) -> dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# All valid label names that the LLM must choose from.
VALID_LABELS = [
    "Normal",
    "Depression",
    "Anxiety",
    "Bipolar",
    "Personality disorder",
    "Stress",
    "Suicidal",
]

_SYSTEM_PROMPT = """\
You are a clinical NLP assistant. Your task is to classify social-media posts \
into exactly one mental-health category.

Available categories (use the EXACT spelling):
{label_list}

Rules:
1. Read the post carefully and choose the SINGLE best-matching category.
2. Reply with a valid JSON object — no extra text before or after — in this format:
{{
  "label": "<category>",
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<one sentence rationale>"
}}
3. "label" must be one of the categories listed above (case-insensitive matching is applied).
4. "confidence" reflects how certain you are (1.0 = very certain, 0.0 = pure guess).
""".strip()

_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "text": "i haven't left my bed in three days. nothing feels worth it anymore.",
        "label": "Depression",
        "explanation": "Persistent low energy and anhedonia are hallmarks of depression.",
    },
    {
        "text": "my heart starts racing as soon as i think about tomorrow's presentation.",
        "label": "Anxiety",
        "explanation": "Physical arousal triggered by anticipation is a classic anxiety symptom.",
    },
    {
        "text": "had a productive morning, helped a friend, and feeling grateful.",
        "label": "Normal",
        "explanation": "Positive, adaptive functioning with no mental-health indicators.",
    },
]


def _build_system_prompt(label_names: Optional[List[str]] = None) -> str:
    labels = label_names or VALID_LABELS
    return _SYSTEM_PROMPT.format(label_list="\n".join(f"  • {l}" for l in labels))


def _build_zero_shot_messages(
    text: str, label_names: Optional[List[str]] = None
) -> List[dict]:
    return [
        {"role": "system", "content": _build_system_prompt(label_names)},
        {"role": "user", "content": f"Post:\n{text}"},
    ]


def _build_few_shot_messages(
    text: str,
    examples: List[Dict[str, str]],
    label_names: Optional[List[str]] = None,
) -> List[dict]:
    messages: List[dict] = [
        {"role": "system", "content": _build_system_prompt(label_names)},
    ]
    for ex in examples:
        messages.append({"role": "user", "content": f"Post:\n{ex['text']}"})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "label": ex["label"],
                        "confidence": 0.95,
                        "explanation": ex["explanation"],
                    }
                ),
            }
        )
    messages.append({"role": "user", "content": f"Post:\n{text}"})
    return messages


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json(raw: str) -> Optional[dict]:
    """Extract and parse the first JSON object found in *raw*."""
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try stripping markdown fences
    stripped = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------


def _normalise_label(raw_label: str, label_map: Dict[int, str]) -> Tuple[str, int]:
    """Map the LLM's raw label string to (normalised_label, label_id).

    Returns ("unknown", -1) when no match is found.
    """
    raw_lower = raw_label.strip().lower()
    for lid, lname in label_map.items():
        if lname.lower() == raw_lower or lname.lower().startswith(raw_lower):
            return lname, lid
    return "unknown", -1


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin wrapper around the OpenAI chat completion API.

    Supports any OpenAI-compatible endpoint by setting ``base_url``.

    Args:
        model: model name, e.g. ``"gpt-4o-mini"``.
        api_key_env: name of the environment variable holding the API key.
        base_url: override base URL (e.g. ``"http://localhost:11434/v1"`` for Ollama).
        temperature: sampling temperature (0 for deterministic output).
        max_tokens: maximum completion tokens.
        request_timeout: HTTP timeout in seconds.
        max_retries: number of retries on transient errors or parse failures.
        input_price_per_1k: cost per 1 000 input tokens in USD.
        output_price_per_1k: cost per 1 000 output tokens in USD.
        budget_cap_usd: abort when estimated cumulative cost exceeds this.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
        request_timeout: int = 30,
        max_retries: int = 3,
        input_price_per_1k: float = 0.00015,
        output_price_per_1k: float = 0.00060,
        budget_cap_usd: float = 5.0,
    ):
        # Import lazily to avoid hard dependency if the module is not used
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for LLM prompting. "
                "Install it with: pip install openai>=1.0.0"
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"API key not found. Set the '{api_key_env}' environment variable."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        self.cost = CostAccumulator(
            input_price_per_1k=input_price_per_1k,
            output_price_per_1k=output_price_per_1k,
            budget_cap_usd=budget_cap_usd,
        )

    def classify(
        self,
        text: str,
        label_map: Dict[int, str],
        mode: str = "zero_shot",
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> LLMPrediction:
        """Classify a single text using the configured LLM.

        Args:
            text: the social-media post to classify.
            label_map: mapping from label_id → label_name (used for normalisation).
            mode: ``"zero_shot"`` or ``"few_shot"``.
            few_shot_examples: list of example dicts (required when mode='few_shot').

        Returns:
            LLMPrediction with parsed label, confidence, explanation, and token usage.
        """
        if self.cost.budget_exceeded():
            raise RuntimeError(
                f"Budget cap of ${self.cost.budget_cap_usd} reached. "
                "Increase budget_cap_usd in config to continue."
            )

        label_names = list(label_map.values())

        for attempt in range(1, self.max_retries + 1):
            if mode == "few_shot" and few_shot_examples:
                messages = _build_few_shot_messages(
                    text, few_shot_examples, label_names
                )
            else:
                messages = _build_zero_shot_messages(text, label_names)

            t0 = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.request_timeout,
                )
            except Exception as exc:
                logger.warning(
                    "API call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries:
                    self.cost.failed_requests += 1
                    return LLMPrediction(
                        text=text,
                        raw_response="",
                        predicted_label="unknown",
                        predicted_label_id=-1,
                        confidence=0.0,
                        explanation="API call failed",
                        parse_error=True,
                    )
                time.sleep(2**attempt)
                continue

            latency = time.perf_counter() - t0
            raw = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            self.cost.update(prompt_tokens, completion_tokens)

            parsed = _extract_json(raw)
            if parsed is None or "label" not in parsed:
                logger.warning(
                    "JSON parse failed (attempt %d/%d). Raw: %r",
                    attempt,
                    self.max_retries,
                    raw[:200],
                )
                if attempt == self.max_retries:
                    self.cost.failed_requests += 1
                    return LLMPrediction(
                        text=text,
                        raw_response=raw,
                        predicted_label="unknown",
                        predicted_label_id=-1,
                        confidence=0.0,
                        explanation="parse failure",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_s=latency,
                        parse_error=True,
                    )
                continue  # retry

            norm_label, label_id = _normalise_label(parsed["label"], label_map)
            return LLMPrediction(
                text=text,
                raw_response=raw,
                predicted_label=norm_label,
                predicted_label_id=label_id,
                confidence=float(parsed.get("confidence", 0.0)),
                explanation=str(parsed.get("explanation", "")),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_s=latency,
                parse_error=label_id == -1,
            )

        # Should not reach here
        self.cost.failed_requests += 1
        return LLMPrediction(
            text=text,
            raw_response="",
            predicted_label="unknown",
            predicted_label_id=-1,
            confidence=0.0,
            explanation="max retries exceeded",
            parse_error=True,
        )
