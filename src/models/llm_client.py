"""
LLM client for zero-shot / few-shot mental-health label classification.

Uses Google Gemini API (google-genai SDK). Supports gemini-2.5-flash and
gemini-2.5-pro. Structured output: each response must contain a JSON block
with keys ``label``, ``confidence``, and ``explanation``.

Security note:
  - The API key is read from an environment variable (never hard-coded).
  - User-supplied text is inserted into the *user* turn only, not the system
    instruction, so prompt-injection into the system prompt is not possible.
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
    """Wrapper around the Google Gemini API (google-genai SDK).

    Supports ``gemini-2.5-flash`` (default, fast & cheap) and
    ``gemini-2.5-pro`` (higher quality). The interface is intentionally
    kept identical to the previous OpenAI-based version so that
    ``run_llm_prompting.py`` and all downstream code work unchanged.

    Args:
        model: Gemini model name, e.g. ``"gemini-2.5-flash"``.
        api_key_env: name of the environment variable holding the Google API key.
        base_url: unused; kept for interface compatibility.
        temperature: sampling temperature (0 for deterministic output).
        max_tokens: maximum output tokens per request.
        request_timeout: HTTP timeout in seconds (applied at client level).
        max_retries: number of retries on transient errors or parse failures.
        input_price_per_1k: cost per 1 000 input tokens in USD.
        output_price_per_1k: cost per 1 000 output tokens in USD.
        budget_cap_usd: abort when estimated cumulative cost exceeds this.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GOOGLE_API_KEY",
        base_url: Optional[str] = None,  # kept for interface compat, not used
        temperature: float = 0.0,
        max_tokens: int = 256,
        request_timeout: int = 30,
        max_retries: int = 3,
        input_price_per_1k: float = 0.000075,  # gemini-2.5-flash input pricing
        output_price_per_1k: float = 0.000300,  # gemini-2.5-flash output pricing
        budget_cap_usd: float = 5.0,
    ):
        try:
            from google import genai
            from google.genai import types as gentypes
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for LLM prompting. "
                "Install it with: pip install google-genai>=1.0.0"
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"API key not found. Set the '{api_key_env}' environment variable."
            )

        self._client = genai.Client(
            api_key=api_key,
            http_options=gentypes.HttpOptions(timeout=request_timeout * 1000),
        )
        self._gentypes = gentypes
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

    def _messages_to_gemini(self, messages: List[dict]):
        """Convert OpenAI-style message list to Gemini system_instruction + contents."""
        types = self._gentypes
        system_instruction: Optional[str] = None
        contents = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=content)])
                )
            elif role == "assistant":
                # Gemini uses "model" as the assistant role
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=content)])
                )
        return system_instruction, contents

    def classify(
        self,
        text: str,
        label_map: Dict[int, str],
        mode: str = "zero_shot",
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> LLMPrediction:
        """Classify a single text using Gemini.

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
        types = self._gentypes

        for attempt in range(1, self.max_retries + 1):
            if mode == "few_shot" and few_shot_examples:
                messages = _build_few_shot_messages(
                    text, few_shot_examples, label_names
                )
            else:
                messages = _build_zero_shot_messages(text, label_names)

            system_instruction, contents = self._messages_to_gemini(messages)

            t0 = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        # Disable thinking for classification (thinking tokens
                        # eat into max_output_tokens on gemini-2.5-flash,
                        # leaving too little budget for the JSON response).
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
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
            raw = response.text or ""
            prompt_tokens = (
                getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            )
            completion_tokens = (
                getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            )
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
