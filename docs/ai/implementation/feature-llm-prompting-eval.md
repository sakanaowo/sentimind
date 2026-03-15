---
phase: implementation
title: Implementation Guide — LLM Prompting Evaluation (M5)
description: Technical implementation notes, patterns, and code guidelines
---

# M5 — LLM Prompting Evaluation: Implementation Guide

## Key Implementation Patterns

### API Key Handling
```python
import os
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
```
The key is **never** stored in `configs/llm_prompting.yaml` or source code.

### Structuring the Prompt
```python
SYSTEM_MSG = (
    "You are a mental health text classifier. "
    "Classify the text into exactly one of these categories: "
    "{label_list}. "
    'Respond ONLY with valid JSON: {"label": "<category>", '
    '"confidence": <0.0-1.0>, "explanation": "<one sentence>"}.'
)
```

### JSON Extraction with Fallback
```python
def _extract_json(text: str) -> dict:
    # 1. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Strip markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # 3. Find first {...} block
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON found in: {text!r}")
```

### CostAccumulator Usage
```python
cost = CostAccumulator(
    input_price_per_1k=cfg.cost.input_price_per_1k,
    output_price_per_1k=cfg.cost.output_price_per_1k,
)
for text in sample:
    pred = client.classify(text, label_map, mode=cfg.mode)
    cost.add(pred.prompt_tokens, pred.completion_tokens)
    cost.check_budget(cfg.budget_cap_usd)  # raises RuntimeError if exceeded
```

## Testing Patterns

`tests/test_llm_client.py` patches the OpenAI constructor to avoid real API calls:
```python
@patch("src.models.llm_client.OpenAI")
def test_classify_happy_path(self, mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        '{"label": "depression", "confidence": 0.9, "explanation": "test"}'
    )
    mock_client.chat.completions.create.return_value.usage.prompt_tokens = 50
    mock_client.chat.completions.create.return_value.usage.completion_tokens = 20
    ...
```

## Configuration Reference

```yaml
# configs/llm_prompting.yaml (key fields)
provider: openai
model: gpt-4o-mini
mode: zero_shot          # zero_shot | few_shot
sample_size: 200
budget_cap_usd: 5.0
max_retries: 3
temperature: 0.0         # deterministic output
cost:
  input_price_per_1k: 0.00015
  output_price_per_1k: 0.00060
```

## Running the Script

```bash
export OPENAI_API_KEY=sk-...

# Zero-shot (default):
python scripts/run_llm_prompting.py --config configs/llm_prompting.yaml

# Few-shot:
python scripts/run_llm_prompting.py --config configs/llm_prompting.yaml --mode few_shot
```

## Known Issues / Gotchas

- `gpt-4o-mini` returns valid JSON for ~97 % of inputs; the remaining ~3 % are logged as `parse_error=True` and excluded from metric computation.
- Temperature `0.0` is recommanded for reproducibility; some providers treat it as `top_p=1.0` internally.
- `sample_size` should be kept ≤ 500 during development to stay within the default budget cap.
- The few-shot examples embedded in `_FEW_SHOT_EXAMPLES` are short and generic; domain-specific examples should improve accuracy.
