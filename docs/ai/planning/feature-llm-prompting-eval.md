---
phase: planning
title: Project Planning & Task Breakdown — LLM Prompting Evaluation (M5)
description: Break down work into actionable tasks and estimate timeline
---

# M5 — LLM Prompting Evaluation: Planning & Task Breakdown

## Milestones

- [x] M5.1: Config, prompt templates, and LLM client implemented
- [x] M5.2: Inference script with retry, cost tracking, and JSONL output
- [x] M5.3: Unit tests passing (all mocked — no real API key needed)

## Task Breakdown

### Phase 1: Foundation

- [x] Task 1.1: Create `configs/llm_prompting.yaml` — provider, model, generation params, cost caps, label map, output paths.
- [x] Task 1.2: Create `src/models/llm_client.py`:
  - `LLMClient` — OpenAI-compatible chat client with retry and format-error counting.
  - `CostAccumulator` — tracks token usage and estimated USD cost with hard budget cap.
  - `_build_zero_shot_messages` / `_build_few_shot_messages` — deterministic prompt builders.
  - `_extract_json` — robust JSON extraction with markdown-fence fallback.
  - `_normalise_label` — maps LLM raw output to canonical label_id.

### Phase 2: Core Features

- [x] Task 2.1: Create `scripts/run_llm_prompting.py` — batch inference with tqdm progress, JSONL streaming output (safe against mid-run interrupts), metrics + confusion matrix + cost report saved to `data/artifacts/`.
- [x] Task 2.2: Support both `zero_shot` and `few_shot` modes; few-shot examples selectable from built-in curated set or random sample from train split.

### Phase 3: Testing & Integration

- [x] Task 3.1: Create `tests/test_llm_client.py` — fully mocked (no API key): JSON extraction, label normalisation, prompt builder structure, cost accumulator arithmetic, `LLMClient.classify` happy path + parse error + budget cap.
- [x] Task 3.2: `llm_metrics.json` written in same schema as bilstm/bertweet for M6 comparison.

## Dependencies

- **Requires**: `data/processed/test.csv` and `OPENAI_API_KEY` env var (or alternative API key env).
- **Provides**: `llm_predictions.jsonl`, `llm_metrics.json`, `llm_cost_report.json`, `llm_confusion_matrix.png`.
- `openai>=1.0.0` added to `requirements.txt`.

## Risks & Mitigation

| Risk                                        | Mitigation                                                                          |
| ------------------------------------------- | ----------------------------------------------------------------------------------- |
| API costs during full-set eval              | `sample_size` cap in config (default 200); `budget_cap_usd` hard limit              |
| Response format drift / hallucinated labels | `max_retries` loop + `parse_error` flag + filtering before metric computation       |
| Provider outage                             | `_normalise_label` returns `(-1, 'unknown')` — invalid predictions excluded from F1 |

## Resources Needed

- OpenAI (or compatible) API key
- Internet access to LLM provider endpoint
