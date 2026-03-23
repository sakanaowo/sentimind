---
phase: planning
title: Project Planning & Task Breakdown — BERTweet Fine-tuning (M4)
description: Break down work into actionable tasks and estimate timeline
---

# M4 — BERTweet Fine-tuning: Planning & Task Breakdown

## Milestones

- [x] M4.1: Config, dataset, and model skeleton in place
- [x] M4.2: Training and evaluation scripts operational
- [x] M4.3: Unit tests passing (no real model download required)

## Task Breakdown

### Phase 1: Foundation

- [x] Task 1.1: Create `configs/bertweet.yaml` — mirrors bilstm.yaml structure, adds HF-specific keys (`pretrained_name`, `gradient_accumulation_steps`, `warmup_ratio`, `fp16`).
- [x] Task 1.2: Create `src/data/bertweet_dataset.py` — `TransformerSentimentDataset` (HF tokenizer) + `build_transformer_loaders()` factory.
- [x] Task 1.3: Create `src/models/bertweet.py` — `BERTweetClassifier` wrapping `AutoModelForSequenceClassification`; supports `freeze_base` flag; `save_checkpoint` / `from_checkpoint`.

### Phase 2: Core Features

- [x] Task 2.1: Create `scripts/train_bertweet.py` — full fine-tuning loop with gradient accumulation, linear warmup (via `get_linear_schedule_with_warmup`), optional AMP (fp16), early stopping on `macro_f1`, training history JSON.
- [x] Task 2.2: Create `scripts/eval_bertweet.py` — load checkpoint, run test inference, save metrics + confusion matrix using shared `compute_metrics` / `save_metrics`.

### Phase 3: Testing & Integration

- [x] Task 3.1: Create `tests/test_bertweet_model.py` — 100 % mocked (no HF download): `TransformerSentimentDataset`, `BERTweetClassifier` forward shape, checkpoint round-trip, `build_transformer_loaders`, metric schema compatibility.
- [x] Task 3.2: Verify metric output matches data_contract §7 schema so M6 comparison script can pick it up automatically.

## Dependencies

- **Requires**: `data/processed/{train,val,test}.csv` from M2 (run `scripts/preprocess.py` first).
- **Provides**: `data/artifacts/bertweet_best.pt`, `bertweet_metrics.json`, `bertweet_confusion_matrix.png`.
- `transformers>=4.30.0`, `tokenizers>=0.13.0` (already in `requirements.txt`).

## Risks & Mitigation

| Risk                                                 | Mitigation                                                                                      |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| GPU OOM with `vinai/bertweet-base`                   | Reduce `batch_size` + increase `gradient_accumulation_steps` in YAML                            |
| Class 4 (Personality disorder) absent in data        | `class_weighted_loss` fills weight; zero-support class gets 0 F1 in report — expected behaviour |
| Domain shift: BERTweet pre-trained on generic tweets | Full fine-tuning (default); `freeze_base: true` option for head-only training                   |

## Resources Needed

- GPU with ≥8 GB VRAM (or CPU fallback; ~6× slower)
- HuggingFace Hub access to download `vinai/bertweet-base`
