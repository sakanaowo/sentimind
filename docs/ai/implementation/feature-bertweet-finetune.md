---
phase: implementation
title: Implementation Guide — BERTweet Fine-tuning (M4)
description: Technical implementation notes, patterns, and code guidelines
---

# M4 — BERTweet Fine-tuning: Implementation Guide

## Key Implementation Patterns

### HuggingFace Auto-classes
```python
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
model  = AutoModelForSequenceClassification.from_pretrained(
    model_name, config=config, ignore_mismatched_sizes=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```
`ignore_mismatched_sizes=True` is required because the pre-trained checkpoint has a different number of output classes.

### Gradient Accumulation Loop
```python
optimizer.zero_grad()
for step, batch in enumerate(train_loader):
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss   = criterion(logits, batch["labels"]) / cfg.gradient_accumulation_steps
    loss.backward()
    if (step + 1) % cfg.gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### Checkpoint Serialisation
`BERTweetClassifier.save_checkpoint(path, epoch, best_metric)` stores:
```json
{
  "model_state_dict": {...},
  "model_name": "vinai/bertweet-base",
  "num_classes": 7,
  "epoch": 5,
  "best_metric": 0.812
}
```
`BERTweetClassifier.from_checkpoint(path, device)` reconstructs architecture from the embedded `model_name` + `num_classes` before loading state dict.

## Testing Patterns

All tests in `tests/test_bertweet_model.py` mock HuggingFace model loading to avoid network downloads:

```python
from unittest.mock import MagicMock, patch

@patch("src.models.bertweet.AutoConfig.from_pretrained")
@patch("src.models.bertweet.AutoModelForSequenceClassification.from_pretrained")
def test_forward_shape(self, mock_model_cls, mock_config):
    mock_hf = MagicMock()
    mock_hf.return_value.logits = torch.zeros(2, 7)
    mock_model_cls.return_value = mock_hf
    ...
```
The `TransformerSentimentDataset` tests also mock `AutoTokenizer.from_pretrained` to return a simple lambda tokenizer.

## Configuration Reference

```yaml
# configs/bertweet.yaml (key fields)
pretrained_name: vinai/bertweet-base
num_classes: 7
batch_size: 16
gradient_accumulation_steps: 2   # effective batch = 32
lr: 2.0e-5
warmup_ratio: 0.06
epochs: 10
early_stopping_patience: 3
fp16: false                       # set true on GPU for ~2× speedup
```

## Running the Scripts

```bash
# Fine-tune:
python scripts/train_bertweet.py --config configs/bertweet.yaml

# Evaluate on test split:
python scripts/eval_bertweet.py --config configs/bertweet.yaml \
    --checkpoint data/artifacts/bertweet_best.pt
```

## Known Issues / Gotchas

- `vinai/bertweet-base` tokenizer normalises Twitter mentions/URLs (`@USER`, `HTTPURL`) — input text should NOT be pre-cleaned of these if using the standard BERTweet tokenizer.
- `freeze_base: true` is useful for quick experiments but typically yields ~3–5 % lower macro_F1 than full fine-tuning.
- On CPU, one training epoch (~35 000 train rows) takes approximately 45–60 minutes.
