---
phase: implementation
title: Implementation — Feature: Dataset Preprocessing (M2)
description: How to run the preprocessing pipeline and extend it
status: done
---

# Implementation — Dataset Preprocessing (M2)

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Place raw data
# Download: https://www.kaggle.com/datasets/suchintikasarkat/sentiment-analysis-for-mental-health
# Save to: data/raw/kaggle_mental_health.csv
```

## Running the Pipeline

```bash
# Run with default config
python scripts/preprocess.py

# Override raw data path
python scripts/preprocess.py --raw_path path/to/file.csv

# Use custom config
python scripts/preprocess.py --config configs/preprocessing.yaml
```

**Outputs:**
```
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/artifacts/preprocessing_report.json
```

## Code Structure

```
src/data/
    preprocess.py   -- clean_text(), normalise_label(), preprocess_dataframe()
    dataset.py      -- Vocabulary, SentimentDataset, build_vocab_and_loaders()
scripts/
    preprocess.py   -- CLI entry-point
configs/
    preprocessing.yaml
```

## Key Implementation Notes

### Cleaning pipeline
`clean_text()` is a **pure function** — no global state, no I/O. Cleaning steps execute in a fixed order (contract §4) to ensure determinism across environments.

### Label mapping
Defined in `configs/preprocessing.yaml` under `label_map`. If a raw label does not match any key (case-insensitive), the row is dropped with a warning — never silently reassigned.

### Split reproducibility
Uses `sklearn.model_selection.train_test_split` with `stratify=label_id` and `random_state=42`. The seed is also stored in config for easy auditing.

### Quality report schema
```json
{
  "initial_count": 30000,
  "dropped": {
    "null_rows": 12,
    "short_text": 45,
    "unknown_labels": 3,
    "duplicates": 120
  },
  "final_count": 29820,
  "class_counts": {"0": 5000, "1": 8000, ...},
  "splits": {"train": 20874, "val": 4473, "test": 4473}
}
```

## Extending the Pipeline

- **Add a new label**: add the key-value pair to `label_map` in `configs/preprocessing.yaml`. No code change required.
- **Different cleaning rules**: modify `clean_text()` in `src/data/preprocess.py`. Add a unit test to `tests/test_preprocessing.py`.
- **Custom split ratio**: edit `split.*` keys in the config.

