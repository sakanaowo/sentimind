---
version: "1.0"
status: locked
authors: ["Person A", "Person B"]
---

# Data Contract — Sentimind

> **Locked document.** To change this contract, open a PR on branch `contract-change/*` and have both team members approve before merging.

---

## 1. Raw Input Schema

### 1.1 Kaggle Mental Health dataset (`data/raw/kaggle_mental_health.csv`)

| Column  | Type | Description                        | Nullable |
|---------|------|------------------------------------|----------|
| `text`  | str  | Raw social-media text              | No       |
| `label` | str  | Condition label (see §3)           | No       |

### 1.2 TweetEval sentiment subset (optional transfer source)

Loaded via HuggingFace `datasets` library (`cardiffnlp/tweet_eval`, split `sentiment`).
Used only for pre-training / transfer; NOT mixed into eval split.

---

## 2. Processed Output Schema

Files: `data/processed/{train,val,test}.csv`

| Column     | Type | Description                          | Nullable |
|------------|------|--------------------------------------|----------|
| `text`     | str  | Cleaned text (see §4 cleaning rules) | No       |
| `label`    | str  | Original label string (normalised)   | No       |
| `label_id` | int  | Integer class index (see §3)         | No       |

---

## 3. Label Mapping

| `label_id` | `label`    | Notes                       |
|------------|------------|-----------------------------|
| 0          | Normal     | No mental-health indicator  |
| 1          | Depression |                             |
| 2          | Anxiety    |                             |
| 3          | Bipolar    |                             |
| 4          | PTSD       |                             |
| 5          | Stress     |                             |
| 6          | Suicidal   | May be merged with Depression per analysis |

Label normalisation: lower-case raw label → look up mapping → drop unknown labels with a warning.

---

## 4. Text Cleaning Rules (deterministic, in order)

1. Unicode NFKC normalisation.
2. Remove URLs (`http://…`, `https://…`, `www.…`).
3. Remove @mentions.
4. Preserve hashtag words — strip `#` prefix only (e.g. `#mentalhealth` → `mentalhealth`).
5. Remove HTML entities (`&amp;`, `&lt;`, …).
6. Remove non-printable control characters.
7. Collapse whitespace to single space and strip.
8. Lowercase entire string.

**Minimum valid text length after cleaning: 3 characters.** Shorter rows are dropped and logged.

---

## 5. Split Ratios

| Split | Fraction | Seed |
|-------|----------|------|
| Train | 70 %     | 42   |
| Val   | 15 %     | 42   |
| Test  | 15 %     | 42   |

Strategy: **stratified** by `label_id`.

---

## 6. Artifact Paths

| Artifact                              | Path                                          |
|---------------------------------------|-----------------------------------------------|
| Raw Kaggle CSV                        | `data/raw/kaggle_mental_health.csv`           |
| Processed train split                 | `data/processed/train.csv`                    |
| Processed val split                   | `data/processed/val.csv`                      |
| Processed test split                  | `data/processed/test.csv`                     |
| Preprocessing quality report          | `data/artifacts/preprocessing_report.json`   |
| Vocabulary (word → index)             | `data/artifacts/vocab.json`                   |
| BiLSTM best checkpoint                | `data/artifacts/bilstm_best.pt`               |
| BiLSTM metrics                        | `data/artifacts/bilstm_metrics.json`          |
| BiLSTM confusion matrix               | `data/artifacts/bilstm_confusion_matrix.png`  |

---

## 7. Metric Contract (all models must output this schema)

```json
{
  "model": "<string>",
  "split": "test",
  "accuracy": 0.0,
  "macro_f1": 0.0,
  "weighted_f1": 0.0,
  "per_class": {
    "<label_name>": {
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "support": 0
    }
  },
  "confusion_matrix": [[0]]
}
```

---

## 8. Reproducibility Guarantees

- All random seeds pinned to **42** (Python, NumPy, PyTorch).
- Split files are regenerated deterministically from the same raw CSV.
- Preprocessing is a pure function: same input → same output (no randomness).
