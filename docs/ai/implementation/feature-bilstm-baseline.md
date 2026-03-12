---
phase: implementation
title: Implementation — Feature: BiLSTM Baseline (M3)
description: How to train, evaluate, and reproduce the BiLSTM classifier
status: done
---

# Implementation — BiLSTM Baseline (M3)

## Prerequisites

```bash
pip install -r requirements.txt
# Ensure M2 preprocessing has been run:
python scripts/preprocess.py
```

## Training

```bash
# Train with default config (CPU auto-detected)
python scripts/train_bilstm.py

# Force GPU
python scripts/train_bilstm.py --device cuda

# Custom config
python scripts/train_bilstm.py --config configs/bilstm.yaml
```

**Outputs:**
```
data/artifacts/bilstm_best.pt          # best checkpoint (by macro F1)
data/artifacts/bilstm_train_history.json
data/artifacts/vocab.json
```

## Evaluation

```bash
# Evaluate on test split (default)
python scripts/eval_bilstm.py

# Evaluate on val split
python scripts/eval_bilstm.py --split val
```

**Outputs:**
```
data/artifacts/bilstm_metrics.json          # metric contract JSON
data/artifacts/bilstm_confusion_matrix.png
```

## Code Structure

```
src/
    models/bilstm.py        -- BiLSTMClassifier architecture + GloVe loader
    data/dataset.py         -- Vocabulary, SentimentDataset, dataloaders
    training/trainer.py     -- train(), EarlyStopping, _train_epoch, _eval_epoch
    utils/metrics.py        -- compute_metrics(), save_metrics(), confusion matrix
scripts/
    train_bilstm.py         -- CLI training entry-point
    eval_bilstm.py          -- CLI evaluation entry-point
configs/
    bilstm.yaml             -- all hyperparameters
```

## Key Hyperparameters

| Parameter | Default | Why |
|---|---|---|
| `embedding_dim` | 300 | Compatible with GloVe Twitter 27B 300d |
| `hidden_dim` | 256 | Balances capacity vs. training speed |
| `num_layers` | 2 | Two-layer BiLSTM captures more abstract patterns |
| `dropout` | 0.3 | Regularisation for limited dataset |
| `max_seq_len` | 128 | Covers 99%+ of social media text lengths |
| `early_stopping_patience` | 5 | Prevents premature stopping on noisy val curve |
| `class_weighted_loss` | true | Handles label imbalance |

## Using Pretrained GloVe Embeddings (Optional)

```bash
# Download: https://nlp.stanford.edu/projects/glove/ (glove.twitter.27B.zip)
# Extract 300d file to: data/raw/glove.twitter.27B.300d.txt
```

Then in `configs/bilstm.yaml`:
```yaml
model:
  pretrained_embeddings: data/raw/glove.twitter.27B.300d.txt
  freeze_embeddings: false
```

## Reproducing the Experiment

```bash
# Full reproducible run:
python scripts/preprocess.py
python scripts/train_bilstm.py
python scripts/eval_bilstm.py
```

All random seeds are pinned to 42. The checkpoint and metrics should be identical across reruns on the same machine with the same environment.

