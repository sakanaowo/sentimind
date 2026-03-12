---
phase: design
title: Design — Feature: BiLSTM Baseline (M3)
description: Architecture, model design, and training flow for the RNN baseline
status: done
---

# Design — BiLSTM Baseline (M3)

## Architecture Overview

```mermaid
graph TD
    CSV["data/processed/{train,val,test}.csv"] --> VOCAB["Vocabulary (word2idx)"]
    VOCAB --> DS["SentimentDataset (PyTorch)"]
    DS --> LOADER["DataLoader (batch_size=64)"]
    LOADER --> MODEL

    subgraph MODEL ["BiLSTMClassifier"]
        EMB["Embedding (vocab_size, 300)"] --> DROP1["EmbeddingDropout"]
        DROP1 --> LSTM["BiLSTM (2 layers, hidden=256)"]
        LSTM --> POOL["Mean Pooling over sequence"]
        POOL --> DROP2["Dropout(0.3)"]
        DROP2 --> FC["Linear(512, num_classes)"]
        FC --> LOGITS["Logits"]
    end

    LOGITS --> LOSS["CrossEntropyLoss (class-weighted)"]
    LOSS --> OPT["Adam optimizer + grad clip"]
    OPT --> ES["EarlyStopping (macro F1, patience=5)"]
    ES --> CKPT["data/artifacts/bilstm_best.pt"]
    CKPT --> EVAL["eval_bilstm.py"]
    EVAL --> METRICS["bilstm_metrics.json + confusion_matrix.png"]
```

## Data Models

| Artifact | Shape / Type | Description |
|---|---|---|
| `vocab.json` | `{word: idx}` | Word-to-index mapping |
| Model input | `(batch, seq_len=128)` | Long tensor of token ids |
| Model output | `(batch, num_classes)` | Raw logits |
| Checkpoint | `.pt` file | `state_dict` of best epoch |
| Metrics | JSON per `data_contract.md §7` | Evaluation report |

## Component Breakdown

| Module | Responsibility |
|---|---|
| `src/models/bilstm.py` | `BiLSTMClassifier` with optional GloVe loader |
| `src/data/dataset.py` | `Vocabulary`, `SentimentDataset`, dataloaders factory |
| `src/training/trainer.py` | `train()` loop, `EarlyStopping`, `_train_epoch`, `_eval_epoch` |
| `src/utils/metrics.py` | `compute_metrics()`, `save_metrics()`, confusion matrix plot |
| `scripts/train_bilstm.py` | CLI: build vocab → build model → train → save checkpoint |
| `scripts/eval_bilstm.py` | CLI: load checkpoint → evaluate → save metrics + plot |
| `configs/bilstm.yaml` | All hyperparameters |

## Design Decisions

1. **Mean-pooling over LSTM outputs** — more robust than last-hidden on variable-length text.
2. **Class-weighted loss (default on)** — addresses label imbalance without augmentation.
3. **Early stopping on macro F1** — avoids optimising for majority class.
4. **Gradient clipping (norm=5)** — prevents exploding gradients in deep LSTM.
5. **Vocabulary saved separately** — enables consistent encoding between train and inference.
6. **Optional GloVe embeddings** — pluggable; random init is the default for fast baseline.

## Non-Functional Requirements

- Target accuracy: 75–80% on test split (subject to data quality).
- Training time: < 2 hours on CPU with early stopping.
- Fully reproducible from documented commands with seed=42.
- No external API dependency (self-contained).

