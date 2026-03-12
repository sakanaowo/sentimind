---
phase: planning
title: Planning — Feature: BiLSTM Baseline (M3)
description: Task breakdown for RNN baseline classifier implementation
status: done
---

# Planning — BiLSTM Baseline (M3)

## Milestones

- [x] M3.1: Model architecture implemented (`src/models/bilstm.py`)
- [x] M3.2: Training loop with early stopping (`src/training/trainer.py`)
- [x] M3.3: Metric utilities following contract (`src/utils/metrics.py`)
- [x] M3.4: Train entry-point script (`scripts/train_bilstm.py`)
- [x] M3.5: Eval entry-point script (`scripts/eval_bilstm.py`)
- [x] M3.6: Unit tests pass (`tests/test_bilstm_model.py`)

## Task Breakdown

### Phase 1: Config & Architecture
- [x] Task 1.1: Create `configs/bilstm.yaml` with all hyperparameters
- [x] Task 1.2: Implement `BiLSTMClassifier` (Embedding → BiLSTM → Dropout → Dense → Softmax)
- [x] Task 1.3: Add GloVe pretrained embedding loader (optional)

### Phase 2: Training Infrastructure
- [x] Task 2.1: Implement `Vocabulary` with fit/encode/save/load
- [x] Task 2.2: Implement `SentimentDataset` (PyTorch Dataset with padding/truncation)
- [x] Task 2.3: Implement `compute_class_weights()` for inverse-frequency weighting
- [x] Task 2.4: Implement training loop with gradient clipping and early stopping
- [x] Task 2.5: Implement `train_bilstm.py` entry-point script

### Phase 3: Evaluation
- [x] Task 3.1: Implement `compute_metrics()` following metric contract (§7 of `data_contract.md`)
- [x] Task 3.2: Implement confusion matrix plot and saving
- [x] Task 3.3: Implement `eval_bilstm.py` entry-point script
- [x] Task 3.4: Write unit tests for model forward pass, vocabulary, early stopping, metrics

## Dependencies

- Requires M2 outputs: `data/processed/{train,val,test}.csv`
- PyTorch (CPU training supported, GPU optional)
- `scikit-learn` for metric computation

## Timeline & Estimates

| Task group | Estimate |
|---|---|
| Config + architecture | 2 h |
| Training infrastructure | 3 h |
| Evaluation + scripts | 2 h |
| Tests | 1 h |
| **Total** | **~1 day** |

## Risks & Mitigation

| Risk | Mitigation |
|---|---|
| OOV tokens degrade performance | UNK token + optional GloVe embeddings |
| Class imbalance skews metrics | Class-weighted cross-entropy loss (default on) |
| Overfitting on small dataset | Dropout, early stopping on macro F1 |
| Long training time | Early stopping patience=5, target <2h on CPU |

## Resources Needed

- `torch>=2.0`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `pyyaml`
- Preprocessed split CSV files from M2
- Optional: GloVe Twitter 27B 300d embeddings

