---
phase: testing
title: Testing — Feature: BiLSTM Baseline (M3)
description: Test coverage and smoke-check commands for the BiLSTM classifier
status: done
---

# Testing — BiLSTM Baseline (M3)

## Running Tests

```bash
# All BiLSTM tests
pytest tests/test_bilstm_model.py -v

# Full test suite
pytest tests/ -v

# With coverage
pytest tests/test_bilstm_model.py -v --cov=src --cov-report=term-missing
```

## Test Coverage

### `Vocabulary` (unit)
- [x] `fit()` creates more than 2 tokens
- [x] `<PAD>` at index 0, `<UNK>` at index 1
- [x] `encode()` outputs the correct padded length
- [x] `encode()` truncates long text
- [x] `encode()` pads short text with PAD token
- [x] OOV tokens map to UNK
- [x] `save()` + `load()` round-trip preserves all mappings

### `BiLSTMClassifier` (unit)
- [x] Forward pass produces shape `(batch, num_classes)`
- [x] Output is finite (no NaN/Inf)
- [x] PAD embedding is zero-initialised
- [x] All parameters are trainable
- [x] Works with batch sizes 1, 8, 16

### `SentimentDataset` (unit)
- [x] `__len__` returns correct count
- [x] `__getitem__` returns `(seq_tensor, label_tensor)` with correct shapes

### `EarlyStopping` (unit)
- [x] Stops after patience non-improvements
- [x] Resets counter on improvement
- [x] Min-mode (val_loss) works correctly

### `compute_metrics()` + `save_metrics()` (unit)
- [x] Perfect predictions → accuracy=1.0, macro_f1=1.0
- [x] Output dict contains all metric contract keys
- [x] Per-class dict contains `{precision, recall, f1, support}`
- [x] Confusion matrix has correct shape
- [x] Saved JSON loads correctly

## Smoke Test (no real data needed)

```python
python -c "
import torch, pandas as pd
from src.data.dataset import Vocabulary, SentimentDataset
from src.models.bilstm import BiLSTMClassifier
from torch.utils.data import DataLoader

texts = ['i feel depressed today', 'anxiety is overwhelming', 'had a good day']
labels = [1, 2, 0]
vocab = Vocabulary(min_freq=1).fit(texts)
ds = SentimentDataset(pd.DataFrame({'text': texts, 'label_id': labels}), vocab, max_len=10)
model = BiLSTMClassifier(len(vocab), 32, 16, num_classes=3, num_layers=1, dropout=0.0)
batch_x, batch_y = next(iter(DataLoader(ds, batch_size=3)))
logits = model(batch_x)
assert logits.shape == (3, 3)
print('Smoke test PASSED')
"
```

## Definition of Done (DoD)

- [x] All unit tests pass: `pytest tests/test_bilstm_model.py -v`
- [x] Smoke test runs without error (no real data required)
- [x] `train_bilstm.py` runs end-to-end on preprocessed data and saves checkpoint
- [x] `eval_bilstm.py` loads checkpoint and saves `bilstm_metrics.json` following contract §7
- [x] Confusion matrix PNG is generated
- [x] Re-run from same checkpoint produces identical metrics


- Unit test coverage target (default: 100% of new/changed code)
- Integration test scope (critical paths + error handling)
- End-to-end test scenarios (key user journeys)
- Alignment with requirements/design acceptance criteria

## Unit Tests
**What individual components need testing?**

### Component/Module 1
- [ ] Test case 1: [Description] (covers scenario / branch)
- [ ] Test case 2: [Description] (covers edge case / error handling)
- [ ] Additional coverage: [Description]

### Component/Module 2
- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]
- [ ] Additional coverage: [Description]

## Integration Tests
**How do we test component interactions?**

- [ ] Integration scenario 1
- [ ] Integration scenario 2
- [ ] API endpoint tests
- [ ] Integration scenario 3 (failure mode / rollback)

## End-to-End Tests
**What user flows need validation?**

- [ ] User flow 1: [Description]
- [ ] User flow 2: [Description]
- [ ] Critical path testing
- [ ] Regression of adjacent features

## Test Data
**What data do we use for testing?**

- Test fixtures and mocks
- Seed data requirements
- Test database setup

## Test Reporting & Coverage
**How do we verify and communicate test results?**

- Coverage commands and thresholds (`npm run test -- --coverage`)
- Coverage gaps (files/functions below 100% and rationale)
- Links to test reports or dashboards
- Manual testing outcomes and sign-off

## Manual Testing
**What requires human validation?**

- UI/UX testing checklist (include accessibility)
- Browser/device compatibility
- Smoke tests after deployment

## Performance Testing
**How do we validate performance?**

- Load testing scenarios
- Stress testing approach
- Performance benchmarks

## Bug Tracking
**How do we manage issues?**

- Issue tracking process
- Bug severity levels
- Regression testing strategy

