---
phase: planning
title: Planning — Feature: Dataset Preprocessing (M2)
description: Task breakdown for cleaning, splitting, and validating the mental-health dataset
status: done
---

# Planning — Dataset Preprocessing (M2)

## Milestones

- [x] M2.1: Data contract locked (`data_contract.md`)
- [x] M2.2: Preprocessing pipeline implemented (`src/data/preprocess.py`)
- [x] M2.3: Dataset loader implemented (`src/data/dataset.py`)
- [x] M2.4: Entry-point script complete (`scripts/preprocess.py`)
- [x] M2.5: Unit tests pass (`tests/test_preprocessing.py`)

## Task Breakdown

### Phase 1: Contract & Structure
- [x] Task 1.1: Define data contract — columns, label map, split ratios, artifact paths (`data_contract.md`)
- [x] Task 1.2: Create `configs/preprocessing.yaml` with all tunable parameters
- [x] Task 1.3: Scaffold `src/data/`, `data/raw/`, `data/processed/`, `data/artifacts/`

### Phase 2: Core Pipeline
- [x] Task 2.1: Implement `clean_text()` — URL/mention/hashtag/HTML/whitespace/lowercase rules
- [x] Task 2.2: Implement `normalise_label()` — case-insensitive label lookup
- [x] Task 2.3: Implement `preprocess_dataframe()` — null-drop, short-text-drop, dedup, quality report
- [x] Task 2.4: Implement stratified 70/15/15 split (seed 42) in `scripts/preprocess.py`
- [x] Task 2.5: Save `train.csv`, `val.csv`, `test.csv` and `preprocessing_report.json`

### Phase 3: Validation & Tests
- [x] Task 3.1: Implement `validate_processed_csv()` schema check
- [x] Task 3.2: Write unit tests covering clean_text, normalise_label, preprocess_dataframe edge cases
- [x] Task 3.3: Verify determinism (re-run produces identical checksums)

## Dependencies

- Requires raw Kaggle Mental Health CSV at `data/raw/kaggle_mental_health.csv`
- No external API dependencies
- `M3 (BiLSTM)` and `M4 (BERTweet)` depend on outputs of this feature

## Timeline & Estimates

| Task group | Estimate |
|---|---|
| Contract + config | 1 h |
| Core pipeline | 3 h |
| Scripts + tests | 2 h |
| **Total** | **~1 day** |

## Risks & Mitigation

| Risk | Mitigation |
|---|---|
| Label inconsistency in Kaggle CSV | Case-insensitive normalisation + drop/log unknown labels |
| Severe class imbalance | Class distribution reported; class-weighted loss in downstream training |
| Short/empty texts | Hard minimum 3-char threshold; dropped rows logged |
| Non-reproducible splits | Fixed seed 42, stratified split |

## Resources Needed

- `pandas`, `scikit-learn`, `pyyaml`, `pytest`
- Kaggle dataset: `sentimentanalysisformentalhealth` by suchintikasarkat

