---
phase: testing
title: Testing — Feature: Dataset Preprocessing (M2)
description: Test coverage and smoke-check commands for the preprocessing pipeline
status: done
---

# Testing — Dataset Preprocessing (M2)

## Running Tests

```bash
# All preprocessing tests
pytest tests/test_preprocessing.py -v

# With coverage
pytest tests/test_preprocessing.py -v --cov=src/data --cov-report=term-missing
```

## Test Coverage

### `clean_text()` (unit)
- [x] URL removal
- [x] @mention removal
- [x] Hashtag word preservation (`#anxious` → `anxious`)
- [x] Lowercase conversion
- [x] HTML entity removal
- [x] Whitespace collapse
- [x] Non-string input (None, int) → empty string

### `normalise_label()` (unit)
- [x] Exact match (lowercase)
- [x] Case-insensitive match
- [x] Whitespace-trimming
- [x] Unknown label returns `None`

### `preprocess_dataframe()` (integration)
- [x] Output columns are `{text, label, label_id}`
- [x] Null rows are dropped and counted
- [x] Short-text rows are dropped and counted
- [x] Unknown labels are dropped and logged
- [x] Duplicate rows are deduplicated
- [x] `label_id` dtype is integer
- [x] `label_id` values are within the defined label map
- [x] No null values in output
- [x] Custom column names are accepted
- [x] Missing required column raises `ValueError`
- [x] **Determinism**: two runs on same input produce identical output

### `validate_processed_csv()` (schema check)
- [x] Valid file passes
- [x] Missing column fails
- [x] Non-existent file fails

## Smoke Test (End-to-End)

```bash
# Create a tiny sample CSV and run the full script
python -c "
import pandas as pd
pd.DataFrame({
    'text': ['i feel very depressed today']*50 + ['anxiety keeps me up']*50 + ['normal day overall']*50,
    'label': ['depression']*50 + ['anxiety']*50 + ['normal']*50
}).to_csv('/tmp/sample.csv', index=False)
"
python scripts/preprocess.py --raw_path /tmp/sample.csv
# Expected: 3 split CSVs + preprocessing_report.json created
```

## Definition of Done (DoD)

- [x] All unit tests pass: `pytest tests/test_preprocessing.py -v`
- [x] Smoke test runs without error on 150-row sample
- [x] `preprocessing_report.json` is generated and contains `total_dropped`, `class_counts`, `splits`
- [x] Re-running produces identical split checksums


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

