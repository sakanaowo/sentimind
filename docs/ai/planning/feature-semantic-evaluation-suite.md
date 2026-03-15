---
phase: planning
title: Project Planning & Task Breakdown — Semantic Evaluation Suite (M6)
description: Break down work into actionable tasks and estimate timeline
---

# M6 — Semantic Evaluation Suite: Planning & Task Breakdown

## Milestones

- [x] M6.1: Config and STS scoring implemented
- [x] M6.2: UMAP + HDBSCAN clustering with plot output
- [x] M6.3: Cross-model comparison report + unit tests

## Task Breakdown

### Phase 1: Foundation

- [x] Task 1.1: Create `configs/semantic.yaml` — embedding model, STS params, UMAP/HDBSCAN params, comparison metrics paths, output paths.
- [x] Task 1.2: Embedding generation with `sentence-transformers`; deterministic `.npy` cache to avoid re-encoding on repeat runs.

### Phase 2: Core Features

- [x] Task 2.1: `run_sts()` — within-class and cross-class cosine similarity scoring; outputs `sts_report.json`.
- [x] Task 2.2: `run_clustering()` — UMAP 2-D reduction → HDBSCAN; saves `semantic_cluster_plot.png` (2-panel: ground-truth labels + discovered clusters) and `semantic_embeddings_2d.npy`.
- [x] Task 2.3: `run_comparison()` — loads `bilstm_metrics.json`, `bertweet_metrics.json`, `llm_metrics.json`; produces unified `comparison_report.json` ranked by macro_f1.

### Phase 3: Integration & Polish

- [x] Task 3.1: CLI flags `--skip-sts`, `--skip-cluster`, `--skip-comparison` for incremental re-runs.
- [x] Task 3.2: Create `tests/test_semantic_analysis.py` — no real embedding model: cosine similarity, STS with fake embeddings, comparison report building, missing-file handling.

## Dependencies

- **Requires**: `data/processed/test.csv`, and optionally `bilstm_metrics.json` / `bertweet_metrics.json` / `llm_metrics.json` for comparison.
- **Provides**: `sts_report.json`, `semantic_cluster_plot.png`, `semantic_embeddings_2d.npy`, `comparison_report.json`.
- `sentence-transformers>=2.2.0` (already in `requirements.txt`), `umap-learn>=0.5.3`, `hdbscan>=0.8.33` added.

## Risks & Mitigation

| Risk                                     | Mitigation                                                 |
| ---------------------------------------- | ---------------------------------------------------------- |
| UMAP non-determinism                     | `random_state` pinned in config                            |
| HDBSCAN finds 0 clusters on small sample | `min_cluster_size` tunable; plot still shows noise as grey |
| Long embedding time on CPU               | `cache_path` saves `.npy` so only runs once                |

## Resources Needed

- `sentence-transformers` model auto-downloaded from HuggingFace Hub on first run
- ≥4 GB RAM for UMAP on full test set (~7 500 rows × 384-dim)
