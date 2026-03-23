---
phase: implementation
title: Implementation Guide — Semantic Evaluation Suite (M6)
description: Technical implementation notes, patterns, and code guidelines
---

# M6 — Semantic Evaluation Suite: Implementation Guide

## Key Implementation Patterns

### Embedding Generation with Cache
```python
def generate_embeddings(texts, model_name, cache_path):
    cache = Path(cache_path)
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(cache, embs)
    return embs
```

### Lazy Optional Dependency Imports
```python
def run_clustering(...):
    try:
        import umap
        import hdbscan as hdbscan_lib
    except ImportError as e:
        raise ImportError(
            "umap-learn and hdbscan are required for clustering. "
            "Install with: pip install umap-learn hdbscan"
        ) from e
    ...
```

### STS Scoring Pattern
```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# Within-class mean similarity:
idxs = [i for i, lbl in enumerate(labels) if lbl == target_label]
pairs = [(i, j) for i in idxs for j in idxs if i < j]
score = np.mean([cosine_similarity(embs[i], embs[j]) for i, j in pairs])
```

### Comparison Report Building
```python
def run_comparison(metric_paths, out_path):
    rows = []
    for path in metric_paths:
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append({"model": m["model"], "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"]})
    rows.sort(key=lambda x: x["macro_f1"], reverse=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
```

## Testing Patterns

`tests/test_semantic_analysis.py` uses small synthetic numpy arrays — no real embedding model downloaded:
```python
import numpy as np

embs = np.random.rand(20, 8).astype(np.float32)  # 20 texts × 8-dim
labels = [i % 4 for i in range(20)]              # 4 fake classes
```

## Configuration Reference

```yaml
# configs/semantic.yaml (key fields)
model_name: sentence-transformers/all-MiniLM-L6-v2
cache_path: data/artifacts/semantic_embeddings_cache.npy
umap:
  n_components: 2
  n_neighbors: 15
  min_dist: 0.1
  metric: cosine
  random_state: 42
hdbscan:
  min_cluster_size: 30
  min_samples: 5
comparison:
  metric_paths:
    - data/artifacts/bilstm_metrics.json
    - data/artifacts/bertweet_metrics.json
    - data/artifacts/llm_metrics.json
```

## Running the Script

```bash
# Full run (STS + clustering + comparison):
python scripts/run_semantic_analysis.py --config configs/semantic.yaml

# Skip clustering (fast re-run if embeddings already cached):
python scripts/run_semantic_analysis.py --config configs/semantic.yaml --skip-cluster

# Comparison only (all 3 model jsons must exist):
python scripts/run_semantic_analysis.py --config configs/semantic.yaml \
    --skip-sts --skip-cluster
```

## Known Issues / Gotchas

- The `.npy` cache is tied to a specific `(model_name, split, max_len)`; delete it when any of these change.
- HDBSCAN with `min_cluster_size=30` may produce 0 discovered clusters on small samples (<300 rows). Reduce `min_cluster_size` in config if this occurs.
- UMAP is stochastic despite `random_state` when run with multiple threads; set `OMP_NUM_THREADS=1` for exact reproducibility.
- `run_comparison` skips missing metric JSONs with a printed warning; it will output an empty list if none of the 3 model files exist yet.
