#!/usr/bin/env python
"""
Semantic analysis suite: STS scoring, UMAP/HDBSCAN clustering, cross-model comparison.

Usage:
    python scripts/run_semantic_analysis.py
    python scripts/run_semantic_analysis.py --config configs/semantic.yaml
    python scripts/run_semantic_analysis.py --skip-sts     # skip STS (faster)
    python scripts/run_semantic_analysis.py --skip-cluster # skip clustering
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic analysis pipeline")
    parser.add_argument("--config", default="configs/semantic.yaml")
    parser.add_argument(
        "--skip-sts", action="store_true", help="Skip the STS scoring step."
    )
    parser.add_argument(
        "--skip-cluster",
        action="store_true",
        help="Skip the UMAP + HDBSCAN clustering step.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the cross-model comparison step.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def generate_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    device: Optional[str] = None,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """Generate sentence embeddings, using a cached file if available."""
    if cache_path and cache_path.exists():
        logger.info("Loading cached embeddings from %s.", cache_path)
        return np.load(cache_path)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from exc

    logger.info("Loading sentence-transformer: %s", model_name)
    model = SentenceTransformer(model_name, device=device)
    logger.info("Encoding %d texts (batch_size=%d)...", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info("Embeddings cached to %s.", cache_path)

    return embeddings


# ---------------------------------------------------------------------------
# STS scoring
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalised 1-D vectors."""
    return float(np.dot(a, b))


def run_sts(
    embeddings: np.ndarray,
    labels: List[int],
    label_map: Dict[int, str],
    pairs_per_class: int = 50,
    seed: int = 42,
) -> dict:
    """Compute average within-class and cross-class STS scores.

    Returns a dict suitable for JSON export.
    """
    rng = random.Random(seed)

    # Group indices by label
    label_to_indices: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    within_class_scores: Dict[str, float] = {}
    for lbl, indices in sorted(label_to_indices.items()):
        if len(indices) < 2:
            continue
        pairs = [
            (rng.choice(indices), rng.choice(indices)) for _ in range(pairs_per_class)
        ]
        scores = [
            cosine_similarity(embeddings[i], embeddings[j]) for i, j in pairs if i != j
        ]
        if scores:
            class_name = label_map.get(lbl, str(lbl))
            within_class_scores[class_name] = round(float(np.mean(scores)), 4)

    # Cross-class sampling — pick two different classes
    class_ids = sorted(label_to_indices.keys())
    cross_class_scores: List[float] = []
    for _ in range(pairs_per_class * len(class_ids)):
        if len(class_ids) < 2:
            break
        c1, c2 = rng.sample(class_ids, 2)
        i = rng.choice(label_to_indices[c1])
        j = rng.choice(label_to_indices[c2])
        cross_class_scores.append(cosine_similarity(embeddings[i], embeddings[j]))

    sts_report = {
        "within_class_avg_cosine": within_class_scores,
        "cross_class_avg_cosine": (
            round(float(np.mean(cross_class_scores)), 4) if cross_class_scores else None
        ),
        "interpretation": (
            "Higher within-class scores indicate semantically cohesive groups. "
            "A larger within/cross-class gap suggests cleaner semantic boundaries."
        ),
    }
    return sts_report


# ---------------------------------------------------------------------------
# UMAP + HDBSCAN clustering
# ---------------------------------------------------------------------------


def run_clustering(
    embeddings: np.ndarray,
    labels: List[int],
    label_map: Dict[int, str],
    umap_cfg: dict,
    hdbscan_cfg: dict,
    artifacts_dir: Path,
    plot_name: str,
    embeddings_2d_name: str,
    seed: int = 42,
) -> dict:
    """Reduce embeddings to 2-D with UMAP, cluster with HDBSCAN, and save a plot."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for clustering. Install with: pip install umap-learn"
        ) from exc

    try:
        import hdbscan as hdbscan_lib
    except ImportError as exc:
        raise ImportError(
            "hdbscan is required for clustering. Install with: pip install hdbscan"
        ) from exc

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    logger.info("Running UMAP dimensionality reduction...")
    umap_model = umap.UMAP(
        n_components=umap_cfg.get("n_components", 2),
        n_neighbors=umap_cfg.get("n_neighbors", 15),
        min_dist=umap_cfg.get("min_dist", 0.1),
        metric=umap_cfg.get("metric", "cosine"),
        random_state=umap_cfg.get("random_state", seed),
    )
    embeddings_2d = umap_model.fit_transform(embeddings)

    # Cache 2-D embeddings
    emb_2d_path = artifacts_dir / embeddings_2d_name
    np.save(emb_2d_path, embeddings_2d)
    logger.info("2-D embeddings saved to %s.", emb_2d_path)

    logger.info("Running HDBSCAN clustering...")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=hdbscan_cfg.get("min_cluster_size", 30),
        min_samples=hdbscan_cfg.get("min_samples", 5),
        metric=hdbscan_cfg.get("metric", "euclidean"),
        cluster_selection_method=hdbscan_cfg.get("cluster_selection_method", "eom"),
    )
    cluster_ids = clusterer.fit_predict(embeddings_2d)

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    noise_ratio = float(np.mean(cluster_ids == -1))
    logger.info("HDBSCAN: %d clusters found, noise_ratio=%.3f", n_clusters, noise_ratio)

    # ------------------------------------------------------------------
    # Plot: colour by true ground-truth label
    # ------------------------------------------------------------------
    unique_labels = sorted(set(labels))
    colours = cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_colour = {lbl: colours[i] for i, lbl in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: coloured by true label
    for lbl in unique_labels:
        mask = np.array(labels) == lbl
        axes[0].scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[label_colour[lbl]],
            label=label_map.get(lbl, str(lbl)),
            alpha=0.4,
            s=5,
        )
    axes[0].set_title("UMAP — Ground-truth Labels")
    axes[0].legend(markerscale=3, fontsize=7, loc="best")
    axes[0].axis("off")

    # Right: coloured by HDBSCAN cluster
    unique_clusters = sorted(set(cluster_ids))
    cmap_cluster = cm.tab20(np.linspace(0, 1, max(len(unique_clusters), 1)))
    for i, cid in enumerate(unique_clusters):
        mask = cluster_ids == cid
        colour = "lightgrey" if cid == -1 else cmap_cluster[i]
        label_str = "noise" if cid == -1 else f"cluster {cid}"
        axes[1].scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colour],
            label=label_str,
            alpha=0.4,
            s=5,
        )
    axes[1].set_title(f"UMAP — HDBSCAN Clusters ({n_clusters} clusters)")
    axes[1].legend(markerscale=3, fontsize=7, loc="best", ncol=2)
    axes[1].axis("off")

    plt.tight_layout()
    plot_path = artifacts_dir / plot_name
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Cluster plot saved to %s.", plot_path)

    return {
        "n_clusters": n_clusters,
        "noise_ratio": round(noise_ratio, 4),
        "umap_config": umap_cfg,
        "hdbscan_config": hdbscan_cfg,
    }


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------


def run_comparison(comparison_cfg: dict, artifacts_dir: Path, out_name: str) -> dict:
    """Load per-model metrics JSONs and build a unified comparison table."""
    report: dict = {"models": {}}

    metrics_keys = ["accuracy", "macro_f1", "weighted_f1"]

    for model_key, metrics_path in comparison_cfg.items():
        if not metrics_path:
            continue
        p = Path(metrics_path)
        if not p.exists():
            logger.warning("Metrics file not found for %s: %s. Skipping.", model_key, p)
            continue
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        report["models"][model_key] = {k: m.get(k) for k in metrics_keys}
        report["models"][model_key]["per_class"] = m.get("per_class", {})

    # Simple ranking by macro_f1
    ranked = sorted(
        [(k, v.get("macro_f1") or 0.0) for k, v in report["models"].items()],
        key=lambda x: x[1],
        reverse=True,
    )
    report["ranking_by_macro_f1"] = [k for k, _ in ranked]

    out_path = artifacts_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Comparison report saved to %s.", out_path)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    test_path = Path(data_cfg["test_path"])

    if not test_path.exists():
        logger.error(
            "Test file not found: %s\nRun scripts/preprocess.py first.", test_path
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(test_path)
    sample_size = data_cfg.get("sample_size")
    if sample_size and sample_size < len(df):
        df = df.sample(
            n=sample_size, random_state=data_cfg.get("sample_seed", 42)
        ).reset_index(drop=True)
        logger.info("Sampled %d rows for analysis.", len(df))

    texts = df[data_cfg["text_col"]].astype(str).tolist()
    labels = df[data_cfg["label_col"]].astype(int).tolist()

    # Build label_map from preprocessing config or fall back to a simple range
    try:
        import yaml

        with open("configs/preprocessing.yaml") as f:
            pre_cfg = yaml.safe_load(f)
        raw_map = pre_cfg.get("label_map", {})
        # Invert: id → canonical name (first match wins)
        id_to_label: Dict[int, str] = {}
        for name, lid in raw_map.items():
            if lid not in id_to_label:
                id_to_label[lid] = name.title()
    except Exception:
        id_to_label = {i: str(i) for i in range(7)}

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    emb_cfg = cfg["embeddings"]
    cache_path = Path(
        emb_cfg.get("cache_path", "data/artifacts/semantic_embeddings.npy")
    )
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path

    embeddings = generate_embeddings(
        texts=texts,
        model_name=emb_cfg["model_name"],
        batch_size=emb_cfg.get("batch_size", 64),
        device=emb_cfg.get("device"),
        cache_path=cache_path,
    )
    logger.info("Embeddings shape: %s", embeddings.shape)

    # ------------------------------------------------------------------
    # STS scoring
    # ------------------------------------------------------------------
    sts_report = {}
    if not args.skip_sts:
        logger.info("Running STS scoring...")
        sts_cfg = cfg.get("sts", {})
        sts_report = run_sts(
            embeddings=embeddings,
            labels=labels,
            label_map=id_to_label,
            pairs_per_class=sts_cfg.get("pairs_per_class", 50),
            seed=sts_cfg.get("seed", cfg["seed"]),
        )
        sts_path = artifacts_dir / cfg["output"]["sts_report_name"]
        with open(sts_path, "w", encoding="utf-8") as f:
            json.dump(sts_report, f, indent=2)
        logger.info("STS report saved to %s.", sts_path)
        logger.info("Within-class STS: %s", sts_report.get("within_class_avg_cosine"))
        logger.info(
            "Cross-class STS: %.4f", sts_report.get("cross_class_avg_cosine") or 0.0
        )
    else:
        logger.info("STS step skipped (--skip-sts).")

    # ------------------------------------------------------------------
    # UMAP + HDBSCAN clustering
    # ------------------------------------------------------------------
    cluster_summary = {}
    if not args.skip_cluster:
        logger.info("Running clustering...")
        cluster_summary = run_clustering(
            embeddings=embeddings,
            labels=labels,
            label_map=id_to_label,
            umap_cfg=cfg["clustering"]["umap"],
            hdbscan_cfg=cfg["clustering"]["hdbscan"],
            artifacts_dir=artifacts_dir,
            plot_name=cfg["output"]["cluster_plot_name"],
            embeddings_2d_name=cfg["output"]["embeddings_2d_name"],
            seed=cfg["seed"],
        )
    else:
        logger.info("Clustering step skipped (--skip-cluster).")

    # ------------------------------------------------------------------
    # Cross-model comparison
    # ------------------------------------------------------------------
    if not args.skip_comparison:
        comparison_cfg = cfg.get("comparison", {})
        if comparison_cfg:
            run_comparison(
                comparison_cfg=comparison_cfg,
                artifacts_dir=artifacts_dir,
                out_name=cfg["output"]["comparison_report_name"],
            )
        else:
            logger.info("No comparison config found; skipping comparison step.")
    else:
        logger.info("Comparison step skipped (--skip-comparison).")

    logger.info("Semantic analysis complete.  Artifacts in %s.", artifacts_dir)


if __name__ == "__main__":
    main()
