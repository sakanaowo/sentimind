"""
Evaluation metric utilities shared across all models (metric contract).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[Dict[int, str]] = None,
    model_name: str = "model",
    split: str = "test",
) -> Dict:
    """Compute the standard metric contract (see data_contract.md §7).

    Returns a dict that is both human-readable and machine-parseable.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class from classification_report
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [label_names.get(i, str(i)) for i in labels] if label_names else None
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    per_class: Dict = {}
    for key, val in report.items():
        if key in ("accuracy", "macro avg", "weighted avg"):
            continue
        per_class[key] = {
            "precision": round(float(val["precision"]), 4),
            "recall": round(float(val["recall"]), 4),
            "f1": round(float(val["f1-score"]), 4),
            "support": int(val["support"]),
        }

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    result = {
        "model": model_name,
        "split": split,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": conf_matrix,
    }
    return result


def save_metrics(metrics: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s.", path)


def save_confusion_matrix_plot(
    conf_matrix: List[List[int]],
    label_names: Optional[Dict[int, str]],
    path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a labelled confusion matrix PNG."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available — skipping confusion matrix plot.")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(conf_matrix)
    labels_list = [label_names.get(i, str(i)) for i in range(n)] if label_names else [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(
        np.array(conf_matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_list,
        yticklabels=labels_list,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s.", path)
