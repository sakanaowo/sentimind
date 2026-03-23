#!/usr/bin/env python
"""
Entry-point script: evaluate the best BiLSTM checkpoint on the test split.

Usage:
    python scripts/eval_bilstm.py
    python scripts/eval_bilstm.py --config configs/bilstm.yaml
    python scripts/eval_bilstm.py --config configs/bilstm.yaml --split val
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import SentimentDataset, Vocabulary
from src.models.bilstm import BiLSTMClassifier
from src.training.trainer import set_seed, _eval_epoch
from src.utils.metrics import (
    compute_metrics,
    save_confusion_matrix_plot,
    save_metrics,
)
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM on test / val split")
    parser.add_argument("--config", default="configs/bilstm.yaml")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]
    artifacts_dir = Path(out_cfg["artifacts_dir"])

    # ------------------------------------------------------------------
    # Load vocabulary
    # ------------------------------------------------------------------
    vocab_path = Path(data_cfg["vocab_path"])
    if not vocab_path.exists():
        logger.error("Vocabulary not found at %s. Run train_bilstm.py first.", vocab_path)
        sys.exit(1)
    vocab = Vocabulary.load(vocab_path)

    # ------------------------------------------------------------------
    # Load data split
    # ------------------------------------------------------------------
    split_key = f"{args.split}_path"
    split_path = Path(data_cfg[split_key])
    if not split_path.exists():
        logger.error("Split file not found: %s", split_path)
        sys.exit(1)

    df = pd.read_csv(split_path)
    dataset = SentimentDataset(
        df, vocab,
        max_len=data_cfg["max_seq_len"],
        text_col=data_cfg["text_col"],
        label_col=data_cfg["label_col"],
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Label names from preprocessing config
    # ------------------------------------------------------------------
    try:
        with open("configs/preprocessing.yaml") as f:
            pre_cfg = yaml.safe_load(f)
        label_map = pre_cfg["label_map"]
        num_classes = len(set(label_map.values()))
        id_to_label = {}
        for label_name, label_id in label_map.items():
            id_to_label.setdefault(label_id, label_name.title())
    except Exception:
        num_classes = 7
        id_to_label = None

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    checkpoint_path = artifacts_dir / out_cfg["checkpoint_name"]
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found at %s. Run train_bilstm.py first.", checkpoint_path)
        sys.exit(1)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=model_cfg["embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_classes=num_classes,
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        pad_idx=0,
        bidirectional=model_cfg["bidirectional"],
    )

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    logger.info("Checkpoint loaded from %s.", checkpoint_path)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    _, _, y_true, y_pred = _eval_epoch(model, loader, criterion, device)

    metrics = compute_metrics(
        y_true, y_pred,
        label_names=id_to_label,
        model_name="bilstm",
        split=args.split,
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    metrics_path = artifacts_dir / out_cfg["metrics_name"]
    save_metrics(metrics, metrics_path)

    cm_path = artifacts_dir / out_cfg["confusion_matrix_name"]
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        label_names=id_to_label,
        path=cm_path,
        title=f"BiLSTM — {args.split.capitalize()} Confusion Matrix",
    )

    # Print summary
    print("\n" + "=" * 55)
    print(f"  BiLSTM Evaluation Results ({args.split} split)")
    print("=" * 55)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print("=" * 55)
    for lbl, scores in metrics["per_class"].items():
        print(f"  {lbl:<12} P={scores['precision']:.3f} R={scores['recall']:.3f} "
              f"F1={scores['f1']:.3f}  n={scores['support']}")
    print("=" * 55 + "\n")

    logger.info("Evaluation complete. Metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
