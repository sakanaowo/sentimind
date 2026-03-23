#!/usr/bin/env python
"""
Entry-point script: evaluate a fine-tuned BERTweet model on the test split.

Usage:
    python scripts/eval_bertweet.py
    python scripts/eval_bertweet.py --config configs/bertweet.yaml
    python scripts/eval_bertweet.py --config configs/bertweet.yaml --device cuda
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.bertweet_dataset import build_transformer_loaders
from src.data.preprocess import ID_TO_LABEL
from src.models.bertweet import BERTweetClassifier
from src.utils.metrics import compute_metrics, save_confusion_matrix_plot, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BERTweet classifier on test split"
    )
    parser.add_argument("--config", default="configs/bertweet.yaml")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[list, list]:
    """Run inference on loader. Returns (y_true, y_pred)."""
    model.eval()
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="  eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]

    artifacts_dir = Path(out_cfg["artifacts_dir"])
    checkpoint_path = artifacts_dir / out_cfg["checkpoint_name"]

    # ------------------------------------------------------------------
    # Validate prerequisites
    # ------------------------------------------------------------------
    for split in ["train_path", "val_path", "test_path"]:
        p = Path(data_cfg[split])
        if not p.exists():
            logger.error(
                "Split file not found: %s\nRun scripts/preprocess.py first.", p
            )
            sys.exit(1)

    if not checkpoint_path.exists():
        logger.error(
            "Checkpoint not found: %s\nRun scripts/train_bertweet.py first.",
            checkpoint_path,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    _, _, test_loader = build_transformer_loaders(
        train_path=data_cfg["train_path"],
        val_path=data_cfg["val_path"],
        test_path=data_cfg["test_path"],
        model_name=model_cfg["pretrained_name"],
        max_len=data_cfg["max_seq_len"],
        batch_size=model_cfg.get("eval_batch_size", cfg["training"]["batch_size"]),
        seed=cfg["seed"],
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = BERTweetClassifier.from_checkpoint(checkpoint_path, device=device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    logger.info("Running evaluation on test split …")
    y_true, y_pred = evaluate(model, test_loader, device)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(
        y_true,
        y_pred,
        label_names=ID_TO_LABEL,
        model_name="bertweet",
        split="test",
    )

    logger.info(
        "Test results — accuracy=%.4f  macro_f1=%.4f  weighted_f1=%.4f",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
    )

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    metrics_path = artifacts_dir / out_cfg["metrics_name"]
    save_metrics(metrics, metrics_path)

    cm_path = artifacts_dir / out_cfg["confusion_matrix_name"]
    label_names = [
        ID_TO_LABEL.get(i, str(i)) for i in sorted(set(y_true) | set(y_pred))
    ]
    save_confusion_matrix_plot(
        y_true,
        y_pred,
        label_names=label_names,
        title="BERTweet — Test Confusion Matrix",
        save_path=cm_path,
    )

    logger.info("All artifacts written to %s.", artifacts_dir)


if __name__ == "__main__":
    main()
