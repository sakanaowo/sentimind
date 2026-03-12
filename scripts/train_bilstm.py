#!/usr/bin/env python
"""
Entry-point script: train BiLSTM classifier on preprocessed data.

Usage:
    python scripts/train_bilstm.py
    python scripts/train_bilstm.py --config configs/bilstm.yaml
    python scripts/train_bilstm.py --config configs/bilstm.yaml --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_vocab_and_loaders, compute_class_weights
from src.models.bilstm import BiLSTMClassifier
from src.training.trainer import set_seed, train
from src.data.preprocess import ID_TO_LABEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiLSTM sentiment classifier")
    parser.add_argument("--config", default="configs/bilstm.yaml")
    parser.add_argument("--device", default=None,
                        help="'cpu', 'cuda', or 'cuda:0'. Auto-detected if omitted.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = cfg["seed"]
    set_seed(seed)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    artifacts_dir = Path(out_cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validate input files
    # ------------------------------------------------------------------
    for split in ["train_path", "val_path", "test_path"]:
        p = Path(data_cfg[split])
        if not p.exists():
            logger.error("Required file not found: %s\nRun scripts/preprocess.py first.", p)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Build vocabulary and dataloaders
    # ------------------------------------------------------------------
    vocab, train_loader, val_loader, test_loader = build_vocab_and_loaders(
        train_path=data_cfg["train_path"],
        val_path=data_cfg["val_path"],
        test_path=data_cfg["test_path"],
        vocab_path=data_cfg["vocab_path"],
        max_len=data_cfg["max_seq_len"],
        vocab_min_freq=data_cfg["vocab_min_freq"],
        batch_size=train_cfg["batch_size"],
        seed=seed,
    )

    num_classes = len(set(cfg.get("label_map", {}).values()) or range(7))
    # Infer from label_map or default to 7
    try:
        with open("configs/preprocessing.yaml") as f:
            pre_cfg = yaml.safe_load(f)
        num_classes = len(pre_cfg["label_map"])
    except Exception:
        num_classes = 7

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
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
    logger.info("Model architecture:\n%s", model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: {:,}".format(total_params))

    # Optional pretrained embeddings
    if model_cfg.get("pretrained_embeddings"):
        model.load_pretrained_embeddings(
            glove_path=model_cfg["pretrained_embeddings"],
            word2idx=vocab.word2idx,
            freeze=model_cfg.get("freeze_embeddings", False),
        )

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    class_weights = None
    if train_cfg.get("class_weighted_loss"):
        class_weights = compute_class_weights(
            data_cfg["train_path"], num_classes=num_classes
        )
        logger.info("Class weights: %s", class_weights.tolist())

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    checkpoint_path = artifacts_dir / out_cfg["checkpoint_name"]
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        class_weights=class_weights,
        early_stopping_patience=train_cfg["early_stopping_patience"],
        early_stopping_metric=train_cfg["early_stopping_metric"],
        gradient_clip=train_cfg["gradient_clip"],
        checkpoint_path=checkpoint_path,
        device=device,
        seed=seed,
    )

    # Save training history
    history_path = artifacts_dir / "bilstm_train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s.", history_path)
    logger.info("Best checkpoint at: %s", checkpoint_path)
    logger.info("Next step: python scripts/eval_bilstm.py --config %s", args.config)


if __name__ == "__main__":
    main()
