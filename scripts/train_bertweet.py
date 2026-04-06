#!/usr/bin/env python
"""
Entry-point script: fine-tune BERTweet / Twitter-RoBERTa on preprocessed data.

Usage:
    python scripts/train_bertweet.py
    python scripts/train_bertweet.py --config configs/bertweet.yaml
    python scripts/train_bertweet.py --config configs/bertweet.yaml --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.bertweet_dataset import build_transformer_loaders
from src.data.preprocess import ID_TO_LABEL
from src.models.bertweet import BERTweetClassifier
from src.training.trainer import EarlyStopping, set_seed
from src.utils.metrics import compute_metrics, save_confusion_matrix_plot, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing & config loading
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERTweet sentiment classifier"
    )
    parser.add_argument("--config", default="configs/bertweet.yaml")
    parser.add_argument(
        "--device",
        default=None,
        help="'cpu', 'cuda', or 'cuda:0'.  Auto-detected when omitted.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a .pt checkpoint to resume training from. "
        "Model weights are loaded; optimizer/scheduler start fresh for remaining epochs.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------


def _train_epoch(
    model: BERTweetClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    accumulation_steps: int,
    scaler,  # torch.cuda.amp.GradScaler or None
) -> tuple[float, float]:
    """One training epoch with optional gradient accumulation and AMP."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False), start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()

        preds = logits.detach().argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * accumulation_steps * labels.size(0)

        if step % accumulation_steps == 0 or step == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(
    model: BERTweetClassifier,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list, list]:
    """One evaluation epoch. Returns (loss, accuracy, y_true, y_pred)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="  eval ", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return total_loss / total, correct / total, y_true, y_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
            logger.error(
                "Required file not found: %s\nRun scripts/preprocess.py first.", p
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
    # Build DataLoaders
    # ------------------------------------------------------------------
    train_loader, val_loader, _ = build_transformer_loaders(
        train_path=data_cfg["train_path"],
        val_path=data_cfg["val_path"],
        test_path=data_cfg["test_path"],
        model_name=model_cfg["pretrained_name"],
        max_len=data_cfg["max_seq_len"],
        batch_size=train_cfg["batch_size"],
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Class weights for imbalanced data
    # ------------------------------------------------------------------
    num_classes = model_cfg["num_classes"]
    class_weights = None
    if train_cfg.get("class_weighted_loss", True):
        import pandas as pd, numpy as np

        df_train = pd.read_csv(data_cfg["train_path"])
        counts = np.zeros(num_classes)
        for lbl, cnt in df_train["label_id"].value_counts().items():
            counts[int(lbl)] = cnt
        counts = np.where(
            counts == 0, 1, counts
        )  # avoid div-by-zero for absent classes
        weights = 1.0 / counts
        weights /= weights.sum()
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        logger.info("Class weights: %s", class_weights.cpu().numpy().round(4).tolist())

    # ------------------------------------------------------------------
    # Build model  (optionally resume from checkpoint)
    # ------------------------------------------------------------------
    resume_epoch = 0
    resume_best_metric = float("-inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error("Resume checkpoint not found: %s", resume_path)
            sys.exit(1)
        ckpt_meta = torch.load(resume_path, map_location="cpu", weights_only=False)
        resume_epoch = ckpt_meta.get("epoch", 0)
        resume_best_metric = ckpt_meta.get("best_metric", float("-inf"))
        logger.info(
            "Resuming from checkpoint %s  (epoch=%d, best_metric=%.4f)",
            resume_path,
            resume_epoch,
            resume_best_metric,
        )
        model = BERTweetClassifier(
            model_name=model_cfg["pretrained_name"],
            num_classes=num_classes,
            dropout=model_cfg["dropout"],
            freeze_base=model_cfg.get("freeze_base", False),
        )
        model.load_state_dict(ckpt_meta["model_state_dict"])
    else:
        model = BERTweetClassifier(
            model_name=model_cfg["pretrained_name"],
            num_classes=num_classes,
            dropout=model_cfg["dropout"],
            freeze_base=model_cfg.get("freeze_base", False),
        )

    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s trainable / %s total", f"{trainable:,}", f"{total:,}")

    # ------------------------------------------------------------------
    # Optimizer, scheduler, criterion
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    remaining_epochs = train_cfg["epochs"] - resume_epoch
    if remaining_epochs <= 0:
        logger.info(
            "Checkpoint already at epoch %d >= target %d. Nothing to train.",
            resume_epoch,
            train_cfg["epochs"],
        )
        sys.exit(0)

    total_steps = (
        len(train_loader) // train_cfg.get("gradient_accumulation_steps", 1)
    ) * remaining_epochs
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.06))

    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed-precision scaler (only on CUDA)
    scaler = None
    if train_cfg.get("fp16", False) and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed-precision training enabled (fp16).")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    early_stopping = EarlyStopping(
        patience=train_cfg["early_stopping_patience"],
        mode="max" if train_cfg["early_stopping_metric"] == "macro_f1" else "min",
    )

    best_metric = resume_best_metric
    checkpoint_path = artifacts_dir / out_cfg["checkpoint_name"]
    history = []

    logger.info(
        "Starting training: epochs %d → %d  (remaining: %d).",
        resume_epoch + 1,
        train_cfg["epochs"],
        remaining_epochs,
    )
    t0 = time.time()

    for epoch in range(resume_epoch + 1, train_cfg["epochs"] + 1):
        train_loss, train_acc = _train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip=train_cfg.get("gradient_clip", 1.0),
            accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
            scaler=scaler,
        )
        scheduler.step()

        val_loss, val_acc, y_true, y_pred = _eval_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        val_metrics = compute_metrics(
            y_true,
            y_pred,
            label_names=ID_TO_LABEL,
            model_name="bertweet",
            split="val",
        )
        val_macro_f1 = val_metrics["macro_f1"]
        monitor = (
            val_macro_f1
            if train_cfg["early_stopping_metric"] == "macro_f1"
            else -val_loss
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_macro_f1": round(val_macro_f1, 4),
            }
        )

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f macro_f1=%.4f",
            epoch,
            train_cfg["epochs"],
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_macro_f1,
        )

        if monitor > best_metric:
            best_metric = monitor
            model.save_checkpoint(checkpoint_path, epoch=epoch, best_metric=best_metric)

        if early_stopping(monitor, epoch):
            logger.info(
                "Early stopping triggered at epoch %d (best epoch %d).",
                epoch,
                early_stopping.best_epoch,
            )
            break

    elapsed = time.time() - t0
    logger.info("Training finished in %.1f s.  Best metric: %.4f", elapsed, best_metric)

    # ------------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------------
    history_path = artifacts_dir / out_cfg["history_name"]
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s.", history_path)

    # ------------------------------------------------------------------
    # Auto-evaluate on test split
    # ------------------------------------------------------------------
    logger.info("Running test-set evaluation ...")
    from src.data.bertweet_dataset import build_transformer_loaders as _btl

    _, _, test_loader = _btl(
        train_path=data_cfg["train_path"],
        val_path=data_cfg["val_path"],
        test_path=data_cfg["test_path"],
        model_name=model_cfg["pretrained_name"],
        max_len=data_cfg["max_seq_len"],
        batch_size=train_cfg["batch_size"],
        seed=seed,
    )

    best_model = BERTweetClassifier.from_checkpoint(checkpoint_path, device=device)
    best_model.eval()
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  test", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = best_model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            y_true_test.extend(labels.cpu().tolist())
            y_pred_test.extend(preds.cpu().tolist())

    test_metrics = compute_metrics(
        y_true_test,
        y_pred_test,
        label_names=ID_TO_LABEL,
        model_name="bertweet",
        split="test",
    )
    metrics_path = artifacts_dir / out_cfg["metrics_name"]
    save_metrics(test_metrics, metrics_path)

    from sklearn.metrics import confusion_matrix as _cm

    cm_path = artifacts_dir / out_cfg["confusion_matrix_name"]
    conf_matrix = _cm(
        y_true_test, y_pred_test, labels=list(range(num_classes))
    ).tolist()
    save_confusion_matrix_plot(
        conf_matrix,
        ID_TO_LABEL,
        cm_path,
        title="BERTweet - Test Confusion Matrix",
    )

    logger.info(
        "Test results | accuracy=%.4f  macro_f1=%.4f  weighted_f1=%.4f",
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
        test_metrics["weighted_f1"],
    )
    logger.info("Metrics saved to %s.", metrics_path)
    logger.info("Confusion matrix saved to %s.", cm_path)


if __name__ == "__main__":
    main()
