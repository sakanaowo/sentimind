"""
Training and evaluation loop for BiLSTM (and compatible models).
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, value: float, epoch: int) -> bool:
        """Return True if training should stop."""
        if self.best is None:
            self.best = value
            self.best_epoch = epoch
            return False

        improved = (
            (value > self.best + self.min_delta) if self.mode == "max"
            else (value < self.best - self.min_delta)
        )

        if improved:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 5.0,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_x, batch_y in tqdm(loader, desc="  train", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)

    return total_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, list, list]:
    """Run one evaluation epoch. Returns (avg_loss, accuracy, y_true, y_pred)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    for batch_x, batch_y in tqdm(loader, desc="  eval ", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * len(batch_y)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)

        y_true.extend(batch_y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return total_loss / total, correct / total, y_true, y_pred


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    early_stopping_patience: int = 5,
    early_stopping_metric: str = "macro_f1",   # "macro_f1" | "val_loss"
    gradient_clip: float = 5.0,
    checkpoint_path: str | Path = "data/artifacts/bilstm_best.pt",
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict:
    """Full training loop with early stopping and checkpointing.

    Returns:
        history: dict with per-epoch metrics for analysis.
    """
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    es_mode = "max" if early_stopping_metric == "macro_f1" else "min"
    early_stopper = EarlyStopping(patience=early_stopping_patience, mode=es_mode)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: Dict = {"train_loss": [], "val_loss": [], "train_acc": [],
                     "val_acc": [], "val_macro_f1": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = _train_epoch(
            model, train_loader, optimizer, criterion, device, gradient_clip
        )

        from sklearn.metrics import f1_score
        val_loss, val_acc, y_true, y_pred = _eval_epoch(
            model, val_loader, criterion, device
        )
        val_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_macro_f1, 4))

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f "
            "macro_f1=%.4f | %.1fs",
            epoch, epochs, train_loss, train_acc,
            val_loss, val_acc, val_macro_f1, elapsed,
        )

        # Save best checkpoint
        monitor_val = val_macro_f1 if early_stopping_metric == "macro_f1" else val_loss
        if early_stopper.best is None or (
            (es_mode == "max" and monitor_val > early_stopper.best) or
            (es_mode == "min" and monitor_val < early_stopper.best)
        ):
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  New best checkpoint saved (epoch %d, %s=%.4f).",
                        epoch, early_stopping_metric, monitor_val)

        if early_stopper(monitor_val, epoch):
            logger.info(
                "Early stopping triggered at epoch %d (best at epoch %d).",
                epoch, early_stopper.best_epoch,
            )
            break

    logger.info("Training complete. Best epoch: %d | best %s: %.4f",
                early_stopper.best_epoch, early_stopping_metric, early_stopper.best)
    history["best_epoch"] = early_stopper.best_epoch
    return history
