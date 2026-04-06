#!/usr/bin/env python
"""
Entry-point script: preprocess raw dataset and produce train/val/test splits.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --config configs/preprocessing.yaml
    python scripts/preprocess.py --raw_path data/raw/kaggle_mental_health.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import preprocess_dataframe, validate_processed_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentimind preprocessing pipeline")
    parser.add_argument("--config", default="configs/preprocessing.yaml",
                        help="Path to preprocessing config YAML")
    parser.add_argument("--raw_path", default=None,
                        help="Override raw CSV path from config")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    raw_path = args.raw_path or cfg["input"]["raw_path"]
    text_col = cfg["input"]["text_col"]
    label_col = cfg["input"]["label_col"]
    label_map = cfg["label_map"]
    processed_dir = Path(cfg["output"]["processed_dir"])
    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    seed = cfg["seed"]
    min_text_length = cfg["cleaning"]["min_text_length"]

    train_ratio = cfg["split"]["train"]
    val_ratio = cfg["split"]["val"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    raw_path_p = Path(raw_path)
    if not raw_path_p.exists():
        logger.error(
            "Raw data file not found: %s\n"
            "Download the Kaggle Mental Health dataset and place it at that path.",
            raw_path,
        )
        sys.exit(1)

    logger.info("Loading raw data from %s …", raw_path)
    df_raw = pd.read_csv(raw_path)
    logger.info("Raw shape: %s", df_raw.shape)

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    df_clean, report = preprocess_dataframe(
        df_raw,
        text_col=text_col,
        label_col=label_col,
        label_map=label_map,
        min_text_length=min_text_length,
    )

    # ------------------------------------------------------------------
    # Stratified split: (train+val) vs test, then train vs val
    # ------------------------------------------------------------------
    test_ratio = 1.0 - train_ratio - val_ratio
    val_ratio_of_train_val = val_ratio / (train_ratio + val_ratio)

    df_trainval, df_test = train_test_split(
        df_clean,
        test_size=test_ratio,
        stratify=df_clean["label_id"],
        random_state=seed,
    )
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio_of_train_val,
        stratify=df_trainval["label_id"],
        random_state=seed,
    )

    # Reset indices
    for df in [df_train, df_val, df_test]:
        df.reset_index(drop=True, inplace=True)

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(df_train), len(df_val), len(df_test),
    )

    # ------------------------------------------------------------------
    # Save splits
    # ------------------------------------------------------------------
    df_train.to_csv(processed_dir / "train.csv", index=False)
    df_val.to_csv(processed_dir / "val.csv", index=False)
    df_test.to_csv(processed_dir / "test.csv", index=False)
    logger.info("Splits saved to %s.", processed_dir)

    # ------------------------------------------------------------------
    # Save quality report
    # ------------------------------------------------------------------
    report["splits"] = {
        "train": len(df_train),
        "val": len(df_val),
        "test": len(df_test),
    }
    report_path = artifacts_dir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Quality report saved to %s.", report_path)

    # ------------------------------------------------------------------
    # Schema validation smoke-check
    # ------------------------------------------------------------------
    all_valid = True
    for split_name in ["train", "val", "test"]:
        ok = validate_processed_csv(processed_dir / f"{split_name}.csv")
        if not ok:
            logger.error("Schema validation FAILED for %s split.", split_name)
            all_valid = False

    if all_valid:
        logger.info("All split files pass schema validation.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
