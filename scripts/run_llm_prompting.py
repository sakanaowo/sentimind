#!/usr/bin/env python
"""
Entry-point script: LLM zero-shot / few-shot classification of mental-health posts.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/run_llm_prompting.py
    python scripts/run_llm_prompting.py --config configs/llm_prompting.yaml --mode few_shot
    python scripts/run_llm_prompting.py --sample 50   # quick dev run on 50 samples
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.llm_client import (
    LLMClient,
    LLMPrediction,
    _FEW_SHOT_EXAMPLES,
)
from src.utils.metrics import compute_metrics, save_confusion_matrix_plot, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM prompting evaluation pipeline")
    parser.add_argument("--config", default="configs/llm_prompting.yaml")
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot"],
        default=None,
        help="Override the prompting mode from config.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Override the sample_size from config for a quick test run.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_test_data(cfg: dict, sample_override: int | None) -> pd.DataFrame:
    """Load and optionally sample the test split."""
    test_path = Path(cfg["data"]["test_path"])
    if not test_path.exists():
        logger.error(
            "Test file not found: %s\nRun scripts/preprocess.py first.", test_path
        )
        sys.exit(1)

    df = pd.read_csv(test_path)
    sample_size = sample_override or cfg["data"].get("sample_size")
    if sample_size and sample_size < len(df):
        rng = random.Random(cfg["data"].get("sample_seed", 42))
        indices = rng.sample(range(len(df)), sample_size)
        df = df.iloc[indices].reset_index(drop=True)
        logger.info("Sampled %d rows from test split.", len(df))
    else:
        logger.info("Using full test split (%d rows).", len(df))
    return df


def _select_few_shot_examples(cfg: dict, train_path: str | None = None) -> list:
    """Return few-shot examples (built-in defaults or random from train set)."""
    selection = cfg["prompting"].get("few_shot_selection", "random")
    n = cfg["prompting"].get("num_few_shot_examples", 3)

    if selection == "random" and train_path and Path(train_path).exists():
        df = pd.read_csv(train_path)
        label_map: dict = cfg["label_map"]
        samples = df.sample(n=n, random_state=cfg["data"].get("sample_seed", 42))
        return [
            {
                "text": row["text"],
                "label": label_map.get(int(row["label_id"]), "Normal"),
                "explanation": "Selected from training set.",
            }
            for _, row in samples.iterrows()
        ]

    # Fall back to the curated built-in examples
    return _FEW_SHOT_EXAMPLES[:n]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    mode = args.mode or cfg["prompting"]["mode"]
    label_map: dict = {int(k): v for k, v in cfg["label_map"].items()}

    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build LLM client
    # ------------------------------------------------------------------
    gen_cfg = cfg["generation"]
    cost_cfg = cfg["cost"]

    client = LLMClient(
        model=gen_cfg["model"],
        api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
        base_url=cfg.get("base_url"),
        temperature=gen_cfg.get("temperature", 0.0),
        max_tokens=gen_cfg.get("max_tokens", 128),
        request_timeout=gen_cfg.get("request_timeout", 30),
        max_retries=gen_cfg.get("max_retries", 3),
        input_price_per_1k=cost_cfg.get("input_price_per_1k", 0.00015),
        output_price_per_1k=cost_cfg.get("output_price_per_1k", 0.00060),
        budget_cap_usd=cost_cfg.get("budget_cap_usd", 5.0),
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = _load_test_data(cfg, args.sample)

    few_shot_examples = []
    if mode == "few_shot":
        few_shot_examples = _select_few_shot_examples(
            cfg,
            train_path=cfg["data"]["test_path"].replace("test.csv", "train.csv"),
        )
        logger.info("Using %d few-shot examples.", len(few_shot_examples))

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    predictions: list[LLMPrediction] = []
    jsonl_path = artifacts_dir / cfg["output"]["predictions_name"]
    parse_errors = 0

    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM classify"):
            try:
                pred = client.classify(
                    text=str(row["text"]),
                    label_map=label_map,
                    mode=mode,
                    few_shot_examples=few_shot_examples,
                )
            except RuntimeError as exc:
                # Budget cap reached
                logger.error("%s", exc)
                break

            if pred.parse_error:
                parse_errors += 1

            rec = {
                "text": pred.text,
                "true_label_id": int(row["label_id"]),
                "predicted_label": pred.predicted_label,
                "predicted_label_id": pred.predicted_label_id,
                "confidence": pred.confidence,
                "explanation": pred.explanation,
                "prompt_tokens": pred.prompt_tokens,
                "completion_tokens": pred.completion_tokens,
                "latency_s": round(pred.latency_s, 3),
                "parse_error": pred.parse_error,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            predictions.append(pred)

    logger.info("Inference done. Parse errors: %d / %d", parse_errors, len(predictions))

    # ------------------------------------------------------------------
    # Metrics (filter out parse failures so they don't contaminate scores)
    # ------------------------------------------------------------------
    valid = [
        (int(df.iloc[i]["label_id"]), p.predicted_label_id)
        for i, p in enumerate(predictions)
        if not p.parse_error and p.predicted_label_id != -1
    ]
    if not valid:
        logger.error("No valid predictions — cannot compute metrics.")
        sys.exit(1)

    y_true, y_pred = zip(*valid)
    metrics = compute_metrics(
        list(y_true),
        list(y_pred),
        label_names=label_map,
        model_name="llm",
        split="test",
    )
    metrics["prompting_mode"] = mode
    metrics["model"] = gen_cfg["model"]
    metrics["total_samples"] = len(predictions)
    metrics["valid_samples"] = len(valid)
    metrics["parse_errors"] = parse_errors

    logger.info(
        "Metrics — accuracy=%.4f  macro_f1=%.4f  weighted_f1=%.4f",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
    )

    save_metrics(metrics, artifacts_dir / cfg["output"]["metrics_name"])

    # ------------------------------------------------------------------
    # Cost report
    # ------------------------------------------------------------------
    cost_report = client.cost.to_dict()
    cost_report["model"] = gen_cfg["model"]
    cost_report["prompting_mode"] = mode
    with open(
        artifacts_dir / cfg["output"]["cost_report_name"], "w", encoding="utf-8"
    ) as f:
        json.dump(cost_report, f, indent=2)
    logger.info(
        "Estimated cost: $%.4f  (%d prompt + %d completion tokens).",
        client.cost.estimated_cost_usd,
        client.cost.total_prompt_tokens,
        client.cost.total_completion_tokens,
    )

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    label_names = [label_map.get(i, str(i)) for i in sorted(set(y_true) | set(y_pred))]
    save_confusion_matrix_plot(
        list(y_true),
        list(y_pred),
        label_names=label_names,
        title=f"LLM ({gen_cfg['model']}) — Test Confusion Matrix",
        save_path=artifacts_dir / cfg["output"]["confusion_matrix_name"],
    )

    logger.info("All artifacts written to %s.", artifacts_dir)


if __name__ == "__main__":
    main()
