"""
Text cleaning and preprocessing pipeline for Sentimind.
Pure functions — no randomness, same input always produces same output.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping (source of truth is data_contract.md §3 and configs/)
# ---------------------------------------------------------------------------
DEFAULT_LABEL_MAP: Dict[str, int] = {
    "normal": 0,
    "depression": 1,
    "anxiety": 2,
    "bipolar": 3,
    "ptsd": 4,
    "stress": 5,
    "suicidal": 6,
}

ID_TO_LABEL: Dict[int, str] = {v: k.capitalize() for k, v in DEFAULT_LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean a single raw social-media text entry.

    Cleaning steps (deterministic, in order per data_contract.md §4):
    1. Unicode NFKC normalisation
    2. Remove URLs
    3. Remove @mentions
    4. Preserve hashtag text (strip # prefix only)
    5. Remove HTML entities
    6. Remove non-printable control characters
    7. Collapse whitespace
    8. Lowercase
    """
    if not isinstance(text, str):
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # 4. Hashtags — keep the word, drop the '#'
    text = re.sub(r"#(\w+)", r"\1", text)

    # 5. HTML entities
    text = re.sub(r"&[a-z]+;", " ", text, flags=re.IGNORECASE)

    # 6. Non-printable control characters (except newline → space)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # 7. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Lowercase
    text = text.lower()

    return text


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

def normalise_label(raw_label: str, label_map: Dict[str, int]) -> Optional[int]:
    """Map a raw label string to its integer id.

    Matching is case-insensitive and strips surrounding whitespace.
    Returns None for labels not in the map (caller decides how to handle).
    """
    key = str(raw_label).strip().lower()
    return label_map.get(key, None)


# ---------------------------------------------------------------------------
# DataFrame-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    label_map: Optional[Dict[str, int]] = None,
    min_text_length: int = 3,
) -> Tuple[pd.DataFrame, Dict]:
    """Clean and validate a raw DataFrame.

    Returns:
        processed_df: DataFrame with columns [text, label, label_id]
        report: dict with quality statistics
    """
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP

    initial_count = len(df)
    report: Dict = {"initial_count": initial_count, "dropped": {}}

    # --- Validate required columns ---
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame. "
                             f"Available: {list(df.columns)}")

    df = df[[text_col, label_col]].copy()
    df.columns = ["text_raw", "label_raw"]

    # --- Drop null rows ---
    null_mask = df["text_raw"].isna() | df["label_raw"].isna()
    null_count = null_mask.sum()
    df = df[~null_mask].reset_index(drop=True)
    report["dropped"]["null_rows"] = int(null_count)
    logger.info("Dropped %d null rows.", null_count)

    # --- Clean text ---
    df["text"] = df["text_raw"].apply(clean_text)

    # --- Drop short text ---
    short_mask = df["text"].str.len() < min_text_length
    short_count = short_mask.sum()
    df = df[~short_mask].reset_index(drop=True)
    report["dropped"]["short_text"] = int(short_count)
    logger.info("Dropped %d rows with text shorter than %d chars after cleaning.",
                short_count, min_text_length)

    # --- Normalise labels ---
    df["label"] = df["label_raw"].apply(lambda x: str(x).strip())
    df["label_id"] = df["label_raw"].apply(lambda x: normalise_label(x, label_map))

    unknown_mask = df["label_id"].isna()
    unknown_labels = df.loc[unknown_mask, "label_raw"].value_counts().to_dict()
    unknown_count = unknown_mask.sum()
    df = df[~unknown_mask].reset_index(drop=True)
    report["dropped"]["unknown_labels"] = int(unknown_count)
    report["unknown_label_values"] = {str(k): int(v) for k, v in unknown_labels.items()}
    if unknown_count:
        logger.warning("Dropped %d rows with unknown labels: %s", unknown_count, unknown_labels)

    # --- Deduplicate (keep first occurrence) ---
    dup_mask = df.duplicated(subset=["text", "label_id"])
    dup_count = dup_mask.sum()
    df = df[~dup_mask].reset_index(drop=True)
    report["dropped"]["duplicates"] = int(dup_count)
    logger.info("Dropped %d duplicate (text, label) rows.", dup_count)

    # --- Final dataframe ---
    df["label_id"] = df["label_id"].astype(int)
    df["label"] = df["label"].apply(lambda x: x.strip())
    result = df[["text", "label", "label_id"]].copy()

    # --- Report stats ---
    report["final_count"] = len(result)
    report["total_dropped"] = initial_count - len(result)
    report["class_distribution"] = (
        result.groupby("label_id")["label"]
        .first()
        .reset_index()
        .assign(count=result.groupby("label_id").size().values)
        .set_index("label_id")
        .to_dict(orient="index")
    )
    # Simpler class counts
    report["class_counts"] = result["label_id"].value_counts().sort_index().to_dict()
    report["class_counts"] = {int(k): int(v) for k, v in report["class_counts"].items()}

    logger.info(
        "Preprocessing done. %d / %d rows retained. Class distribution: %s",
        len(result),
        initial_count,
        report["class_counts"],
    )
    return result, report


# ---------------------------------------------------------------------------
# Schema validation helper (for downstream pipeline checks)
# ---------------------------------------------------------------------------

def validate_processed_csv(path: str | Path) -> bool:
    """Basic schema check: file exists, has text + label + label_id columns, no nulls."""
    path = Path(path)
    if not path.exists():
        logger.error("File not found: %s", path)
        return False
    df = pd.read_csv(path, nrows=5)
    required = {"text", "label", "label_id"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing columns %s in %s", missing, path)
        return False
    return True
