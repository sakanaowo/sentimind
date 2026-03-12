"""
Smoke tests for the preprocessing pipeline.
Run: pytest tests/test_preprocessing.py -v
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocess import (
    clean_text,
    normalise_label,
    preprocess_dataframe,
    validate_processed_csv,
    DEFAULT_LABEL_MAP,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_url_removal(self):
        assert "http://example.com" not in clean_text("Visit http://example.com for help")

    def test_mention_removal(self):
        assert "@user" not in clean_text("Hey @user how are you?")

    def test_hashtag_word_preserved(self):
        result = clean_text("Feeling #anxious today")
        assert "anxious" in result
        assert "#" not in result

    def test_lowercase(self):
        assert clean_text("FEELING GREAT") == "feeling great"

    def test_html_entity(self):
        assert "&amp;" not in clean_text("good &amp; bad")

    def test_whitespace_collapse(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result

    def test_non_string_input(self):
        assert clean_text(None) == ""
        assert clean_text(42) == ""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_pure_url_becomes_short(self):
        # URL-only text collapses to empty after cleaning
        result = clean_text("https://example.com")
        assert len(result) < 5  # whitespace or empty


# ---------------------------------------------------------------------------
# normalise_label
# ---------------------------------------------------------------------------

class TestNormaliseLabel:
    def test_exact_match_lowercase(self):
        assert normalise_label("depression", DEFAULT_LABEL_MAP) == 1

    def test_case_insensitive(self):
        assert normalise_label("Depression", DEFAULT_LABEL_MAP) == 1
        assert normalise_label("ANXIETY", DEFAULT_LABEL_MAP) == 2

    def test_whitespace_stripped(self):
        assert normalise_label("  normal  ", DEFAULT_LABEL_MAP) == 0

    def test_unknown_label(self):
        assert normalise_label("happiness", DEFAULT_LABEL_MAP) is None


# ---------------------------------------------------------------------------
# preprocess_dataframe
# ---------------------------------------------------------------------------

def _make_df(**kwargs):
    defaults = {
        "text": [
            "I can't sleep, everything feels hopeless",
            "My heart races every time https://example.com I go outside @someone",
            "Today was great! #Blessed",
            "   ",   # too short after cleaning → should be dropped
            None,    # null → dropped
        ],
        "label": ["depression", "anxiety", "normal", "depression", "depression"],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestPreprocessDataframe:
    def test_output_columns(self):
        df, _ = preprocess_dataframe(_make_df())
        assert set(df.columns) == {"text", "label", "label_id"}

    def test_null_rows_dropped(self):
        df_raw = _make_df()
        df, report = preprocess_dataframe(df_raw)
        assert report["dropped"]["null_rows"] == 1

    def test_short_text_dropped(self):
        df, report = preprocess_dataframe(_make_df(), min_text_length=3)
        assert report["dropped"]["short_text"] >= 1

    def test_label_ids_are_integers(self):
        df, _ = preprocess_dataframe(_make_df())
        assert df["label_id"].dtype.kind == "i"

    def test_label_id_values_within_map(self):
        df, _ = preprocess_dataframe(_make_df())
        valid_ids = set(DEFAULT_LABEL_MAP.values())
        assert set(df["label_id"].unique()).issubset(valid_ids)

    def test_unknown_labels_dropped(self):
        df_raw = _make_df(label=["depression", "anxiety", "unknown_condition", "normal", "depression"])
        df, report = preprocess_dataframe(df_raw)
        assert report["dropped"]["unknown_labels"] == 1

    def test_deterministic_output(self):
        df_raw = _make_df()
        df1, _ = preprocess_dataframe(df_raw.copy())
        df2, _ = preprocess_dataframe(df_raw.copy())
        pd.testing.assert_frame_equal(df1, df2)

    def test_no_nulls_in_output(self):
        df, _ = preprocess_dataframe(_make_df())
        assert df.isnull().sum().sum() == 0

    def test_custom_column_names(self):
        df_raw = pd.DataFrame({
            "content": ["I feel so anxious today"],
            "category": ["anxiety"],
        })
        df, _ = preprocess_dataframe(df_raw, text_col="content", label_col="category")
        assert len(df) == 1

    def test_missing_column_raises(self):
        df_raw = pd.DataFrame({"text": ["hello"]})
        with pytest.raises(ValueError, match="label"):
            preprocess_dataframe(df_raw)

    def test_deduplication(self):
        df_raw = pd.DataFrame({
            "text": ["I feel depressed", "I feel depressed"],
            "label": ["depression", "depression"],
        })
        df, report = preprocess_dataframe(df_raw)
        assert len(df) == 1
        assert report["dropped"]["duplicates"] == 1


# ---------------------------------------------------------------------------
# validate_processed_csv
# ---------------------------------------------------------------------------

class TestValidateProcessedCsv:
    def test_valid_csv(self, tmp_path):
        csv = tmp_path / "train.csv"
        pd.DataFrame({"text": ["hello"], "label": ["normal"], "label_id": [0]}).to_csv(csv, index=False)
        assert validate_processed_csv(csv) is True

    def test_missing_column(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"text": ["hello"]}).to_csv(csv, index=False)
        assert validate_processed_csv(csv) is False

    def test_nonexistent_file(self, tmp_path):
        assert validate_processed_csv(tmp_path / "ghost.csv") is False
