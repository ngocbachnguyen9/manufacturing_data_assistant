import json
import pandas as pd
import pytest

from src.data_generation.error_tracker import ErrorTracker


def test_q1_logging_and_dataframe():
    tracker = ErrorTracker()
    tracker.log_q1_space_injection(5, "colA", "orig", "corr")
    df = tracker.get_log_as_df("Q1")

    # Columns and single entry correct
    assert list(df.columns) == [
        "row", "column", "original", "corrupted", "error_type", "timestamp"
    ]
    assert df.iloc[0]["row"] == 5
    assert df.iloc[0]["error_type"] == "Q1_SPACE"


def test_q2_logging_and_removed_char():
    tracker = ErrorTracker()
    tracker.log_q2_char_missing(7, "colB", "ABCDE", "ABDE", "C", 2)
    df = tracker.get_log_as_df("Q2")

    assert df.shape[0] == 1
    row = df.iloc[0]
    assert row["removed_char"] == "C"
    assert row["position"] == 2
    assert row["error_type"] == "Q2_CHAR_MISSING"


def test_q3_logging_and_json_roundtrip(tmp_path):
    tracker = ErrorTracker()
    original = {"a": 123, "b": "xyz"}
    tracker.log_q3_missing_record(
        3,
        json.dumps(original),
        affected_relationships="gear_to_order",
        impact_assessment="LOW"
    )
    df = tracker.get_log_as_df("Q3")
    rec = json.loads(df.iloc[0]["removed_record"])
    assert rec == original
    assert df.iloc[0]["error_type"] == "Q3_MISSING_RECORD"


def test_save_log_creates_csv(tmp_path):
    tracker = ErrorTracker()
    tracker.log_q1_space_injection(1, "c", "o", "c")
    out = tmp_path / "q1_errors.csv"
    tracker.save_log(str(out), "Q1")

    assert out.exists()
    text = out.read_text()
    assert "Q1_SPACE" in text