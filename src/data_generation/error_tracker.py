import pandas as pd
import os
from datetime import datetime


class ErrorTracker:
    """Logs data corruption events for each quality condition."""

    def __init__(self):
        self.q1_log = []
        self.q2_log = []
        self.q3_log = []

    def log_q1_space_injection(
        self, row: int, column: str, original: str, corrupted: str
    ):
        """Logs a Q1 space injection error."""
        self.q1_log.append(
            {
                "row": row,
                "column": column,
                "original": original,
                "corrupted": corrupted,
                "error_type": "Q1_SPACE",
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_q2_char_missing(
        self,
        row: int,
        column: str,
        original: str,
        corrupted: str,
        removed_char: str,
        position: int,
    ):
        """Logs a Q2 character missing error."""
        self.q2_log.append(
            {
                "row": row,
                "column": column,
                "original": original,
                "corrupted": corrupted,
                "error_type": "Q2_CHAR_MISSING",
                "removed_char": removed_char,
                "position": position,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_q3_missing_record(
        self,
        row: int,
        removed_record: str,
        affected_relationships: str,
        impact_assessment: str,
    ):
        """Logs a Q3 missing record error."""
        self.q3_log.append(
            {
                "row": row,
                "removed_record": removed_record,
                "error_type": "Q3_MISSING_RECORD",
                "affected_relationships": affected_relationships,
                "impact_assessment": impact_assessment,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_log_as_df(self, quality_condition: str) -> pd.DataFrame:
        """Returns the specified error log as a pandas DataFrame."""
        if quality_condition == "Q1":
            return pd.DataFrame(self.q1_log)
        elif quality_condition == "Q2":
            return pd.DataFrame(self.q2_log)
        elif quality_condition == "Q3":
            return pd.DataFrame(self.q3_log)
        return pd.DataFrame()

    def save_log(self, output_path: str, quality_condition: str):
        """Saves the specified error log to a CSV file."""
        log_df = self.get_log_as_df(quality_condition)
        if not log_df.empty:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            log_df.to_csv(output_path, index=False)
            print(f"Saved {quality_condition} error log to {output_path}")