import pandas as pd
import os
from datetime import datetime


class ErrorTracker:
    """Logs order-level data corruption events for each quality condition."""

    def __init__(self):
        self.q1_log = []
        self.q2_log = []
        self.q3_log = []

    def log_q1_space_injection(self, row: int, table_column: str, original: str, corrupted: str):
        """Log a Q1 space injection error with table context."""
        self.q1_log.append({
            "row": row,
            "table_column": table_column,
            "original": original,
            "corrupted": corrupted,
            "error_type": "Q1_SPACE_ORDER_LEVEL",
            "timestamp": datetime.now().isoformat(),
        })

    def log_q2_char_missing(self, row: int, table_column: str, original: str, 
                          corrupted: str, removed_char: str, position: int):
        """Log a Q2 character missing error with table context."""
        self.q2_log.append({
            "row": row,
            "table_column": table_column,
            "original": original,
            "corrupted": corrupted,
            "error_type": "Q2_CHAR_MISSING_ORDER_LEVEL",
            "removed_char": removed_char,
            "position": position,
            "timestamp": datetime.now().isoformat(),
        })

    def log_q3_missing_record(self, row: int, removed_record: str, 
                            affected_relationships: str, impact_assessment: str):
        """Log a Q3 gear→order relationship deletion."""
        self.q3_log.append({
            "row": row,
            "removed_record": removed_record,
            "error_type": "Q3_GEAR_ORDER_DELETION",
            "affected_relationships": affected_relationships,
            "impact_assessment": impact_assessment,
            "timestamp": datetime.now().isoformat(),
        })

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
        """Save the specified error log to a CSV file."""
        log_df = self.get_log_as_df(quality_condition)
        if not log_df.empty:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            log_df.to_csv(output_path, index=False)
            print(f"Saved {quality_condition} error log: {output_path} ({len(log_df)} entries)")
        else:
            print(f"No {quality_condition} errors to save")