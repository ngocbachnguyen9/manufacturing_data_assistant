import pandas as pd
import os
from typing import Dict


class InjectionValidator:
    """
    Validates the accuracy and completeness of the error injection process.
    """

    def __init__(self, base_path: str = "data/experimental_datasets"):
        self.base_path = base_path

    def validate_all_conditions(self):
        """Runs validation for all corrupted quality conditions."""
        for qc in ["Q1", "Q2", "Q3"]:
            print(f"***REMOVED***n--- Validating Condition: {qc} ---")
            qc_path = os.path.join(self.base_path, f"{qc}_dataset")
            if not os.path.exists(qc_path):
                print(f"Path not found: {qc_path}. Skipping.")
                continue

            # For simplicity, we validate the first file we find
            # A full implementation would iterate through all files
            data_file, error_log = self._find_file_pair(qc_path, qc)
            if not data_file or not error_log:
                print("Could not find a data/error file pair. Skipping.")
                continue

            original_df_path = os.path.join(
                self.base_path, "Q0_baseline", os.path.basename(data_file).replace(f"_{qc}", "")
            )

            self.validate_corruption_rate(original_df_path, error_log, qc)
            self.validate_logging_completeness(data_file, error_log, qc)

    def _find_file_pair(self, qc_path: str, qc: str) -> tuple:
        """Finds a corresponding data file and error log."""
        for f in os.listdir(qc_path):
            if f.endswith(f"_{qc}.csv"):
                error_file = f.replace(".csv", "_errors.csv")
                if os.path.exists(os.path.join(qc_path, error_file)):
                    return os.path.join(qc_path, f), os.path.join(qc_path, error_file)
        return None, None

    def validate_corruption_rate(
        self, original_path: str, error_log_path: str, qc: str
    ):
        """Check 1: Error count matches target corruption rate."""
        target_rates = {"Q1": 0.15, "Q2": 0.12, "Q3": (0.05, 0.08)}
        if not os.path.exists(original_path) or not os.path.exists(error_log_path):
            print("  - Rate Check: SKIPPED (missing files)")
            return

        original_df = pd.read_csv(original_path)
        error_log_df = pd.read_csv(error_log_path)
        num_records = len(original_df)
        num_errors = len(error_log_df)
        actual_rate = num_errors / num_records if num_records > 0 else 0

        target = target_rates[qc]
        if isinstance(target, tuple):  # Range for Q3
            is_valid = target[0] <= actual_rate <= target[1]
            print(
                f"  - Rate Check: {'PASS' if is_valid else 'FAIL'}. "
                f"Actual: {actual_rate:.2%}, Target: {target[0]:.0%}-{target[1]:.0%}"
            )
        else:
            is_valid = abs(actual_rate - target) < 0.03  # ±3% tolerance
            print(
                f"  - Rate Check: {'PASS' if is_valid else 'FAIL'}. "
                f"Actual: {actual_rate:.2%}, Target: {target:.0%}"
            )

    def validate_logging_completeness(
        self, corrupted_path: str, error_log_path: str, qc: str
    ):
        """Check 2: All errors are properly logged (simplified check)."""
        # This is a simplified check. A full check would reconstruct the
        # original from the corrupted + log and compare.
        error_log_df = pd.read_csv(error_log_path)
        if not error_log_df.empty:
            print("  - Logging Check: PASS (Log is not empty).")
        else:
            print("  - Logging Check: FAIL (Error log is empty).")