import pandas as pd
import numpy as np
import os
import random
import json
from typing import Tuple, Dict
from src.data_generation.error_tracker import ErrorTracker


class DataQualityController:
    """
    Applies systematic data corruption to a baseline dataset to generate
    Q1, Q2, and Q3 quality conditions.
    """

    def __init__(self, baseline_path: str = "data/experimental_datasets/Q0_baseline"):
        self.baseline_path = baseline_path
        self.datasets = self._load_baseline()
        self.id_fields = {
            "location_data": ["_value"],
            "worker_data": ["_value"],
            "relationship_data": ["child", "parent"],
        }

    def _load_baseline(self) -> Dict[str, pd.DataFrame]:
        """Loads the Q0 baseline dataset."""
        datasets = {}
        for filename in os.listdir(self.baseline_path):
            if filename.endswith(".csv"):
                key = filename.replace(".csv", "")
                datasets[key] = pd.read_csv(
                    os.path.join(self.baseline_path, filename),
                    dtype=str,
                    keep_default_na=False,
                )
        return datasets

    def apply_corruption(
        self, quality_condition: str
    ) -> Tuple[Dict[str, pd.DataFrame], ErrorTracker]:
        """
        Applies a specific quality corruption to the loaded baseline data.
        """
        corrupted_data = {k: v.copy() for k, v in self.datasets.items()}
        error_tracker = ErrorTracker()

        print(f"***REMOVED***n--- Applying {quality_condition} Corruption ---")

        if quality_condition == "Q1":
            self._apply_q1_spaces(corrupted_data, error_tracker)
        elif quality_condition == "Q2":
            self._apply_q2_char_missing(corrupted_data, error_tracker)
        elif quality_condition == "Q3":
            self._apply_q3_missing_records(corrupted_data, error_tracker)
        else:
            print(f"Warning: Unknown quality condition '{quality_condition}'")

        return corrupted_data, error_tracker

    def _apply_q1_spaces(self, data: Dict, tracker: ErrorTracker):
        """Q1: Injects space errors into 15% of barcode/ID fields."""
        corruption_rate = 0.15
        for df_name, id_cols in self.id_fields.items():
            df = data[df_name]
            for col in id_cols:
                if col not in df.columns:
                    continue
                # Select 15% of rows to corrupt
                indices_to_corrupt = df.sample(frac=corruption_rate).index
                for idx in indices_to_corrupt:
                    original_val = str(df.at[idx, col])
                    if not original_val:
                        continue

                    # Insert 1-3 spaces at start, end, or both
                    num_spaces = random.randint(1, 3)
                    spaces = " " * num_spaces
                    pos = random.choice(["start", "end", "both"])
                    if pos == "start":
                        corrupted_val = spaces + original_val
                    elif pos == "end":
                        corrupted_val = original_val + spaces
                    else:
                        corrupted_val = spaces + original_val + spaces

                    df.at[idx, col] = corrupted_val
                    tracker.log_q1_space_injection(
                        idx, col, original_val, corrupted_val
                    )

    def _apply_q2_char_missing(self, data: Dict, tracker: ErrorTracker):
        """Q2: Removes a single character from 12% of Gear/Order IDs."""
        corruption_rate = 0.12
        targets = ["3DOR", "ORBOX"]
        for df_name, id_cols in self.id_fields.items():
            df = data[df_name]
            for col in id_cols:
                if col not in df.columns:
                    continue
                # Filter for rows containing target IDs
                target_rows = df[
                    df[col].str.startswith(tuple(targets), na=False)
                ]
                indices_to_corrupt = target_rows.sample(
                    frac=corruption_rate
                ).index

                for idx in indices_to_corrupt:
                    original_val = str(df.at[idx, col])
                    if len(original_val) < 2:
                        continue

                    # Avoid removing leading chars to maintain some format
                    pos_to_remove = random.randint(1, len(original_val) - 1)
                    removed_char = original_val[pos_to_remove]
                    corrupted_val = (
                        original_val[:pos_to_remove]
                        + original_val[pos_to_remove + 1 :]
                    )

                    df.at[idx, col] = corrupted_val
                    tracker.log_q2_char_missing(
                        idx,
                        col,
                        original_val,
                        corrupted_val,
                        removed_char,
                        pos_to_remove,
                    )

    def _apply_q3_missing_records(self, data: Dict, tracker: ErrorTracker):
        """Q3: Strategically deletes 5-8% of records."""
        # For this example, we'll focus on relationship_data
        df_name = "relationship_data"
        corruption_rate = random.uniform(0.05, 0.08)
        df = data[df_name]

        if df.empty:
            return

        # Priority: Intermediate links (gear-to-order)
        target_rows = df[
            df["parent"].str.startswith("ORBOX", na=False)
            & df["child"].str.startswith("3DOR", na=False)
        ]
        indices_to_remove = target_rows.sample(frac=corruption_rate).index

        for idx in indices_to_remove:
            removed_record = df.loc[idx].to_dict()
            tracker.log_q3_missing_record(
                idx,
                json.dumps(removed_record),
                "gear_to_order",
                "MEDIUM",
            )

        # Drop the selected rows from the dataframe
        data[df_name] = df.drop(indices_to_remove).reset_index(drop=True)

    def save_corrupted_data(
        self,
        corrupted_data: Dict[str, pd.DataFrame],
        error_tracker: ErrorTracker,
        quality_condition: str,
    ):
        """Saves the corrupted data and its corresponding error log."""
        output_dir = f"data/experimental_datasets/{quality_condition}_dataset"
        os.makedirs(output_dir, exist_ok=True)

        for name, df in corrupted_data.items():
            # Save corrupted data file
            corrupted_path = os.path.join(
                output_dir, f"{name}_{quality_condition}.csv"
            )
            df.to_csv(corrupted_path, index=False)

            # Save corresponding error log
            log_path = os.path.join(
                output_dir, f"{name}_{quality_condition}_errors.csv"
            )
            error_tracker.save_log(log_path, quality_condition)