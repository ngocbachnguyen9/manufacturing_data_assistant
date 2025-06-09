import pandas as pd
import os
import json
from typing import Dict, Any, List


class GroundTruthValidator:
    """
    Performs automated validation of the generated ground truth files against
    the Q0 baseline dataset.
    """

    def __init__(
        self,
        baseline_path: str = "data/experimental_datasets/Q0_baseline",
        ground_truth_dir: str = "data/ground_truth",
    ):
        self.baseline_path = baseline_path
        self.ground_truth_dir = ground_truth_dir
        self.datasets = self._load_datasets()
        self.answers = self._load_json("baseline_answers.json")
        self.paths = self._load_json("data_traversal_paths.json")
        self.all_ids = self._collect_all_baseline_ids()

    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Loads all Q0 baseline CSVs into a dictionary of DataFrames."""
        data = {}
        for filename in os.listdir(self.baseline_path):
            if filename.endswith(".csv"):
                key = filename.replace(".csv", "")
                data[key] = pd.read_csv(
                    os.path.join(self.baseline_path, filename),
                    dtype=str,
                    keep_default_na=False,
                )
        return data

    def _load_json(self, filename: str) -> Any:
        """Loads a JSON file from the ground truth directory."""
        path = os.path.join(self.ground_truth_dir, filename)
        with open(path, "r") as f:
            return json.load(f)

    def _collect_all_baseline_ids(self) -> set:
        """Gathers all unique IDs from all baseline tables for existence checks."""
        all_ids = set()
        for df in self.datasets.values():
            for col in df.columns:
                # Assuming any column with 'id' or '_value' contains relevant IDs
                if "id" in col.lower() or "_value" in col.lower():
                    all_ids.update(df[col].unique())
        return all_ids

    def run_all_validations(self):
        """Executes all validation checks and prints a summary."""
        print("--- Running Ground Truth Validation ---")
        results = []
        results.append(self.validate_entity_references())
        results.append(self.validate_path_consistency())
        results.append(self.validate_hard_task_logic())

        print("***REMOVED***n--- Validation Summary ---")
        if all(results):
            print("✅ All ground truth validation checks passed.")
        else:
            print("❌ Some ground truth validation checks failed.")

    def validate_entity_references(self) -> bool:
        """Check 1: Verifies that all IDs in answers exist in the baseline data."""
        print("***REMOVED***n1. Validating Entity References...")
        all_valid = True
        for task in self.answers:
            answer = task["baseline_answer"]
            for key, value in answer.items():
                ids_to_check = []
                if isinstance(value, str) and value in self.all_ids:
                    ids_to_check.append(value)
                elif isinstance(value, list):
                    ids_to_check.extend(value)

                for entity_id in ids_to_check:
                    if entity_id not in self.all_ids:
                        print(
                            f"  - FAIL: In task {task['task_id']}, entity ID '{entity_id}' not found in baseline data."
                        )
                        all_valid = False
        if all_valid:
            print("  - PASS: All entity references in answers are valid.")
        return all_valid

    def validate_path_consistency(self) -> bool:
        """Check 2: Verifies that data sources in paths are valid files."""
        print("***REMOVED***n2. Validating Traversal Path Consistency...")
        all_valid = True
        valid_sources = {f.replace(".csv", "") for f in os.listdir(self.baseline_path)}
        for task_id, path_info in self.paths.items():
            for source in path_info["data_sources"]:
                if source not in valid_sources:
                    print(
                        f"  - FAIL: In path for {task_id}, data source '{source}' is not a valid dataset."
                    )
                    all_valid = False
        if all_valid:
            print("  - PASS: All data sources in traversal paths are valid.")
        return all_valid

    def validate_hard_task_logic(self) -> bool:
        """Check 3: Re-calculates and verifies a 'Hard' task answer."""
        print("***REMOVED***n3. Validating 'Hard' Task Manufacturing Logic...")
        # Find the first hard task to re-calculate
        hard_task = next(
            (t for t in self.answers if t["complexity_level"] == "hard"),
            None,
        )
        if not hard_task:
            print("  - SKIP: No 'Hard' tasks found to validate.")
            return True

        # Re-calculate the answer using the same logic as the generator
        product_id = hard_task["baseline_answer"]["product_id"]
        rel_df = self.datasets["relationship_data"]
        loc_df = self.datasets["location_data"]
        gears = rel_df[rel_df["parent"] == product_id]["child"].unique()
        scans = loc_df[
            loc_df["_value"].isin(gears)
            & (loc_df["location"] == "Parts Warehouse")
        ]
        recalculated_date = (
            pd.to_datetime(scans["_time"]).max().strftime("%Y-%m-%d")
            if not scans.empty
            else "UNKNOWN"
        )

        stored_date = hard_task["baseline_answer"]["warehouse_arrival_date"]

        if recalculated_date == stored_date:
            print(
                f"  - PASS: Recalculated arrival date for {product_id} ('{recalculated_date}') matches stored ground truth."
            )
            return True
        else:
            print(
                f"  - FAIL: Logic mismatch for {product_id}. Stored: '{stored_date}', Recalculated: '{recalculated_date}'."
            )
            return False


if __name__ == "__main__":
    validator = GroundTruthValidator()
    validator.run_all_validations()