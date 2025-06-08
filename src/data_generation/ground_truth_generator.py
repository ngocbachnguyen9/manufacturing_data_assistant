import pandas as pd
import os
import json
from typing import Dict, List, Any


class GroundTruthGenerator:
    """
    Generates ground truth answers and documents the data traversal paths
    from the perfect (Q0) baseline dataset.
    """

    def __init__(
        self,
        baseline_path: str = "data/experimental_datasets/Q0_baseline",
        output_dir: str = "data/ground_truth",
    ):
        self.baseline_path = baseline_path
        self.output_dir = output_dir
        self.answers_output_path = os.path.join(
            output_dir, "baseline_answers.json"
        )
        self.paths_output_path = os.path.join(
            output_dir, "data_traversal_paths.json"
        )
        self.data = self._load_data()
        self.relationships = self._build_relationship_map()

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        # ... (this method remains unchanged) ...
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

    def _build_relationship_map(self) -> Dict[str, List[str]]:
        # ... (this method remains unchanged) ...
        rel_map = {}
        df = self.data.get("relationship_data")
        if df is None or df.empty:
            return rel_map
        for _, row in df.iterrows():
            parent, child = row["parent"], row["child"]
            if parent not in rel_map:
                rel_map[parent] = []
            if child not in rel_map:
                rel_map[child] = []
            rel_map[parent].append(child)
        return rel_map

    def generate_all_ground_truths(self):
        # ... (this method remains unchanged) ...
        all_answers = []
        all_paths = {}

        orders = [
            p for p in self.relationships.keys() if p.startswith("ORBOX")
        ]
        gears = [
            c for c in self.relationships.keys() if c.startswith("3DOR")
        ]

        for i, order_id in enumerate(orders[:5]):
            answer, path = self.generate_easy_task(order_id, i)
            all_answers.append(answer)
            all_paths[answer["task_id"]] = path

        for i, gear_id in enumerate(gears[:5]):
            answer, path = self.generate_medium_task(gear_id, i)
            all_answers.append(answer)
            all_paths[answer["task_id"]] = path

        for i, order_id in enumerate(orders[:5]):
            answer, path = self.generate_hard_task(order_id, i)
            all_answers.append(answer)
            all_paths[answer["task_id"]] = path

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.answers_output_path, "w") as f:
            json.dump(all_answers, f, indent=2)
        print(f"Ground truth answers saved to {self.answers_output_path}")

        with open(self.paths_output_path, "w") as f:
            json.dump(all_paths, f, indent=2)
        print(f"Data traversal paths saved to {self.paths_output_path}")

    def generate_easy_task(self, order_id: str, index: int) -> tuple:
        """
        UPDATED: Finds the UNIQUE set of gears for an order.
        """
        raw_gears = self.relationships.get(order_id, [])
        # NEW: Deduplicate and sort the list of gears
        unique_gears = sorted(list(set(raw_gears)))

        answer = {
            "task_id": f"easy_{order_id}_{index}",
            "complexity_level": "easy",
            "query_instance": f"Find all gears for Order {order_id}",
            "baseline_answer": {
                "order_id": order_id,
                "gear_count": len(unique_gears),  # Count of unique gears
                "gear_list": unique_gears,  # The unique list
            },
        }
        path = {
            "data_sources": ["relationship_data"],
            "steps": [
                f"1. Query relationship_data where parent='{order_id}'",
                f"2. Collect all child entities and find unique set: {unique_gears}",
            ],
        }
        return answer, path

    def generate_medium_task(self, gear_id: str, index: int) -> tuple:
        # ... (this method remains unchanged) ...
        printer_id = "UNKNOWN"
        for parent, children in self.relationships.items():
            if gear_id in children and parent.startswith("Printer"):
                printer_id = parent
                break
        answer = {
            "task_id": f"medium_{gear_id}_{index}",
            "complexity_level": "medium",
            "query_instance": f"Determine printer for Part {gear_id}",
            "baseline_answer": {
                "part_id": gear_id,
                "assigned_printer": printer_id,
            },
        }
        path = {
            "data_sources": ["relationship_data"],
            "steps": [
                f"1. Query relationship_data where child='{gear_id}'",
                f"2. Find parent entity starting with 'Printer_': {printer_id}",
            ],
        }
        return answer, path

    def generate_hard_task(self, product_id: str, index: int) -> tuple:
        # ... (this method remains unchanged) ...
        location_df = self.data["location_data"]
        cert_date = "2024-10-28"
        gears_for_order = self.relationships.get(product_id, [])
        arrival_date = "UNKNOWN"
        path_steps = [
            f"1. Query relationship_data for children of '{product_id}': Found {len(gears_for_order)} gears.",
        ]

        if gears_for_order:
            warehouse_scans = location_df[
                location_df["_value"].isin(gears_for_order)
                & (location_df["location"] == "Parts Warehouse")
            ]
            path_steps.append(
                f"2. Query location_data for these gears at 'Parts Warehouse': Found {len(warehouse_scans)} scans."
            )
            if not warehouse_scans.empty:
                max_timestamp = pd.to_datetime(
                    warehouse_scans["_time"]
                ).max()
                arrival_date = max_timestamp.strftime("%Y-%m-%d")
                path_steps.append(
                    f"3. Determined latest arrival timestamp: {arrival_date}"
                )

        date_match = cert_date == arrival_date
        answer = {
            "task_id": f"hard_{product_id}_{index}",
            "complexity_level": "hard",
            "query_instance": f"Verify ARC date vs warehouse arrival for {product_id}",
            "baseline_answer": {
                "product_id": product_id,
                "certificate_date": cert_date,
                "warehouse_arrival_date": arrival_date,
                "date_match_status": date_match,
            },
        }
        path = {
            "data_sources": ["relationship_data", "location_data"],
            "steps": path_steps,
        }
        return answer, path