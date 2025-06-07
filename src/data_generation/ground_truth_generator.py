import pandas as pd
import os
import json
from typing import Dict, List, Any


class GroundTruthGenerator:
    """
    Generates ground truth answers from the perfect (Q0) baseline dataset.
    """

    def __init__(
        self,
        baseline_path: str = "data/experimental_datasets/Q0_baseline",
        output_path: str = "data/ground_truth/baseline_answers.json",
    ):
        self.baseline_path = baseline_path
        self.output_path = output_path
        self.data = self._load_data()
        self.relationships = self._build_relationship_map()

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Loads all CSV files from the baseline directory."""
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
        """Creates a dictionary for easy traversal of parent-child links."""
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
        """
        Generates and saves ground truth for all task types.
        """
        all_answers = []
        # Easy Tasks: Find all gears for each order
        orders = [
            p for p in self.relationships.keys() if p.startswith("ORBOX")
        ]
        for i, order_id in enumerate(orders[:5]):  # Limit for example
            all_answers.append(self.generate_easy_task(order_id, i))

        # Medium Tasks: Find printer and part count for a sample of gears
        gears = [
            c for c in self.relationships.keys() if c.startswith("3DOR")
        ]
        for i, gear_id in enumerate(gears[:5]):  # Limit for example
            all_answers.append(self.generate_medium_task(gear_id, i))

        # Hard Tasks: Document cross-reference (using order ID as product ID)
        for i, order_id in enumerate(orders[:5]):  # Limit for example
            all_answers.append(self.generate_hard_task(order_id, i))

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(all_answers, f, indent=2)
        print(f"Ground truth saved to {self.output_path}")

    def generate_easy_task(self, order_id: str, index: int) -> Dict:
        """Ground truth for: 'Find all gears for Order X'"""
        gears = self.relationships.get(order_id, [])
        return {
            "task_id": f"easy_{order_id}_{index}",
            "complexity_level": "easy",
            "query_instance": f"Find all gears for Order {order_id}",
            "baseline_answer": {
                "order_id": order_id,
                "gear_count": len(gears),
                "gear_list": sorted(gears),
            },
        }

    def generate_medium_task(self, gear_id: str, index: int) -> Dict:
        """Ground truth for: 'Determine printer for Part Y'"""
        printer_id = "UNKNOWN"
        for parent, children in self.relationships.items():
            if gear_id in children and parent.startswith("Printer"):
                printer_id = parent
                break
        return {
            "task_id": f"medium_{gear_id}_{index}",
            "complexity_level": "medium",
            "query_instance": f"Determine printer for Part {gear_id}",
            "baseline_answer": {"part_id": gear_id, "assigned_printer": printer_id},
        }

    def generate_hard_task(self, product_id: str, index: int) -> Dict:
        """Ground truth for: 'Verify ARC dates vs warehouse arrival'"""
        # Note: This is a simplified version. Real implementation would parse
        # the PDF and location data more deeply.
        location_df = self.data["location_data"]
        warehouse_scans = location_df[
            (location_df["_value"] == product_id)
            & (location_df["location"] == "Parts Warehouse")
        ]
        arrival_date = (
            warehouse_scans["_time"].min()[:10]
            if not warehouse_scans.empty
            else "UNKNOWN"
        )
        cert_date = "2024-10-28"  # From template
        return {
            "task_id": f"hard_{product_id}_{index}",
            "complexity_level": "hard",
            "query_instance": f"Verify ARC date vs warehouse arrival for {product_id}",
            "baseline_answer": {
                "product_id": product_id,
                "certificate_date": cert_date,
                "warehouse_arrival_date": arrival_date,
                "date_match_status": cert_date == arrival_date,
            },
        }