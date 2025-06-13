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
        all_answers = []
        all_paths = {}

        # Correctly source Packing List IDs for easy tasks
        packing_list_dir = "data/generated_documents/packing_lists"
        packing_lists = []
        if os.path.exists(packing_list_dir):
            for f in os.listdir(packing_list_dir):
                if f.startswith("PackingList-") and f.endswith(".docx"):
                    packing_lists.append(f.replace("PackingList-", "").replace(".docx", ""))
        
        if not packing_lists:
             # Fallback for testing when docs don't exist
            packing_lists = [p.replace("ORBOX", "PL") for p in self.relationships.keys() if p.startswith("ORBOX")]


        gears = [
            c for c in self.relationships.keys() if c.startswith("3DOR")
        ]
        orders = [
            p for p in self.relationships.keys() if p.startswith("ORBOX")
        ]

        # Iterate over packing_list_ids for easy tasks
        for i, pl_id in enumerate(packing_lists[:5]):
            answer, path = self.generate_easy_task(pl_id, i)
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


    def generate_easy_task(self, packing_list_id: str, index: int) -> tuple:
        """
        UPDATED: Simulates the full flow from Packing List to gears.
        """
        # Step 1: Find the Order ID from the packing list document.
        # This simulates the packing_list_parser_tool.
        # NOTE: This assumes a simple mapping. In a real system, this would parse the file.
        order_id = "ORBOX" + packing_list_id.replace("PL", "")

        # Step 2: Find the unique gears for that Order ID.
        raw_gears = self.relationships.get(order_id, [])
        unique_gears = sorted(list(set(raw_gears)))

        answer = {
            # The task_id is now based on the Packing List ID.
            "task_id": f"easy_{packing_list_id}_{index}",
            "complexity_level": "easy",
            "query_instance": f"Find all gears for Packing List {packing_list_id}",
            "baseline_answer": {
                "packing_list_id": packing_list_id,
                "order_id": order_id,
                "gear_count": len(unique_gears),
                "gear_list": unique_gears,
            },
        }
        path = {
            "data_sources": ["packing_lists", "relationship_data"],
            "steps": [
                f"1. Parse Packing List '{packing_list_id}' to find Order ID: '{order_id}'",
                f"2. Query relationship_data where parent='{order_id}'",
                f"3. Collect unique child entities: {unique_gears}",
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