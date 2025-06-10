import pandas as pd
import os
import random
import json
from typing import Tuple, Dict, List, Set
from src.data_generation.error_tracker import ErrorTracker


class DataQualityController:
    """
    Applies order-level data corruption to generate Q1, Q2, and Q3 quality conditions.
    """

    def __init__(self, baseline_path: str = "data/experimental_datasets/Q0_baseline"):
        self.baseline_path = baseline_path
        self.datasets = self._load_baseline()
        
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

    def _get_all_orders(self) -> List[str]:
        """Extract all unique orders from relationship data."""
        rel_df = self.datasets.get("relationship_data")
        if rel_df is None or rel_df.empty:
            return []
        
        # Find all ORBOX entries in parent column
        orders = rel_df[rel_df["parent"].str.startswith("ORBOX", na=False)]["parent"].unique()
        return sorted(list(orders))

    def _get_order_related_entities(self, order_id: str) -> Dict[str, Set[str]]:
        """
        For a given order, find all related entities across all tables.
        Returns dict with entity types as keys and sets of IDs as values.
        """
        rel_df = self.datasets.get("relationship_data")
        if rel_df is None or rel_df.empty:
            return {}
        
        entities = {
            "orders": {order_id},
            "gears": set(),
            "print_jobs": set(),
            "printers": set()
        }
        
        # Find gears for this order
        order_gears = rel_df[rel_df["parent"] == order_id]["child"].tolist()
        entities["gears"].update(order_gears)
        
        # Find print jobs and printers for these gears
        for gear in order_gears:
            # Find what produced this gear (print jobs)
            gear_parents = rel_df[rel_df["child"] == gear]["parent"].tolist()
            for parent in gear_parents:
                if parent.startswith("Printer_"):
                    entities["printers"].add(parent)
                elif parent.isdigit() or parent not in entities["orders"]:
                    # Numeric print job IDs or other production entities
                    entities["print_jobs"].add(parent)
        
        return entities

    def _find_and_corrupt_related_records(self, corruption_map: Dict[str, str], 
                                        tracker: ErrorTracker, 
                                        quality_condition: str) -> None:
        """Apply corruption to all records using a pre-defined corruption map."""
        
        # Corrupt records in each table
        for table_name, df in self.datasets.items():
            if df.empty:
                continue
                
            id_columns = []
            if table_name == "relationship_data":
                id_columns = ["child", "parent"]
            elif table_name in ["location_data", "worker_data"]:
                id_columns = ["_value"]
            
            for col in id_columns:
                if col not in df.columns:
                    continue
                    
                # Find rows where this column contains an ID from our map
                mask = df[col].isin(corruption_map.keys())
                indices_to_corrupt = df[mask].index.tolist()
                
                for idx in indices_to_corrupt:
                    original_val = str(df.at[idx, col])
                    if original_val in corruption_map:
                        corrupted_val = corruption_map[original_val] # Use the map
                        df.at[idx, col] = corrupted_val
                        
                        # Log the corruption
                        if quality_condition == "Q1":
                            tracker.log_q1_space_injection(idx, f"{table_name}.{col}", 
                                                         original_val, corrupted_val)
                        elif quality_condition == "Q2":
                            removed_char, position = self._find_removed_char(original_val, corrupted_val)
                            tracker.log_q2_char_missing(idx, f"{table_name}.{col}", 
                                                       original_val, corrupted_val,
                                                       removed_char, position)

    def _find_removed_char(self, original: str, corrupted: str) -> Tuple[str, int]:
        """Find which character was removed and its position."""
        if len(original) <= len(corrupted):
            return "", -1
            
        for i in range(len(original)):
            if i >= len(corrupted) or original[i] != corrupted[i]:
                return original[i], i
        return original[-1], len(original) - 1

    # UPDATED: The main apply method now returns the targeted IDs
    def apply_corruption(
        self, quality_condition: str
    ) -> Tuple[Dict[str, pd.DataFrame], ErrorTracker, List[str]]:
        """Apply corruption and return the corrupted data, tracker, and targeted IDs."""
        corrupted_data = {k: v.copy() for k, v in self.datasets.items()}
        error_tracker = ErrorTracker()
        targeted_ids = []

        # Temporarily replace datasets with corrupted copies for processing
        original_datasets = self.datasets
        self.datasets = corrupted_data

        print(f"***REMOVED***n--- Applying {quality_condition} Corruption (Order-Level) ---")

        if quality_condition == "Q1":
            targeted_ids = self._apply_q1_order_level(error_tracker)
        elif quality_condition == "Q2":
            targeted_ids = self._apply_q2_order_level(error_tracker)
        elif quality_condition == "Q3":
            targeted_ids = self._apply_q3_gear_order_deletion(error_tracker)

        self.datasets = original_datasets
        return corrupted_data, error_tracker, targeted_ids

    def _apply_q1_order_level(self, tracker: ErrorTracker):
        """Q1: Apply space injection to 15% of orders and all their related records."""
        orders = self._get_all_orders()
        if not orders:
            print("No orders available for Q1 corruption.")
            return []
        num_orders_to_corrupt = max(1, int(len(orders) * 0.15))
        selected_orders = random.sample(orders, num_orders_to_corrupt)
        
        print(f"Q1: Corrupting {num_orders_to_corrupt}/{len(orders)} orders: {selected_orders}")
        
        corruption_map = {}
        for order_id in selected_orders:
            entities = self._get_order_related_entities(order_id)
            all_entity_ids = set().union(*entities.values())
            
            for entity_id in all_entity_ids:
                if entity_id not in corruption_map:
                    num_spaces = random.randint(1, 3)
                    spaces = " " * num_spaces
                    pos = random.choice(["start", "end", "both"])
                    if pos == "start":
                        corruption_map[entity_id] = spaces + entity_id
                    elif pos == "end":
                        corruption_map[entity_id] = entity_id + spaces
                    else:
                        corruption_map[entity_id] = spaces + entity_id + spaces
        
        self._find_and_corrupt_related_records(corruption_map, tracker, "Q1")
        
        return selected_orders # NEW: Return the list

    def _apply_q2_order_level(self, tracker: ErrorTracker):
        """Q2: Apply character deletion to 12% of orders and all their related records."""
        orders = self._get_all_orders()
        if not orders:
            print("No orders available for Q2 corruption.")
            return []
        num_orders_to_corrupt = max(1, int(len(orders) * 0.12))
        selected_orders = random.sample(orders, num_orders_to_corrupt)
        
        print(f"Q2: Corrupting {num_orders_to_corrupt}/{len(orders)} orders: {selected_orders}")
        
        corruption_map = {}
        for order_id in selected_orders:
            entities = self._get_order_related_entities(order_id)
            all_entity_ids = set().union(*entities.values())
            
            for entity_id in all_entity_ids:
                if entity_id not in corruption_map:
                    if len(entity_id) < 2:
                        corruption_map[entity_id] = entity_id
                    else:
                        pos_to_remove = random.randint(1, len(entity_id) - 1)
                        corruption_map[entity_id] = entity_id[:pos_to_remove] + entity_id[pos_to_remove + 1:]

        self._find_and_corrupt_related_records(corruption_map, tracker, "Q2")

        return selected_orders # NEW: Return the list

    def _apply_q3_gear_order_deletion(self, tracker: ErrorTracker):
        """Q3: Delete 5-8% of gear→order relationships."""
        rel_df = self.datasets.get("relationship_data")
        if rel_df is None or rel_df.empty:
            print("No relationship data available for Q3 corruption.")
            return []

        # Find all gear→order relationships
        gear_order_mask = (
            rel_df["child"].str.startswith("3DOR", na=False) &
            rel_df["parent"].str.startswith("ORBOX", na=False)
        )
        gear_order_relationships = rel_df[gear_order_mask]
        
        if gear_order_relationships.empty:
            print("No gear→order relationships found for Q3 corruption.")
            return []

        # Select 5-8% of these relationships to delete
        deletion_rate = random.uniform(0.05, 0.08)
        num_to_delete = max(1, int(len(gear_order_relationships) * deletion_rate))
        
        indices_to_delete = gear_order_relationships.sample(n=num_to_delete).index.tolist()
        
        print(f"Q3: Deleting {num_to_delete}/{len(gear_order_relationships)} gear→order relationships ({deletion_rate:.1%})")

        # NEW: Get the set of unique order IDs affected by the deletion
        affected_orders = set(rel_df.loc[indices_to_delete]["parent"].unique())
        
        for idx in indices_to_delete:
            removed_record = rel_df.loc[idx].to_dict()
            gear_id = removed_record["child"]
            order_id = removed_record["parent"]
            
            tracker.log_q3_missing_record(
                idx,
                json.dumps(removed_record),
                f"gear_to_order: {gear_id} → {order_id}",
                "HIGH"  # High impact as it breaks ARC traceability
            )

        # Remove the selected relationships
        self.datasets["relationship_data"] = rel_df.drop(indices_to_delete).reset_index(drop=True)

        return list(affected_orders) # NEW: Return the list of affected orders

    def save_corrupted_data(self, corrupted_data: Dict[str, pd.DataFrame], 
                          error_tracker: ErrorTracker, quality_condition: str):
        """Save the corrupted data and error logs."""
        output_dir = f"data/experimental_datasets/{quality_condition}_dataset"
        os.makedirs(output_dir, exist_ok=True)

        for name, df in corrupted_data.items():
            # Save corrupted data file
            corrupted_path = os.path.join(output_dir, f"{name}_{quality_condition}.csv")
            df.to_csv(corrupted_path, index=False)
            print(f"Saved corrupted data: {corrupted_path}")

        # Save error log
        log_path = os.path.join(output_dir, f"all_tables_{quality_condition}_errors.csv")
        error_tracker.save_log(log_path, quality_condition)