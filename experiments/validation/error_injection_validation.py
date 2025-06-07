import pandas as pd
import os


class InjectionValidator:
    """Validates the accuracy of order-level error injection process."""

    def __init__(self, base_path: str = "data/experimental_datasets"):
        self.base_path = base_path

    def validate_all_conditions(self):
        """Run validation for all corrupted quality conditions."""
        for qc in ["Q1", "Q2", "Q3"]:
            print(f"***REMOVED***n--- Validating Condition: {qc} (Order-Level) ---")
            qc_path = os.path.join(self.base_path, f"{qc}_dataset")
            if not os.path.exists(qc_path):
                print(f"Path not found: {qc_path}. Skipping.")
                continue

            self.validate_order_level_corruption(qc_path, qc)

    def validate_order_level_corruption(self, qc_path: str, qc: str):
        """Validate order-level corruption rates and consistency."""
        # Load error log
        error_log_path = os.path.join(qc_path, f"all_tables_{qc}_errors.csv")
        if not os.path.exists(error_log_path):
            print(f"  - Error log not found: {error_log_path}")
            return

        error_df = pd.read_csv(error_log_path)
        
        if qc in ["Q1", "Q2"]:
            self._validate_order_corruption_rate(qc_path, error_df, qc)
            self._validate_cross_table_consistency(qc_path, error_df, qc)
        elif qc == "Q3":
            self._validate_relationship_deletion_rate(qc_path, error_df, qc)

    def _validate_order_corruption_rate(self, qc_path: str, error_df: pd.DataFrame, qc: str):
        """Validate that corruption rate matches expected order-level percentage."""
        baseline_rel_path = os.path.join(self.base_path, "Q0_baseline", "relationship_data.csv")
        if not os.path.exists(baseline_rel_path):
            print(f"  - Baseline relationship data not found")
            return

        baseline_rel_df = pd.read_csv(baseline_rel_path)
        total_orders = len(baseline_rel_df[baseline_rel_df["parent"].str.startswith("ORBOX", na=False)]["parent"].unique())
        
        # Count unique ORIGINAL orders from the error log
        if error_df.empty:
            corrupted_orders_count = 0
        else:
            # Correctly count only the original order IDs that were targeted
            corrupted_order_ids = set()
            for original_id in error_df["original"].unique():
                if isinstance(original_id, str) and original_id.startswith("ORBOX"):
                    corrupted_order_ids.add(original_id)
            corrupted_orders_count = len(corrupted_order_ids)

        actual_rate = corrupted_orders_count / total_orders if total_orders > 0 else 0
        
        expected_rates = {"Q1": 0.15, "Q2": 0.12}
        expected = expected_rates.get(qc, 0)
        
        tolerance = 0.05
        is_valid = abs(actual_rate - expected) <= tolerance
        
        print(f"  - Order Corruption Rate: {'PASS' if is_valid else 'FAIL'}")
        print(f"    Actual: {actual_rate:.1%} ({corrupted_orders_count}/{total_orders} orders)")
        print(f"    Expected: {expected:.1%} ± {tolerance:.1%}")

    def _validate_cross_table_consistency(self, qc_path: str, error_df: pd.DataFrame, qc: str):
        """Validate that the same order IDs are corrupted consistently across tables."""
        if error_df.empty:
            print("  - Cross-table Consistency: PASS (no corruptions to check)")
            return

        # Group corruptions by original order ID
        order_corruptions = {}
        for _, row in error_df.iterrows():
            original = row.get("original", "")
            corrupted = row.get("corrupted", "")
            table_col = row.get("table_column", "")
            
            if original.startswith("ORBOX"):
                if original not in order_corruptions:
                    order_corruptions[original] = {"corrupted_version": corrupted, "tables": set()}
                order_corruptions[original]["tables"].add(table_col)

        # Check consistency
        consistent = True
        for order_id, corruption_info in order_corruptions.items():
            if len(corruption_info["tables"]) > 1:
                # This order appears in multiple tables - check if corruption is consistent
                unique_corruptions = set()
                for _, row in error_df.iterrows():
                    if row.get("original") == order_id:
                        unique_corruptions.add(row.get("corrupted"))
                
                if len(unique_corruptions) > 1:
                    print(f"    Inconsistent corruption for {order_id}: {unique_corruptions}")
                    consistent = False

        print(f"  - Cross-table Consistency: {'PASS' if consistent else 'FAIL'}")

    def _validate_relationship_deletion_rate(self, qc_path: str, error_df: pd.DataFrame, qc: str):
        """Validate Q3 gear→order relationship deletion rate."""
        # Load original and corrupted relationship data
        baseline_rel_path = os.path.join(self.base_path, "Q0_baseline", "relationship_data.csv")
        corrupted_rel_path = os.path.join(qc_path, "relationship_data_Q3.csv")
        
        if not os.path.exists(baseline_rel_path) or not os.path.exists(corrupted_rel_path):
            print("  - Relationship files not found for Q3 validation")
            return

        baseline_rel_df = pd.read_csv(baseline_rel_path)
        corrupted_rel_df = pd.read_csv(corrupted_rel_path)
        
        # Count gear→order relationships
        baseline_gear_order = len(baseline_rel_df[
            (baseline_rel_df["child"].str.startswith("3DOR", na=False)) &
            (baseline_rel_df["parent"].str.startswith("ORBOX", na=False))
        ])
        
        corrupted_gear_order = len(corrupted_rel_df[
            (corrupted_rel_df["child"].str.startswith("3DOR", na=False)) &
            (corrupted_rel_df["parent"].str.startswith("ORBOX", na=False))
        ])
        
        deleted_relationships = baseline_gear_order - corrupted_gear_order
        actual_rate = deleted_relationships / baseline_gear_order if baseline_gear_order > 0 else 0
        
        # Expected range: 5-8%
        is_valid = 0.05 <= actual_rate <= 0.08
        
        print(f"  - Relationship Deletion Rate: {'PASS' if is_valid else 'FAIL'}")
        print(f"    Actual: {actual_rate:.1%} ({deleted_relationships}/{baseline_gear_order} relationships)")
        print(f"    Expected: 5.0% - 8.0%")
        print(f"    Error log entries: {len(error_df)}")