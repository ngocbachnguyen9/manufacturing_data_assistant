from typing import Dict, Any, List
import re
import pandas as pd


class ReconciliationAgent:
    """
    Validates data consistency, identifies discrepancies, and assesses data quality.
    """

    def __init__(self, tools: Dict[str, Any]):
        self.tools = tools
        self.validator_tool = self.tools.get("barcode_validator_tool")

    def reconcile(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs cross-validation on the entire data context.
        """
        print("  - [ReconciliationAgent] Starting data reconciliation...")
        summary = {
            "issues_found": [],
            "validated_data": context,
            "confidence": 1.0,
        }

        self._check_for_tool_errors(context, summary)
        self._cross_validate_gear_timeline(context, summary)

        print(
            f"  - [ReconciliationAgent] Reconciliation complete. Confidence: {summary['confidence']:.2f}"
        )
        return summary

    def _check_for_tool_errors(self, context: Dict, summary: Dict):
        """Checks for explicit 'error' keys returned by tools."""
        for step_name, result in context.items():
            if isinstance(result, list) and result and "error" in result[0]:
                issue = f"Error from {step_name}: {result[0]['error']}"
                summary["issues_found"].append(issue)
                summary["confidence"] -= 0.5

    def _cross_validate_gear_timeline(self, context: Dict, summary: Dict):
        """
        For a gear, verify warehouse arrival is after its print end time.
        """
        # This logic assumes the plan stored machine logs and location data
        # in the context.
        machine_logs = []
        location_scans = []
        for key, value in context.items():
            if "machine_log" in key:
                machine_logs.extend(value)
            if "location_query" in key:
                location_scans.extend(value)

        if not machine_logs or not location_scans:
            return  # Not enough data to validate

        # Find a gear and its associated printer and timestamps
        gear_id = None
        for scan in location_scans:
            if str(scan.get("_value", "")).startswith("3DOR"):
                gear_id = scan["_value"]
                break
        if not gear_id:
            return

        # Find the warehouse entry time for this gear
        warehouse_entry_time = None
        for scan in location_scans:
            if (
                scan.get("_value") == gear_id
                and scan.get("location") == "Parts Warehouse"
            ):
                warehouse_entry_time = pd.to_datetime(scan["_time"])
                break

        # Find the print end time for this gear (requires traversing relationships)
        # This is a simplified example; a real one would use the relationship tool results.
        print(
            "  - [ReconciliationAgent] NOTE: Timeline validation is a simplified example."
        )
        # if warehouse_entry_time and print_end_time:
        #     if warehouse_entry_time < print_end_time:
        #         issue = f"Timeline inconsistency for {gear_id}: Arrived at warehouse BEFORE print finished."
        #         summary['issues_found'].append(issue)
        #         summary['confidence'] -= 0.4