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
        Validates that warehouse arrival occurs after print end time for all gears.
        Uses relationship data to link gears to print jobs and machine logs.
        """
        # Collect all relevant data from context
        machine_logs = []
        location_scans = []
        relationship_data = []
        for key, value in context.items():
            if "machine_log" in key:
                machine_logs.extend(value)
            if "location_query" in key:
                location_scans.extend(value)
            if "relationship_query" in key:
                relationship_data.extend(value)

        # Validate we have all required data
        if not machine_logs or not location_scans or not relationship_data:
            print("  - [ReconciliationAgent] Insufficient data for timeline validation")
            return

        # Build gear-to-job mapping from relationship data
        gear_to_job = {}
        for rel in relationship_data:
            if rel.get("type") == "printed_by" and rel["_from"].startswith("barcode/"):
                gear_id = rel["_from"].split("/")[1]
                job_id = rel["_to"]
                gear_to_job[gear_id] = job_id

        # Find all warehouse entry events for gears
        warehouse_entries = {}
        for scan in location_scans:
            scan_value = str(scan.get("_value", ""))
            if scan_value.startswith("3DOR") and scan.get("location") == "Parts Warehouse":
                try:
                    warehouse_entries[scan_value] = pd.to_datetime(scan["_time"])
                except (KeyError, ValueError):
                    continue

        # Find print end times from machine logs
        print_end_times = {}
        for log in machine_logs:
            if log.get("event_type") == "PRINT_END":
                try:
                    print_end_times[log["job_id"]] = pd.to_datetime(log["_time"])
                except (KeyError, ValueError):
                    continue

        # Validate timelines for all gears
        for gear_id, warehouse_time in warehouse_entries.items():
            job_id = gear_to_job.get(gear_id)
            if not job_id:
                continue  # No associated print job
            
            print_end_time = print_end_times.get(job_id)
            if not print_end_time:
                continue  # No PRINT_END event found
            
            # Validate timeline consistency
            if warehouse_time < print_end_time:
                issue = (f"Timeline inconsistency for {gear_id}: "
                         f"Arrived at warehouse at {warehouse_time} "
                         f"before print finished at {print_end_time}.")
                summary["issues_found"].append(issue)
                summary["confidence"] -= 0.4