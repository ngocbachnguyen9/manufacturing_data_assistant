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
        self.critical_issues = [
            "missing worker id",
            "incomplete dataset",
            "data not found"
        ]

    def reconcile(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs cross-validation on the entire data context.
        """
        print("  - [ReconciliationAgent] Starting data reconciliation...")
        summary = {
            "issues_found": [],
            "validated_data": context,
            "confidence": 1.0,
            "critical_issue": False,
        }

        self._check_for_tool_errors(context, summary)
        self._cross_validate_gear_timeline(context, summary)

        # Apply confidence floor at 0.0
        summary["confidence"] = max(0.0, summary["confidence"])
        
        # Check for critical issues
        for issue in summary["issues_found"]:
            if any(keyword in issue.lower() for keyword in self.critical_issues):
                summary["critical_issue"] = True
                break

        print(
            f"  - [ReconciliationAgent] Reconciliation complete. Confidence: {summary['confidence']:.2f}, Critical issue: {summary['critical_issue']}"
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

        # Handle partial data scenarios
        missing_data = []
        if not machine_logs:
            missing_data.append("machine logs")
        if not location_scans:
            missing_data.append("location scans")
        if not relationship_data:
            missing_data.append("relationship data")
            
        if missing_data:
            msg = f"Insufficient data for timeline validation. Missing: {', '.join(missing_data)}"
            print(f"  - [ReconciliationAgent] {msg}")
            summary["issues_found"].append(msg)
            summary["confidence"] -= 0.1 * len(missing_data)
            return

        # Build gear-to-job mapping from relationship data with error handling
        gear_to_job = {}
        for rel in relationship_data:
            try:
                if rel.get("type") == "printed_by" and rel["_from"].startswith("barcode/"):
                    gear_id = rel["_from"].split("/")[1]
                    job_id = rel["_to"]
                    gear_to_job[gear_id] = job_id
            except KeyError as e:
                print(f"  - Warning: Missing key in relationship data: {e}")

        # Find warehouse entries with error handling
        warehouse_entries = {}
        for scan in location_scans:
            try:
                scan_value = str(scan.get("_value", ""))
                if scan_value.startswith("3DOR") and scan.get("location") == "Parts Warehouse":
                    warehouse_entries[scan_value] = pd.to_datetime(scan["_time"])
            except (KeyError, ValueError) as e:
                print(f"  - Warning: Invalid location scan entry: {e}")

        # Find print end times with error handling
        print_end_times = {}
        for log in machine_logs:
            try:
                if log.get("event_type") == "PRINT_END":
                    print_end_times[log["job_id"]] = pd.to_datetime(log["_time"])
            except (KeyError, ValueError) as e:
                print(f"  - Warning: Invalid machine log entry: {e}")

        # Validate timelines for all gears
        for gear_id, warehouse_time in warehouse_entries.items():
            job_id = gear_to_job.get(gear_id)
            if not job_id:
                issue = f"Gear {gear_id} has warehouse entry but no associated print job"
                summary["issues_found"].append(issue)
                summary["confidence"] -= 0.2
                continue
            
            print_end_time = print_end_times.get(job_id)
            if not print_end_time:
                issue = f"Gear {gear_id} has warehouse entry but no PRINT_END event for job {job_id}"
                summary["issues_found"].append(issue)
                summary["confidence"] -= 0.3
                continue
            
            # Validate timeline consistency
            if warehouse_time < print_end_time:
                issue = (f"Timeline inconsistency for {gear_id}: "
                         f"Arrived at warehouse at {warehouse_time} "
                         f"before print finished at {print_end_time}.")
                summary["issues_found"].append(issue)
                summary["confidence"] -= 0.4