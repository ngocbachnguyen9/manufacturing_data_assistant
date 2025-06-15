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

    def _check_timeline_tools_used(self, context: Dict) -> bool:
        """
        Check if any timeline-related tools were used in the execution.
        Timeline validation is needed for tasks that involve:
        - Machine logs (printing timeline)
        - Location queries (warehouse arrival timeline)
        - Document parsing (certificate dates)
        """
        # Check if timeline-related tools were actually called by looking for their outputs
        timeline_tool_indicators = [
            # Machine log tool outputs
            "event_type", "Machine", "print_start", "print_end",
            # Location query tool outputs
            "location", "_time", "timestamp", "scan_time", "Parts Warehouse",
            # Document parser tool outputs
            "certificate_date", "arc_date", "document_date", "source_document",
            # Error messages from timeline tools
            "machine_log_tool", "location_query_tool", "document_parser_tool"
        ]

        # Look for evidence that timeline-related tools were called
        for key, value in context.items():
            # Check context key names for timeline tool usage
            key_lower = key.lower()
            if any(indicator in key_lower for indicator in ["machine", "location", "document", "certificate", "warehouse"]):
                return True

            # Check the actual data content
            if isinstance(value, list) and value:
                for item in value:
                    if isinstance(item, dict):
                        # Check for timeline tool output fields
                        if any(indicator in item for indicator in timeline_tool_indicators):
                            return True
                        # Check error messages for timeline tool usage
                        if "error" in item and isinstance(item["error"], str):
                            error_lower = item["error"].lower()
                            if any(tool in error_lower for tool in ["machine", "location", "document", "certificate"]):
                                return True

            # Check if value is a dict (single result from timeline tool)
            elif isinstance(value, dict):
                if any(indicator in value for indicator in timeline_tool_indicators):
                    return True
                if "error" in value and isinstance(value["error"], str):
                    error_lower = value["error"].lower()
                    if any(tool in error_lower for tool in ["machine", "location", "document", "certificate"]):
                        return True

        return False

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
        Only performs timeline validation if timeline-related tools were used.
        """
        # Check if timeline validation is needed based on tools used
        timeline_tools_used = self._check_timeline_tools_used(context)

        if not timeline_tools_used:
            print(f"  - [ReconciliationAgent] Skipping timeline validation - no timeline-related tools used")
            return

        # Collect all relevant data from context
        machine_logs = []
        location_scans = []
        relationship_data = []

        # Look for data based on the actual content structure, not just key names
        for key, value in context.items():
            if isinstance(value, list) and value:
                # Check if this looks like machine log data
                if any(isinstance(item, dict) and item.get("event_type") in ["PRINT_START", "PRINT_END"] for item in value):
                    machine_logs.extend(value)
                # Check if this looks like location scan data
                elif any(isinstance(item, dict) and "location" in item and "_time" in item for item in value):
                    location_scans.extend(value)
                # Check if this looks like relationship data
                elif any(isinstance(item, dict) and ("child" in item or "parent" in item or "_from" in item) for item in value):
                    relationship_data.extend(value)

        # Handle partial data scenarios only if timeline validation is needed
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