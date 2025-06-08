from typing import Dict, Any, List
import re


class ReconciliationAgent:
    """
    Validates data consistency, identifies discrepancies, and assesses data quality.
    """

    def __init__(self, tools: Dict[str, Any]):
        self.tools = tools
        self.validator_tool = self.tools.get("barcode_validator_tool")
        if not self.validator_tool:
            print(
                "Warning: ReconciliationAgent initialized without a barcode_validator_tool."
            )

    def reconcile(self, retrieved_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs cross-validation and consistency checks on retrieved data.
        """
        print("  - [ReconciliationAgent] Starting data reconciliation...")
        summary = {
            "issues_found": [],
            "validated_data": retrieved_data,
            "confidence": 1.0,
        }

        self._check_for_tool_errors(retrieved_data, summary)
        self._validate_barcodes_in_results(retrieved_data, summary)
        # Placeholder for future cross-system temporal checks
        # self._cross_validate_timestamps(retrieved_data, summary)

        print(
            f"  - [ReconciliationAgent] Reconciliation complete. Confidence: {summary['confidence']:.2f}"
        )
        return summary

    def _check_for_tool_errors(
        self, data: Dict[str, Any], summary: Dict
    ):
        """Checks for explicit 'error' keys returned by tools."""
        for tool_step, result in data.items():
            if isinstance(result, list) and result and "error" in result[0]:
                issue = f"Error from {tool_step}: {result[0]['error']}"
                summary["issues_found"].append(issue)
                summary["confidence"] -= 0.5  # High impact for tool failure

    def _validate_barcodes_in_results(
        self, data: Dict[str, Any], summary: Dict
    ):
        """Finds and validates all potential barcodes in the retrieved data."""
        if not self.validator_tool:
            return

        all_ids_found = set()
        # Regex to find potential IDs (alphanumeric, may contain '_')
        id_pattern = re.compile(r"***REMOVED***b([A-Z0-9_]+)***REMOVED***b")

        # Recursively find all potential IDs in the results
        def find_ids(value):
            if isinstance(value, str):
                all_ids_found.update(id_pattern.findall(value))
            elif isinstance(value, list):
                for item in value:
                    find_ids(item)
            elif isinstance(value, dict):
                for k, v in value.items():
                    find_ids(v)

        find_ids(data)

        for entity_id in all_ids_found:
            # Only validate things that look like our specific formats
            if entity_id.startswith(("3DOR", "ORBOX", "Printer_")):
                validation_result = self.validator_tool.run(entity_id)
                if not validation_result["is_valid"]:
                    issue = f"Invalid barcode format for ID '{entity_id}'."
                    summary["issues_found"].append(issue)
                    summary["confidence"] -= 0.1  # Lower confidence for bad data