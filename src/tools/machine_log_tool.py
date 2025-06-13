from typing import Dict, Any, List
import pandas as pd
from .base_tool import BaseTool


class MachineLogTool(BaseTool):
    """Tool for querying 3D printer machine logs."""

    # UPDATED: Signature now accepts **kwargs for consistency
    def run(self, printer_id: str, **kwargs) -> List[Dict]:
        """
        Finds all print jobs for a given printer ID.

        Args:
            printer_id: The ID of the printer (e.g., 'Printer_1').
            **kwargs: Included for signature consistency with the agent caller.
        """
        df = self.datasets.get("machine_log")
        if df is None or df.empty:
            return [{"error": "Machine log data not available."}]

        # This tool uses exact matching as printer IDs are fixed.
        result_df = df[df["Machine"] == printer_id]
        if result_df.empty:
            return [{"error": f"No logs found for printer {printer_id}"}]

        return result_df.to_dict("records")