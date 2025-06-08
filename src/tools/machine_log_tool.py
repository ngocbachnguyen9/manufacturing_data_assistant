from typing import Dict, Any, List
import pandas as pd
from .base_tool import BaseTool


class MachineLogTool(BaseTool):
    """Tool for querying 3D printer machine logs."""

    def run(self, printer_id: str) -> List[Dict]:
        """
        Finds all print jobs for a given printer ID.

        Args:
            printer_id: The ID of the printer (e.g., 'Printer_1').

        Returns:
            A list of dictionaries, where each is a print job log.
        """
        df = self.datasets.get("machine_log")
        if df is None or df.empty:
            return [{"error": "Machine log data not available."}]

        result_df = df[df["Machine"] == printer_id]
        if result_df.empty:
            return [{"message": f"No logs found for printer {printer_id}"}]

        return result_df.to_dict("records")