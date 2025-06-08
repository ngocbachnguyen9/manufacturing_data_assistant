from typing import Dict, Any, List
import pandas as pd
from .base_tool import BaseTool


class WorkerDataTool(BaseTool):
    """Tool for querying worker activity data with optional fuzzy matching."""

    def run(self, worker_id: str, **kwargs) -> List[Dict]:
        """
        Finds all activity scans for a given worker ID.

        Args:
            worker_id: The RFID of the worker to track.
            **kwargs: Can accept 'fuzzy_enabled' and 'threshold'.
        """
        df = self.datasets.get("worker_data")
        if df is None or df.empty:
            return [{"error": "Worker data not available."}]

        # Worker IDs are numeric and less prone to fuzzy errors,
        # but we support the option if needed.
        result_df = df[df["_value"] == worker_id]

        if result_df.empty:
            return [{"message": f"No data found for worker ID {worker_id}"}]

        return result_df.to_dict("records")