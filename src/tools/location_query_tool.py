from typing import Dict, Any, List
import pandas as pd
from .base_tool import BaseTool


class LocationQueryTool(BaseTool):
    """Tool for querying location tracking data."""

    def run(self, entity_id: str) -> List[Dict]:
        """
        Finds all location scan events for a given entity ID.

        Args:
            entity_id: The barcode/ID of the entity to track.

        Returns:
            A list of dictionaries, where each is a location event.
        """
        df = self.datasets.get("location_data")
        if df is None or df.empty:
            return [{"error": "Location data not available."}]

        result_df = df[df["_value"] == entity_id]
        if result_df.empty:
            return [{"message": f"No location data found for ID {entity_id}"}]

        return result_df.to_dict("records")