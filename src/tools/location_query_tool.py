from typing import Dict, Any, List
import pandas as pd
from .base_tool import BaseTool
from thefuzz import process


class LocationQueryTool(BaseTool):
    """Tool for querying location tracking data with optional fuzzy matching."""

    def run(self, entity_id: str, **kwargs) -> List[Dict]:
        """
        Finds location events for an entity ID, with optional fuzzy matching.

        Args:
            entity_id: The barcode/ID of the entity to track.
            **kwargs: Expects 'fuzzy_enabled' (bool) and 'threshold' (float).
        """
        df = self.datasets.get("location_data")
        if df is None or df.empty:
            return [{"error": "Location data not available."}]

        # Perform an exact match first for speed and accuracy
        result_df = df[df["_value"] == entity_id]

        # If no exact match is found and fuzzy search is enabled, try it
        if result_df.empty and kwargs.get("fuzzy_enabled", False):
            print(
                f"    - No exact match for '{entity_id}', trying fuzzy search..."
            )
            threshold = kwargs.get("threshold", 0.8) * 100
            choices = df["_value"].unique()

            # Find the best match above the score cutoff
            best_match = process.extractOne(
                entity_id, choices, score_cutoff=threshold
            )

            if best_match:
                matched_id = best_match[0]
                print(
                    f"    - Found fuzzy match: '{matched_id}' with score {best_match[1]}"
                )
                # Filter the DataFrame using the matched ID
                result_df = df[df["_value"] == matched_id]

        if result_df.empty:
            return [{"error": f"No location data found for ID '{entity_id}'"}]

        return result_df.to_dict("records")