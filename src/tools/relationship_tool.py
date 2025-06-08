from typing import Dict, Any, List
import pandas as pd
from .base_tool import BaseTool


class RelationshipTool(BaseTool):
    """Tool for traversing parent-child relationships."""

    def run(self, entity_id: str) -> List[Dict]:
        """
        Finds all direct parent/child relationships for a given entity ID.

        Args:
            entity_id: The ID to find relationships for.

        Returns:
            A list of dictionaries representing the relationships.
        """
        df = self.datasets.get("relationship_data")
        if df is None or df.empty:
            return [{"error": "Relationship data not available."}]

        result_df = df[
            (df["parent"] == entity_id) | (df["child"] == entity_id)
        ]
        if result_df.empty:
            return [
                {"message": f"No relationships found for ID {entity_id}"}
            ]

        return result_df.to_dict("records")