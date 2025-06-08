from typing import Dict, Any, List
import pandas as pd
from .base_tool import BaseTool


class RelationshipTool(BaseTool):
    """Tool for traversing parent-child relationships with optional fuzzy matching."""

    def run(self, entity_id: str, **kwargs) -> List[Dict]:
        """
        Finds all direct relationships for an entity ID.

        Args:
            entity_id: The ID to find relationships for.
            **kwargs: Can accept 'fuzzy_enabled' and 'threshold'.
        """
        df = self.datasets.get("relationship_data")
        if df is None or df.empty:
            return [{"error": "Relationship data not available."}]

        # For relationships, an exact match is almost always required.
        # Fuzzy matching could create incorrect links.
        # We acknowledge the parameter but will stick to exact matches here.
        fuzzy_enabled = kwargs.get("fuzzy_enabled", False)
        if fuzzy_enabled:
            print(
                "  - Note: Fuzzy search requested for RelationshipTool, but using exact match to ensure integrity."
            )

        result_df = df[
            (df["parent"] == entity_id) | (df["child"] == entity_id)
        ]

        if result_df.empty:
            return [
                {"message": f"No relationships found for ID {entity_id}"}
            ]

        return result_df.to_dict("records")