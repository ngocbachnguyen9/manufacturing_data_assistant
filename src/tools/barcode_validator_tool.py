import re
from typing import Dict, Any
from .base_tool import BaseTool
from src.utils.manufacturing_validator import ManufacturingValidator


class BarcodeValidatorTool(BaseTool):
    """Tool for validating barcode formats."""

    def __init__(self, datasets: Dict[str, Any]):
        super().__init__(datasets)
        self.validator = ManufacturingValidator()

    def run(self, entity_id: str, **kwargs) -> Dict[str, Any]:
        """
        Validates the format of a single entity ID.

        Args:
            entity_id: The ID string to validate.
            **kwargs: Included for signature consistency, but not used.
        """
        # This tool performs validation, not search, so fuzzy logic isn't applicable.
        for pattern_name, regex in self.validator.patterns.items():
            if re.match(regex, entity_id):
                return {
                    "id": entity_id,
                    "format": pattern_name,
                    "is_valid": True,
                }
        return {"id": entity_id, "format": "unknown", "is_valid": False}