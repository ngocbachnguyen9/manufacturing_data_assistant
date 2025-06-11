# In src/tools/packing_list_parser_tool.py

import pypdf
import os
from .base_tool import BaseTool
from typing import Dict, Any

class PackingListParserTool(BaseTool):
    """Tool for parsing Packing List PDFs to find the associated Order ID."""

    def __init__(self, datasets: Dict[str, Any]):
        super().__init__(datasets)
        self.doc_path = "data/generated_documents/packing_lists"

    def run(self, packing_list_id: str, **kwargs) -> Dict[str, str]:
        """
        Finds a packing list by its ID and extracts the OrderNumber field.

        Args:
            packing_list_id: The ID of the packing list (e.g., 'PL1011').

        Returns:
            A dictionary containing the extracted order_id.
        """
        pdf_filename = f"PackingList-{packing_list_id}.pdf"
        pdf_full_path = os.path.join(self.doc_path, pdf_filename)

        if not os.path.exists(pdf_full_path):
            return {"error": f"Packing List not found: {pdf_filename}"}

        try:
            reader = pypdf.PdfReader(pdf_full_path, strict=False)
            fields = reader.get_fields()
            if not fields:
                return {"error": f"No form fields found in {pdf_filename}."}

            # Assumes the PDF template has a field named 'OrderNumber'
            order_id = fields.get("OrderNumber", {}).get("/V", "UNKNOWN")

            if order_id == "UNKNOWN":
                return {"error": f"Could not extract OrderNumber from {pdf_filename}."}

            return {"order_id": order_id, "source_document": pdf_filename}
        except Exception as e:
            return {"error": f"Failed to parse PDF {pdf_filename}: {e}"}