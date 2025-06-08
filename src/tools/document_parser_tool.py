import pypdf
import os
from typing import Dict, Any
from .base_tool import BaseTool


class DocumentParserTool(BaseTool):
    """Tool for parsing FAA 8130-3 certificate PDFs."""

    def __init__(self, datasets: Dict[str, Any]):
        super().__init__(datasets)
        self.doc_path = "data/generated_documents/certificates"

    def run(self, order_id: str) -> Dict[str, Any]:
        """
        Finds and parses the FAA certificate for a given order ID.

        Args:
            order_id: The order ID (e.g., 'ORBOX0014').

        Returns:
            A dictionary of the form fields extracted from the PDF.
        """
        pdf_filename = f"ARC-{order_id}.pdf"
        pdf_full_path = os.path.join(self.doc_path, pdf_filename)

        if not os.path.exists(pdf_full_path):
            return {"error": f"Certificate not found: {pdf_filename}"}

        try:
            reader = pypdf.PdfReader(pdf_full_path, strict=False)
            fields = reader.get_fields()
            if not fields:
                return {"error": f"No form fields found in {pdf_filename}."}

            # Extract text values from the field objects
            extracted_data = {
                key: value.get("/V", "N/A") for key, value in fields.items()
            }
            return extracted_data
        except Exception as e:
            return {"error": f"Failed to parse PDF {pdf_filename}: {e}"}