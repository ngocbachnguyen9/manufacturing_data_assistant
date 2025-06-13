# In src/tools/packing_list_parser_tool.py

import os
import re
from .base_tool import BaseTool
from typing import Dict, Any
from pypdf import PdfReader  # Updated import for PDF parsing

class PackingListParserTool(BaseTool):
    """Tool for parsing PDF Packing Lists to find the associated Order ID."""

    def __init__(self, datasets: Dict[str, Any]):
        super().__init__(datasets)
        self.doc_path = "data/generated_documents/packing_lists"

    def run(self, packing_list_id: str, **kwargs) -> Dict[str, str]:
        """
        Finds a packing list by its ID and extracts the OrderNumber.
        """
        # Updated to use PDF extension
        pdf_filename = f"PackingList-{packing_list_id}.pdf"
        pdf_full_path = os.path.join(self.doc_path, pdf_filename)

        if not os.path.exists(pdf_full_path):
            return {"error": f"Packing List not found: {pdf_filename}"}

        try:
            # Use PyPDF2 to read the PDF
            reader = PdfReader(pdf_full_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            # Search for the order number pattern in the extracted text
            # We look for patterns like "Order Number: ORBOX0011"
            match = re.search(r'Order***REMOVED***s*Number:***REMOVED***s*(***REMOVED***S+)', text)
            if match:
                order_id = match.group(1).strip()
                return {"order_id": order_id, "source_document": pdf_filename}

            return {"error": f"Could not find 'Order Number' in {pdf_filename}."}
        except Exception as e:
            return {"error": f"Failed to parse PDF {pdf_filename}: {e}"}