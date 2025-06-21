import pypdf
import os
from typing import Dict, Any, List
from .base_tool import BaseTool
from thefuzz import process


class DocumentParserTool(BaseTool):
    """Tool for parsing FAA 8130-3 certificate PDFs with fuzzy filename matching."""

    def __init__(self, datasets: Dict[str, Any]):
        super().__init__(datasets)
        self.doc_path = "data/generated_documents/certificates"
        self.available_docs = self._get_available_docs()

    def _get_available_docs(self) -> List[str]:
        """Get a list of all available certificate filenames."""
        if not os.path.exists(self.doc_path):
            return []
        return [f for f in os.listdir(self.doc_path) if f.endswith(".pdf")]

    def run(self, order_id: str, **kwargs) -> Dict[str, Any]:
        """
        Finds and parses the FAA certificate for a given order ID.

        Args:
            order_id: The order ID (e.g., 'ORBOX0014').
            **kwargs: Can accept 'fuzzy_enabled' and 'threshold'.
        """
        pdf_filename = f"ARC-{order_id}.pdf"
        pdf_full_path = os.path.join(self.doc_path, pdf_filename)

        if not os.path.exists(pdf_full_path):
            fuzzy_enabled = kwargs.get("fuzzy_enabled", False)
            # Fast exit if fuzzy is disabled
            if not fuzzy_enabled:
                return {
                    "error": f"Certificate not found: {pdf_filename}",
                    "error_type": "document_not_found",
                    "order_id": order_id,
                    "attempted_filename": pdf_filename,
                    "fallback_available": bool(self.available_docs)
                }
            if fuzzy_enabled and self.available_docs:
                print(
                    f"    - Exact cert '{pdf_filename}' not found, trying fuzzy search..."
                )
                threshold = kwargs.get("threshold", 0.8) * 100
                best_match = process.extractOne(
                    pdf_filename, self.available_docs, score_cutoff=threshold
                )
                if best_match:
                    pdf_filename = best_match[0]
                    pdf_full_path = os.path.join(self.doc_path, pdf_filename)
                    print(f"    - Found fuzzy match: '{pdf_filename}'")
                else:
                    return {
                        "error": f"Certificate not found for {order_id}, no fuzzy match.",
                        "error_type": "document_not_found_fuzzy_failed",
                        "order_id": order_id,
                        "attempted_filename": pdf_filename,
                        "available_docs": len(self.available_docs)
                    }
            else:
                return {
                    "error": f"Certificate not found: {pdf_filename}",
                    "error_type": "document_not_found",
                    "order_id": order_id,
                    "attempted_filename": pdf_filename,
                    "fallback_available": False
                }

        try:
            reader = pypdf.PdfReader(pdf_full_path, strict=False)
            fields = reader.get_fields()
            if not fields:
                return {
                    "error": f"No form fields found in {pdf_filename}.",
                    "error_type": "no_form_fields",
                    "order_id": order_id,
                    "document_found": True,
                    "source_document": pdf_filename
                }

            extracted_data = {
                key: value.get("/V", "N/A") for key, value in fields.items()
            }
            extracted_data["source_document"] = pdf_filename
            extracted_data["order_id"] = order_id  # Include order_id for fallback use
            return extracted_data
        except Exception as e:
            return {
                "error": f"Failed to parse PDF {pdf_filename}: {e}",
                "error_type": "pdf_parse_error",
                "order_id": order_id,
                "document_found": True,
                "source_document": pdf_filename
            }