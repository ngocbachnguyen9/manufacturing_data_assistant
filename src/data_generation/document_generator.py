import pypdf
import os
from pypdf.generic import BooleanObject, NameObject, IndirectObject
from typing import Dict
from mailmerge import MailMerge # Add this import

class FAACertificateGenerator:
    """Generates FAA 8130-3 certificates by filling a PDF template."""

    def __init__(
        self,
        template_path: str = "data/document_templates/faa_8130_blank.pdf",
    ):
        self.template_path = template_path
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(
                f"Template PDF not found at {self.template_path}"
            )

    def _set_need_appearances_writer(self, writer: pypdf.PdfWriter):
        """Ensures form field values are visible in the output PDF."""
        try:
            catalog = writer._root_object
            if "/AcroForm" not in catalog:
                writer._root_object.update(
                    {
                        NameObject("/AcroForm"): IndirectObject(
                            len(writer._objects), 0, writer
                        )
                    }
                )
            need_appearances = NameObject("/NeedAppearances")
            writer._root_object["/AcroForm"][need_appearances] = BooleanObject(
                True
            )
        except Exception as e:
            print(f"Warning: Could not set /NeedAppearances: {repr(e)}")
        return writer

    def generate_certificate(
        self, field_data: Dict[str, str], output_path: str
    ):
        """
        Generates a single certificate PDF.

        Args:
            field_data: A dictionary mapping form field names to their values.
            output_path: The path to save the generated PDF.
        """
        reader = pypdf.PdfReader(open(self.template_path, "rb"), strict=False)
        writer = pypdf.PdfWriter()
        writer.append(reader)

        writer.update_page_form_field_values(
            writer.pages[0], field_data, auto_regenerate=False
        )

        # Set /NeedAppearances to ensure visibility
        self._set_need_appearances_writer(writer)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as output_stream:
            writer.write(output_stream)

# NEW CLASS
class PackingListGenerator:
    """Generates fillable .docx Packing Lists from a template."""

    def __init__(
        self,
        template_path: str = "data/document_templates/packing_list_template.docx",
    ):
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Word template not found: {template_path}")
        self.template_path = template_path

    def generate_packing_list(self, field_data: dict, output_path: str):
        """Fills a .docx template using mail merge and saves the new document."""
        document = MailMerge(self.template_path)
        # The merge method replaces {{FieldName}} with the value from the dict
        document.merge(**field_data)
        document.write(output_path)