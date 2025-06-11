import os
import pandas as pd
from src.utils.data_loader import DataLoader
from src.data_generation.document_generator import FAACertificateGenerator
from src.data_generation.document_generator import PackingListGenerator

class ManufacturingEnvironment:
    """
    Sets up the baseline (Q0) manufacturing environment, including data and
    documents.
    """

    def __init__(
        self,
        base_data_path: str = "data/manufacturing_base",
        output_path: str = "data/experimental_datasets/Q0_baseline",
        faa_doc_path: str = "data/generated_documents/certificates",
        # NEW: Add a path for packing lists
        packing_list_doc_path: str = "data/generated_documents/packing_lists",
    ):
        self.data_loader = DataLoader(base_path=base_data_path)
        self.output_path = output_path
        self.faa_doc_output_path = faa_doc_path
        self.packing_list_output_path = packing_list_doc_path # NEW
        self.faa_generator = FAACertificateGenerator()
        self.packing_list_generator = PackingListGenerator() # NEW
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.faa_doc_output_path, exist_ok=True)
        os.makedirs(self.packing_list_output_path, exist_ok=True) # NEW

    def setup_baseline_environment(self):
        """
        Loads base data, saves it as the Q0 baseline, and generates
        related documents.
        """
        print("--- Setting up Q0 Baseline Environment ---")
        base_data = self.data_loader.load_base_data()

        # 1. Save baseline data to Q0 directory
        for name, df in base_data.items():
            if not df.empty:
                output_file = os.path.join(self.output_path, f"{name}.csv")
                df.to_csv(output_file, index=False)
                print(f"Saved Q0 baseline file: {output_file}")

        # 2. Generate FAA certificates for each order
        self._generate_documents(base_data)

        print("--- Q0 Baseline Environment Setup Complete ---")

    def _generate_documents(self, data: dict):
        """
        Generates FAA certificates based on relationship and location data.
        """
        print("***REMOVED***n--- Generating FAA Certificates ---")
        rel_df = data.get("relationship_data")
        if rel_df is None or rel_df.empty:
            print("Warning: relationship_data not found. Cannot generate docs.")
            return

        # Find all unique orders (assuming they are parents of gears)
        gear_rels = rel_df[rel_df["child"].str.startswith("3DOR", na=False)]
        unique_orders = gear_rels["parent"].unique()

        for order_id in unique_orders:
            if not order_id.startswith("ORBOX"):
                continue

            # Create dummy data for the certificate form
            field_data = {
                "3  Form Tracking Number": f"AA3DPR200{order_id}",
                "4 Organization Name and Address": "3D Printing Factory, IfM, Cambridge, UK",
                "5  Work OrderContractInvoice Number": order_id,
                "6  ItemRow1": "Gears",
                "12  Remarks": "Printed gears using 3D printers. Followed a set design for 3DGR01.",
                "13c ApprovalAuthorization No": "10003001",
                "14c  ApprovalCertificate No": "10003002",
                "13d  Name Typed or Printed": "Worker 1",
                "13e Date ddmmmyyyy": "28/10/2024",
                "14d  Name Typed or Printed": "Worker 2",
                "7  Description": "3D Printed gear pair",
                "8  Part Number": "3DGR01",
                "9  Quantity": "10",
                "10 Serial Number": order_id,
                "11 StatusWork": "3D Printing",
                "14e Date": "28/10/2024",
            }

            output_file = os.path.join(self.faa_doc_output_path, f"ARC-{order_id}.pdf")
            try:
                self.faa_generator.generate_certificate(field_data, output_file)
                print(f"Generated certificate for Order ID: {order_id}")
            except Exception as e:
                print(f"Error generating certificate for {order_id}: {e}")
            
            # --- NEW: Packing List Generation Logic ---
            packing_list_id = f"PL{int(order_id.replace('ORBOX', '')) + 1000}"
            packing_list_fields = {
                "PackingListID": packing_list_id,
                "OrderNumber": order_id, # This is the crucial link
                "Quantity": str(len(gear_rels[gear_rels["parent"] == order_id])),
                "Description": "3D Printed Gears",
            }
            packing_list_output_file = os.path.join(
                self.packing_list_output_path, f"PackingList-{packing_list_id}.docx"
            )
            self.packing_list_generator.generate_packing_list(
                packing_list_fields, packing_list_output_file
            )
            print(f"Generated Packing List {packing_list_id} for Order ID: {order_id}")