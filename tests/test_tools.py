# tests/test_tools.py

import os
import sys
import pytest
import pandas as pd

# Allow imports like `from src.tools...`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.data_loader import DataLoader
from src.tools.machine_log_tool import MachineLogTool
from src.tools.worker_data_tool import WorkerDataTool
from src.tools.relationship_tool import RelationshipTool
from src.tools.location_query_tool import LocationQueryTool
from src.tools.document_parser_tool import DocumentParserTool
from src.tools.barcode_validator_tool import BarcodeValidatorTool
from src.utils.manufacturing_validator import ManufacturingValidator
from src.tools.packing_list_parser_tool import PackingListParserTool # NEW import


@pytest.fixture(scope="module")
def base_datasets():
    loader = DataLoader(base_path="data/manufacturing_base")
    return loader.load_base_data()

# ----------------------------------------
# NEW FIXTURE: Locates the real generated packing lists
# ----------------------------------------
@pytest.fixture
def real_packing_list_dir() -> str:
    """
    Provides the path to the directory containing real, generated packing lists.
    Skips the test if the directory or a sample file does not exist.
    """
    # Construct a path relative to this test file's location
    # tests/../data/generated_documents/packing_lists
    dir_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "generated_documents",
            "packing_lists",
        )
    )

    # The file we expect to find for order ORBOX0011 is PackingList-PL1011.pdf
    # based on the logic in your ManufacturingEnvironment
    sample_file_path = os.path.join(dir_path, "PackingList-PL1011.pdf")

    if not os.path.exists(sample_file_path):
        pytest.skip(
            f"Prerequisite file not found: {sample_file_path}. "
            "Run the Phase 1 data generation script first."
        )

    return dir_path

def test_machine_log_success(base_datasets):
    tool = MachineLogTool(base_datasets)
    df = base_datasets["machine_log"]
    # Pick a real printer ID from the CSV
    printer_id = df["Machine"].iloc[0]
    result = tool.run(printer_id)
    assert isinstance(result, list)
    assert result and result[0]["Machine"] == printer_id


def test_machine_log_no_data():
    tool = MachineLogTool(datasets={})
    res = tool.run("Printer_X")
    assert isinstance(res, list)
    assert res[0]["error"] == "Machine log data not available."


def test_worker_data_success(base_datasets):
    tool = WorkerDataTool(base_datasets)
    df = base_datasets["worker_data"]
    worker_id = df["_value"].iloc[0]
    result = tool.run(worker_id)
    assert isinstance(result, list)
    assert result and result[0]["_value"] == worker_id


def test_worker_data_no_data():
    tool = WorkerDataTool(datasets={})
    res = tool.run("0000000000")
    assert isinstance(res, list)
    assert res[0]["error"] == "Worker data not available."


def test_relationship_tool_success(base_datasets):
    tool = RelationshipTool(base_datasets)
    df = base_datasets["relationship_data"]
    parent_id = df["parent"].iloc[0]
    result = tool.run(parent_id)
    assert isinstance(result, list)
    # Every record must reference the queried ID
    assert all(
        (rec.get("parent") == parent_id) or (rec.get("child") == parent_id)
        for rec in result
    )


def test_relationship_no_data():
    tool = RelationshipTool(datasets={})
    res = tool.run("X")
    assert res[0]["error"] == "Relationship data not available."


def test_location_query_success(base_datasets):
    tool = LocationQueryTool(base_datasets)
    df = base_datasets["location_data"]
    entity_id = df["_value"].iloc[0]
    result = tool.run(entity_id)
    assert isinstance(result, list)
    assert result and result[0]["_value"] == entity_id


def test_location_query_no_data():
    tool = LocationQueryTool(datasets={})
    res = tool.run("YYY")
    assert res[0]["error"] == "Location data not available."


def test_document_parser_success():
    # Uses real files under data/generated_documents/certificates
    tool = DocumentParserTool(datasets={})
    # Known existing order
    out = tool.run("ORBOX0011", fuzzy_enabled=False)
    assert out.get("source_document") == "ARC-ORBOX0011.pdf"
    # Should have parsed at least one field besides source_document
    assert len(out) > 1


def test_document_parser_not_found():
    tool = DocumentParserTool(datasets={})
    bad = tool.run("NO_SUCH_ORDER", fuzzy_enabled=False)
    assert "error" in bad


def test_barcode_validator_valid_and_invalid():
    validator = BarcodeValidatorTool(datasets={})
    ok = validator.run("Printer_9")
    assert ok["is_valid"] is True
    assert ok["format"] == "printer"
    bad = validator.run("INVALID_123")
    assert bad["is_valid"] is False
    assert bad["format"] == "unknown"


def test_manufacturing_validator_patterns():
    mv = ManufacturingValidator()
    # Test series matching for printer pattern
    df = pd.DataFrame({"col": ["Printer_1", "foo", "Printer_42"]})
    mask = mv.validate_barcode_formats(df, "col")
    assert mask.tolist() == [True, False, True]

# ----------------------------------------
# NEW TEST: Uses the real generated PDF
# ----------------------------------------
def test_packing_list_parser_tool_on_real_file(real_packing_list_dir):
    """
    Tests the PackingListParserTool against a real PDF generated by the pipeline.
    """
    # Instantiate the tool and point it to the real document directory
    tool = PackingListParserTool(datasets={})
    tool.doc_path = real_packing_list_dir

    # --- Success Case ---
    # We test against the packing list for ORBOX0011, which your generator
    # names "PL1011".
    result = tool.run("PL1011")

    # Check that the extracted Order ID is correct
    assert "error" not in result, f"Tool returned an error: {result.get('error')}"
    assert result.get("order_id") == "ORBOX0011"
    assert result.get("source_document") == "PackingList-PL1011.pdf"

    # --- Error Case: File not found ---
    result_err = tool.run("PL_MISSING")
    assert "error" in result_err
    assert "not found" in result_err["error"]