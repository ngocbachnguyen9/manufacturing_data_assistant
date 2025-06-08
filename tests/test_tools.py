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


@pytest.fixture(scope="module")
def base_datasets():
    loader = DataLoader(base_path="data/manufacturing_base")
    return loader.load_base_data()


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