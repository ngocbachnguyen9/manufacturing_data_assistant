# tests/test_ground_truth_validation.py

import json
import pandas as pd
import pytest
from pathlib import Path

from src.data_generation.ground_truth_generator import GroundTruthGenerator
from src.utils.manufacturing_validator import ManufacturingValidator


@pytest.fixture
def sample_baseline(tmp_path):
    """
    Create a minimal Q0_baseline directory with:
      - relationship_data.csv
      - location_data.csv
    """
    base = tmp_path / "Q0_baseline"
    base.mkdir()

    # relationship_data: one order with one gear, and one printer-->gear link
    rel_csv = base / "relationship_data.csv"
    rel_csv.write_text(
        "parent,child***REMOVED***n"
        "ORBOX1,3DOR1***REMOVED***n"
        "Printer5,3DOR1***REMOVED***n"
    )

    # location_data: one warehouse scan for ORBOX1
    loc_csv = base / "location_data.csv"
    loc_csv.write_text(
        "_value,location,_time***REMOVED***n"
        "ORBOX1,Parts Warehouse,2020-01-02T14:30:00Z***REMOVED***n"
        "ORBOX2,Other Location,2021-01-01T00:00:00Z***REMOVED***n"
    )

    return str(base)


def test_ground_truth_generator_outputs_expected_json(sample_baseline, tmp_path):
    out_file = tmp_path / "baseline_answers.json"
    gen = GroundTruthGenerator(
        baseline_path=sample_baseline,
        output_path=str(out_file),
    )
    gen.generate_all_ground_truths()

    # JSON file must exist
    assert out_file.exists()

    answers = json.loads(out_file.read_text())
    # We only have one order and one gear, so we expect exactly
    # 1 easy, 1 medium and 1 hard entry.
    easy = [a for a in answers if a["complexity_level"] == "easy"]
    medium = [a for a in answers if a["complexity_level"] == "medium"]
    hard = [a for a in answers if a["complexity_level"] == "hard"]

    assert len(easy) == 1
    assert len(medium) == 1
    assert len(hard) == 1

    # --- Easy task ---
    e = easy[0]
    assert e["task_id"] == "easy_ORBOX1_0"
    assert e["query_instance"] == "Find all gears for Order ORBOX1"
    ba_e = e["baseline_answer"]
    assert ba_e["order_id"] == "ORBOX1"
    assert ba_e["gear_count"] == 1
    assert ba_e["gear_list"] == ["3DOR1"]

    # --- Medium task ---
    m = medium[0]
    assert m["task_id"] == "medium_3DOR1_0"
    assert m["query_instance"] == "Determine printer for Part 3DOR1"
    ba_m = m["baseline_answer"]
    assert ba_m["part_id"] == "3DOR1"
    assert ba_m["assigned_printer"] == "Printer5"

    # --- Hard task ---
    h = hard[0]
    assert h["task_id"] == "hard_ORBOX1_0"
    assert "Verify ARC date vs warehouse arrival for ORBOX1" in h["query_instance"]
    ba_h = h["baseline_answer"]
    # certificate_date is hard‐coded in the implementation
    assert ba_h["certificate_date"] == "2024-10-28"
    # we put a Parts Warehouse scan at 2020-01-02T14:30:00Z
    assert ba_h["warehouse_arrival_date"] == "2020-01-02"
    # they differ, so match should be False
    assert ba_h["date_match_status"] is False


def test_manufacturing_validator_barcode_formats():
    mv = ManufacturingValidator()
    df = pd.DataFrame({
        "worker_rfid": ["1234567890", "ABCDEFGHIJ"],
        "printer_id": ["Printer_42", "Bad_Printer"],
        "gear_code": ["3DOR123456", "3DOR12X"],
        "order_code": ["ORBOX999", "BOX100"],
        "material_batch": ["ABCD1234", "abcd5678"],
    })

    # worker_rfid must be exactly 10 digits
    assert mv.validate_barcode_formats(df, "worker_rfid").tolist() == [True, False]

    # printer must match Printer_<digits>
    assert mv.validate_barcode_formats(df, "printer_id").tolist() == [True, False]

    # gear must match 3DOR<5–6 digits>
    assert mv.validate_barcode_formats(df, "gear_code").tolist() == [True, False]

    # order must match ORBOX<digits>
    assert mv.validate_barcode_formats(df, "order_code").tolist() == [True, False]

    # material must match 4 uppercase letters + 4 digits
    assert mv.validate_barcode_formats(df, "material_batch").tolist() == [True, False]