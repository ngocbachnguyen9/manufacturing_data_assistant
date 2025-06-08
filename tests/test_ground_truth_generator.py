# tests/test_ground_truth_generator.py

import os
import json
import pytest

import sys
# ensure 'src' is on PYTHONPATH
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.data_generation.ground_truth_generator import GroundTruthGenerator


@pytest.fixture
def sample_baseline(tmp_path):
    """
    Create a fake Q0_baseline folder with exactly:
      - relationship_data.csv containing order->gear and printer->gear
      - location_data.csv containing one warehouse scan
    """
    base = tmp_path / "Q0_baseline"
    base.mkdir()

    # relationship_data.csv must link ORBOX1->3DOR1 and Printer5->3DOR1
    rel_csv = base / "relationship_data.csv"
    rel_csv.write_text(
        "parent,child***REMOVED***n"
        "ORBOX1,3DOR1***REMOVED***n"
        "Printer5,3DOR1***REMOVED***n"
    )

    # location_data.csv: one Parts Warehouse scan for gear 3DOR1
    loc_csv = base / "location_data.csv"
    loc_csv.write_text(
        "_time,_value,location***REMOVED***n"
        "2020-01-02T14:30:00Z,3DOR1,Parts Warehouse***REMOVED***n"
    )

    return str(base)


def test_ground_truth_generator_outputs_expected_json_and_paths(
    sample_baseline, tmp_path
):
    # Use tmp_path as our output_dir
    gen = GroundTruthGenerator(
        baseline_path=sample_baseline, output_dir=str(tmp_path)
    )
    gen.generate_all_ground_truths()

    # Check both output files exist
    answers_file = tmp_path / "baseline_answers.json"
    paths_file = tmp_path / "data_traversal_paths.json"
    assert answers_file.exists(), "baseline_answers.json not written"
    assert paths_file.exists(), "data_traversal_paths.json not written"

    # Load contents
    answers = json.loads(answers_file.read_text())
    paths = json.loads(paths_file.read_text())

    # We expect exactly one easy, one medium, one hard
    easy = [a for a in answers if a["complexity_level"] == "easy"]
    medium = [a for a in answers if a["complexity_level"] == "medium"]
    hard = [a for a in answers if a["complexity_level"] == "hard"]

    assert len(easy) == 1
    assert len(medium) == 1
    assert len(hard) == 1

    # --- Easy task assertions ---
    e = easy[0]
    assert e["task_id"] == "easy_ORBOX1_0"
    assert e["query_instance"] == "Find all gears for Order ORBOX1"
    ba_e = e["baseline_answer"]
    assert ba_e["order_id"] == "ORBOX1"
    assert ba_e["gear_count"] == 1
    assert ba_e["gear_list"] == ["3DOR1"]

    # Check its path
    eid = e["task_id"]
    assert eid in paths
    path_e = paths[eid]
    assert path_e["data_sources"] == ["relationship_data"]
    assert path_e["steps"] == [
        "1. Query relationship_data where parent='ORBOX1'",
        "2. Collect all child entities and find unique set: ['3DOR1']",
    ]

    # --- Medium task assertions ---
    m = medium[0]
    assert m["task_id"] == "medium_3DOR1_0"
    assert m["query_instance"] == "Determine printer for Part 3DOR1"
    ba_m = m["baseline_answer"]
    assert ba_m["part_id"] == "3DOR1"
    assert ba_m["assigned_printer"] == "Printer5"

    # Check its path
    mid = m["task_id"]
    assert mid in paths
    path_m = paths[mid]
    assert path_m["data_sources"] == ["relationship_data"]
    assert path_m["steps"] == [
        "1. Query relationship_data where child='3DOR1'",
        "2. Find parent entity starting with 'Printer_': Printer5",
    ]

    # --- Hard task assertions ---
    h = hard[0]
    assert h["task_id"] == "hard_ORBOX1_0"
    assert "Verify ARC date vs warehouse arrival for ORBOX1" in h["query_instance"]
    ba_h = h["baseline_answer"]
    # certificate_date is hard-coded
    assert ba_h["certificate_date"] == "2024-10-28"
    assert ba_h["warehouse_arrival_date"] == "2020-01-02"
    assert ba_h["date_match_status"] is False

    # Check its path
    hid = h["task_id"]
    assert hid in paths
    path_h = paths[hid]
    assert path_h["data_sources"] == [
        "relationship_data",
        "location_data",
    ]

    expected_h_steps = [
        "1. Query relationship_data for children of 'ORBOX1': Found 1 gears.",
        "2. Query location_data for these gears at 'Parts Warehouse': Found 1 scans.",
        "3. Determined latest arrival timestamp: 2020-01-02",
    ]
    assert path_h["steps"] == expected_h_steps