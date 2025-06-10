import os
import random
import pandas as pd
import pytest
import pypdf

from pathlib import Path

from src.utils.data_loader import DataLoader
from src.data_generation.document_generator import FAACertificateGenerator
from src.data_generation.manufacturing_environment import ManufacturingEnvironment
from src.data_generation.data_quality_controller import DataQualityController


# ----------------------------------------
# DataLoader tests
# ----------------------------------------
@pytest.fixture
def tmp_base_with_one_csv(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "location_data.csv").write_text("id,_value***REMOVED***nL1,100***REMOVED***nL2,200***REMOVED***n")
    return str(base)


def test_data_loader_loads_present_csv_and_warns_for_missing(
    tmp_base_with_one_csv, capsys
):
    dl = DataLoader(base_path=tmp_base_with_one_csv)
    data = dl.load_base_data()
    out = capsys.readouterr().out

    # loaded location_data
    assert "location_data" in data
    assert not data["location_data"].empty

    # missing ones produce warnings and empty DataFrames
    for key in ("machine_log", "relationship_data", "worker_data"):
        assert key in data
        assert data[key].empty
        assert "WARNING:" in out


# ----------------------------------------
# FAACertificateGenerator tests
# ----------------------------------------
def test_faa_generator_raises_if_template_missing(tmp_path):
    missing = tmp_path / "nope.pdf"
    with pytest.raises(FileNotFoundError):
        FAACertificateGenerator(template_path=str(missing))


@pytest.fixture
def dummy_pdf_template(tmp_path):
    p = tmp_path / "template.pdf"
    # Minimal PDF header so pypdf doesn't choke
    p.write_bytes(b"%PDF-1.4***REMOVED***n%%EOF")
    return str(p)


def test_faa_generate_certificate_writes_file(
    monkeypatch, dummy_pdf_template, tmp_path
):
    # stub PdfReader and PdfWriter
    monkeypatch.setattr(pypdf, "PdfReader", lambda f, strict: object())

    class DummyWriter:
        def __init__(self):
            self.pages = [{}]
            self._root_object = {}
            self._objects = []

        def append(self, _r):
            pass

        def update_page_form_field_values(
            self, page, field_data, auto_regenerate=False
        ):
            self.fd = field_data

        def write(self, stream):
            stream.write(b"PDF OK")

    monkeypatch.setattr(pypdf, "PdfWriter", DummyWriter)
    monkeypatch.setattr(
        FAACertificateGenerator,
        "_set_need_appearances_writer",
        lambda self, w: w
    )

    out_pdf = tmp_path / "out.pdf"
    gen = FAACertificateGenerator(template_path=dummy_pdf_template)
    gen.generate_certificate({"f1": "v1"}, str(out_pdf))

    assert out_pdf.exists()
    assert out_pdf.read_bytes() == b"PDF OK"


# ----------------------------------------
# ManufacturingEnvironment tests
# ----------------------------------------
def test_manufacturing_environment_writes_baseline_and_invokes_docs(
    monkeypatch, tmp_path
):
    # 1) Stub DataLoader.load_base_data()
    base_data = {
        "location_data": pd.DataFrame({"A": [1]}),
        "machine_log": pd.DataFrame(),
        "relationship_data": pd.DataFrame({
            "parent": ["ORBOX1"], "child": ["3DOR1"]
        }),
        "worker_data": pd.DataFrame(),
    }
    monkeypatch.setattr(DataLoader, "load_base_data", lambda self: base_data)

    # 2) Prevent FAACertificateGenerator.__init__ from raising
    monkeypatch.setattr(
        FAACertificateGenerator,
        "__init__",
        lambda self, template_path=None: None
    )

    # 3) Collect calls to generate_certificate
    calls = []

    class StubGen:
        def generate_certificate(self, fd, path):
            calls.append(path)
            Path(path).write_text("X")

    # 4) Instantiate and override doc_generator
    env = ManufacturingEnvironment(
        base_data_path="unused",
        output_path=str(tmp_path / "out"),
        doc_output_path=str(tmp_path / "docs")
    )
    env.doc_generator = StubGen()
    env.setup_baseline_environment()

    # Baseline CSV was written:
    assert (tmp_path / "out" / "location_data.csv").exists()

    # Exactly one certificate for ORBOX1
    assert len(calls) == 1
    assert "ARC-ORBOX1.pdf" in calls[0]


# ----------------------------------------
# DataQualityController tests
# ----------------------------------------
@pytest.fixture
def tmp_q0(tmp_path):
    q0 = tmp_path / "q0"
    q0.mkdir()
    # location_data
    pd.DataFrame({"_value": ["A", "B", "C", "D", "E"]}) ***REMOVED***
      .to_csv(q0 / "location_data.csv", index=False)
    # worker_data
    pd.DataFrame({"_value": ["W1", "W2", "W3"]}) ***REMOVED***
      .to_csv(q0 / "worker_data.csv", index=False)
    # relationship_data
    pd.DataFrame({
        "parent": ["ORBOX1", "X", "ORBOX2"],
        "child":  ["3DOR1",  "3D", "3DOR2"]
    }).to_csv(q0 / "relationship_data.csv", index=False)
    (q0 / "machine_log.csv").write_text("a,b***REMOVED***n")
    return str(q0)


def test_q1_space_injection(tmp_q0):
    random.seed(0)
    ctrl = DataQualityController(baseline_path=tmp_q0)
    data, tracker, targeted_ids = ctrl.apply_corruption("Q1")
    # make sure we actually got back a list of IDs
    assert isinstance(targeted_ids, list)
    # We expect at least one entry and each 'corrupted' has space + original as substring
    assert len(tracker.q1_log) > 0
    for entry in tracker.q1_log:
        orig = entry["original"]
        corr = entry["corrupted"]
        assert orig != corr
        assert " " in corr
        assert orig in corr


def test_q2_char_missing(tmp_q0):
    random.seed(1)
    ctrl = DataQualityController(baseline_path=tmp_q0)
    data, tracker, targeted_ids = ctrl.apply_corruption("Q2")
    assert isinstance(targeted_ids, list)
    for entry in tracker.q2_log:
           original = entry["original"]
           corrupted = entry["corrupted"]
           removed_char = entry["removed_char"]
           assert len(entry["corrupted"]) + 1 == len(entry["original"])
           # Check that the count of the removed character is one less
           assert original.count(removed_char) == corrupted.count(removed_char) + 1

def test_q3_missing_records(tmp_q0):
    random.seed(2)
    ctrl = DataQualityController(baseline_path=tmp_q0)
    before = pd.read_csv(
        os.path.join(tmp_q0, "relationship_data.csv"),
        dtype=str,
    )
    data, tracker, targeted_ids = ctrl.apply_corruption("Q3")
    assert isinstance(targeted_ids, list)

    # number of dropped records matches log
    assert len(before) - len(data["relationship_data"]) == len(tracker.q3_log)

def test_dqc_load_baseline_handles_missing_dir(tmp_path):
    # Test what happens if the baseline directory doesn't exist at all
    ctrl = DataQualityController(baseline_path=str(tmp_path / "nonexistent"))
    # _load_baseline should handle the FileNotFoundError and return an empty dict
    assert ctrl.datasets == {}


def test_dqc_get_all_orders(tmp_q0):
    # Test the helper that extracts order IDs
    ctrl = DataQualityController(baseline_path=tmp_q0)
    orders = ctrl._get_all_orders()
    # Based on your tmp_q0 fixture, it should find ORBOX1 and ORBOX2
    assert sorted(orders) == ["ORBOX1", "ORBOX2"]


def test_dqc_get_order_related_entities(tmp_q0):
    # Test the helper that maps an order to all its related parts
    ctrl = DataQualityController(baseline_path=tmp_q0)
    entities = ctrl._get_order_related_entities("ORBOX1")
    assert entities["orders"] == {"ORBOX1"}
    assert entities["gears"] == {"3DOR1"}
    # This will be empty as the fixture has no printer relationships
    assert entities["printers"] == set()

# Add this test as well

def test_dqc_apply_corruption_on_empty_dataset(monkeypatch):
    # Create a controller where the baseline data is empty
    monkeypatch.setattr(
        DataQualityController, "_load_baseline", lambda self: {}
    )
    ctrl = DataQualityController(baseline_path="ignored")

    # All corruption methods should run without error and return empty results
    for qc in ["Q1", "Q2", "Q3"]:
        data, tracker, targeted_ids = ctrl.apply_corruption(qc)
        assert data == {}
        assert targeted_ids == []

# Add this test to verify file output

def test_dqc_save_corrupted_data(tmp_path):
    # We don't need a full controller, just an instance to call the method
    ctrl = DataQualityController(baseline_path="ignored")
    
    # Create some dummy data to save
    dummy_data = {"test_table": pd.DataFrame({"col": [1, 2]})}
    
    from src.data_generation.error_tracker import ErrorTracker
    dummy_tracker = ErrorTracker()
    dummy_tracker.log_q1_space_injection(0, "loc", "a", " a")

    # Call the save method
    ctrl.save_corrupted_data(dummy_data, dummy_tracker, "Q1")

    # Check that the files were written to the correct directory
    output_dir = Path("data/experimental_datasets/Q1_dataset")
    assert (output_dir / "test_table_Q1.csv").exists()
    assert (output_dir / "all_tables_Q1_errors.csv").exists()

    # Clean up the created directory after the test
    import shutil
    shutil.rmtree(output_dir)