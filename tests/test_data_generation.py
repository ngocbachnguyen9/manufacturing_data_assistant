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
    data, tracker = ctrl.apply_corruption("Q1")

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
    data, tracker = ctrl.apply_corruption("Q2")

    for entry in tracker.q2_log:
        assert len(entry["corrupted"]) + 1 == len(entry["original"])
        assert entry["removed_char"] not in entry["corrupted"]


def test_q3_missing_records(tmp_q0):
    random.seed(2)
    ctrl = DataQualityController(baseline_path=tmp_q0)
    before = pd.read_csv(
        os.path.join(tmp_q0, "relationship_data.csv"),
        dtype=str,
    )
    data, tracker = ctrl.apply_corruption("Q3")

    # number of dropped records matches log
    assert len(before) - len(data["relationship_data"]) == len(tracker.q3_log)