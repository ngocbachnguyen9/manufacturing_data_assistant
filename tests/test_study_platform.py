import os
import json
import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
from src.human_study.study_platform import StudyPlatform

def test_init_raises_missing(tmp_path):
    missing = tmp_path / "no.json"
    with pytest.raises(FileNotFoundError):
        StudyPlatform(str(missing))

def test_init_creates_log_dir(tmp_path, monkeypatch):
    # write a minimal assignments file
    assign = {"P1": []}
    f = tmp_path / "assign.json"
    f.write_text(json.dumps(assign))

    monkeypatch.chdir(tmp_path)
    sp = StudyPlatform(str(f))
    # log_dir attribute
    assert "session_logs" in sp.log_dir
    assert os.path.isdir(sp.log_dir)

def test__save_results_creates_csv(tmp_path, monkeypatch):
    # same setup
    assign = {"P1": []}
    f = tmp_path / "a.json"
    f.write_text(json.dumps(assign))
    monkeypatch.chdir(tmp_path)
    sp = StudyPlatform(str(f))

    # call the saver
    results = [
        {
            "task_id": "T1",
            "participant_id": "P1",
            "completion_time_sec": 7,
            "accuracy": 1,
            "participant_answer": "yes",
            "notes": "note",
        }
    ]
    sp._save_results("P1", results)

    out = Path(sp.log_dir) / "P1_results.csv"
    assert out.exists()
    df = pd.read_csv(out)
    assert df.loc[0, "task_id"] == "T1"
    assert df.loc[0, "accuracy"] == 1

def test_run_session_unknown_participant(tmp_path, monkeypatch, capsys):
    # write assignments
    assign = {"P1": []}
    f = tmp_path / "assign.json"
    f.write_text(json.dumps(assign))
    monkeypatch.chdir(tmp_path)
    sp = StudyPlatform(str(f))

    # simulate user enters "P2"
    monkeypatch.setattr("builtins.input", lambda prompt="": "P2")
    sp.run_session()

    out = capsys.readouterr().out
    assert "Error: Participant ID 'P2' not found." in out