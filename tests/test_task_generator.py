import os
import json
import random
import pytest
import pandas as pd

# make sure we can import TaskGenerator
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
from src.experiment.task_generator import TaskGenerator

# a minimal experiment config to drive TaskGenerator
BASE_CONFIG = {
    "human_study": {
        "participant_matrix": {
            "P1": {"quality_pattern": "PQ1", "prompt_pattern": "PC1"},
            "P2": {"quality_pattern": "PQ2", "prompt_pattern": "PC2"},
        },
        "quality_patterns": {
            "PQ1": {"Q0": 2, "Q1": 1},
            "PQ2": {"Q0": 1, "Q1": 2},
        },
        "prompt_patterns": {
            "PC1": {"E": 1, "M": 1, "H": 1},
            "PC2": {"E": 2, "M": 1, "H": 0},
        },
    },
    "task_complexity": {
        "easy": {"description": "Do easy task {...}"},
        "medium": {"description": "Do medium task {...}"},
        "hard": {"description": "Do hard task {...}"},
    },
    "experiment": {"random_seed": 0},
}
# We’ll feed in an “empty” dirty_ids map so that all clean IDs remain
EMPTY_DIRTY = {"Q1": [], "Q2": [], "Q3": []}

@pytest.fixture(autouse=True)
def stub_get_valid_ids(monkeypatch):
    """
    Stub out _get_valid_ids so we don't hit the real CSVs.
    Provide predictable lists for easy/medium/hard.
    """
    fake = {
        "easy": ["E1", "E2", "E3"],
        "medium": ["M1", "M2", "M3"],
        "hard": ["H1", "H2", "H3"],
    }
    monkeypatch.setattr(
        TaskGenerator, "_get_valid_ids", lambda self: {k: v.copy() for k, v in fake.items()}
    )

def test_create_task_list_counts_and_shuffle():
    tg = TaskGenerator(BASE_CONFIG, EMPTY_DIRTY)
    # PQ1 => 2×Q0 + 1×Q1
    ql = tg._create_task_list("quality", "PQ1")
    assert sorted(ql) == ["Q0", "Q0", "Q1"]
    # PC1 => E/M/H each once
    pl = tg._create_task_list("prompt", "PC1")
    assert sorted(pl) == ["E", "H", "M"]

def test_generate_all_assignments_structure(tmp_path, monkeypatch):
    # run inside a temp cwd so save_assignments won't pollute project root
    monkeypatch.chdir(tmp_path)
    tg = TaskGenerator(BASE_CONFIG, EMPTY_DIRTY)
    assignments = tg.generate_all_assignments()

    # should have one entry per participant
    assert set(assignments.keys()) == {"P1", "P2"}

    for pid, tasks in assignments.items():
        # tasks count equals total quality-pattern slots
        qp = BASE_CONFIG["human_study"]["participant_matrix"][pid]["quality_pattern"]
        expected = sum(BASE_CONFIG["human_study"]["quality_patterns"][qp].values())
        assert len(tasks) == expected

        for t in tasks:
            # basic fields
            assert t["participant_id"] == pid
            assert t["complexity"] in ("easy", "medium", "hard")
            assert t["quality_condition"].startswith("Q")
            # substituted query
            assert "{" not in t["query_string"]
            # correct dataset path
            qc = t["quality_condition"]
            if qc == "Q0":
                assert "Q0_baseline" in t["dataset_path"]
            else:
                assert f"{qc}_dataset" in t["dataset_path"]

def test_generate_all_assignments_exhausted_ids(monkeypatch):
    # stub valid_ids so 'easy' is empty
    monkeypatch.setattr(
        TaskGenerator, "_get_valid_ids", lambda self: {"easy": [], "medium": ["M1"], "hard": ["H1"]}
    )
    tg = TaskGenerator(BASE_CONFIG, EMPTY_DIRTY)
    with pytest.raises(ValueError):
        tg.generate_all_assignments()

def test_save_assignments_writes_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tg = TaskGenerator(BASE_CONFIG, EMPTY_DIRTY)
    sample = {"P1": [{"task_id": "P1_t1"}]}
    tg.save_assignments(sample)

    out = tmp_path / "experiments" / "human_study" / "participant_assignments.json"
    assert out.exists()

    loaded = json.loads(out.read_text())
    assert loaded == sample

def test_get_clean_ids_filters_out_dirty(monkeypatch):
    """
    Ensure TaskGenerator._get_clean_ids drops exactly the IDs
    passed in via dirty_ids, for both orders and gears.
    """
    # Stub valid IDs
    fake_valid = {
        "easy":   ["OR1", "OR2", "OR3"],
        "medium": ["3D1", "3D2", "3D3"],
        "hard":   ["OR1", "OR2", "OR3"],
    }
    # Dirty includes one order and one gear
    dirty_ids = {
        "Q1": ["OR2", "3D3"],
        "Q2": [],
        "Q3": [],
    }
    monkeypatch.setattr(
        TaskGenerator, "_get_valid_ids",
        lambda self: {k: v.copy() for k, v in fake_valid.items()}
    )
    tg = TaskGenerator(BASE_CONFIG, dirty_ids)
    clean = tg._get_clean_ids()
    # OR2 and 3D3 should be removed
    assert clean["easy"] == ["OR1", "OR3"]
    assert clean["hard"] == ["OR1", "OR3"]
    assert clean["medium"] == ["3D1", "3D2"]

def test_query_string_substitution(monkeypatch):
    """
    Verify that TaskGenerator.format replacement uses ENTITY_ID
    (not the old '{...}') correctly.
    """
    # Minimal config with one participant and one task
    cfg = {
        "human_study": {
            "participant_matrix": {"P1": {"quality_pattern": "PQ1", "prompt_pattern": "PC1"}},
            "quality_patterns": {"PQ1": {"Q0": 1}},
            "prompt_patterns": {"PC1": {"E": 1}},
        },
        # Use a template with {ENTITY_ID}
        "task_complexity": {"easy": {"description": "Check order {ENTITY_ID}"}},
        "experiment": {"random_seed": 0},
    }
    # Stub valid_ids to contain exactly ORBOX123
    monkeypatch.setattr(
        TaskGenerator, "_get_valid_ids",
        lambda self: {"easy": ["ORBOX123"], "medium": [], "hard": ["ORBOX123"]}
    )
    tg = TaskGenerator(cfg, dirty_ids={"Q1": [], "Q2": [], "Q3": []})
    assignments = tg.generate_all_assignments()
    qs = assignments["P1"][0]["query_string"]
    assert "ORBOX123" in qs
    assert "ENTITY_ID" not in qs
    assert "{...}" not in qs