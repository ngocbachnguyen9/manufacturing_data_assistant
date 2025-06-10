# tests/test_full_pipeline.py

import os
import json
import pytest
import pandas as pd
import yaml
import random
import sys
from pathlib import Path

# ensure project root on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_phase1_generation     import main as setup_main
from run_phase3_human_study    import main as phase3_main

from src.data_generation.manufacturing_environment import ManufacturingEnvironment
from src.data_generation.data_quality_controller   import DataQualityController
from src.data_generation.ground_truth_generator     import GroundTruthGenerator
from src.experiment.task_generator                  import TaskGenerator
from src.human_study.study_platform                 import StudyPlatform

def test_full_pipeline_writes_dirty_ids(tmp_path, monkeypatch):
    # Set up a fake Q0_baseline folder
    q0 = tmp_path / "data" / "experimental_datasets" / "Q0_baseline"
    q0.mkdir(parents=True)
    import pandas as pd
    pd.DataFrame({"parent":["OR1"], "child":["3D1"]})***REMOVED***
      .to_csv(q0 / "relationship_data.csv", index=False)

    # Run in tmp dir
    monkeypatch.chdir(tmp_path)

    # Stub out phase-1 dependencies
    monkeypatch.setattr(ManufacturingEnvironment, "setup_baseline_environment", lambda self: None)
    # apply_corruption now returns 3-tuple
    monkeypatch.setattr(
        DataQualityController,
        "apply_corruption",
        lambda self, qc: ({}, type("ET", (), {"q1_log":[], "q2_log":[], "q3_log":[]} ), [])
    )
    monkeypatch.setattr(GroundTruthGenerator, "generate_all_ground_truths", lambda self: None)

    # 1) Run phase1
    setup_main()

    # Should have dirty_ids.json
    di_path = tmp_path / "experiments" / "human_study" / "dirty_ids.json"
    assert di_path.exists()
    di = json.loads(di_path.read_text())
    assert set(di.keys()) == {"Q1","Q2","Q3"}

    # Prepare config for phase3
    cfg = {
        "human_study": {
            "participant_matrix": {"P1": {"quality_pattern": "PQ1", "prompt_pattern": "PC1"}},
            "quality_patterns": {"PQ1": {"Q0":1}},
            "prompt_patterns": {"PC1": {"E":1}},
        },
        "task_complexity": {"easy": {"description": "Do task {ENTITY_ID}"}},
        "experiment": {"random_seed": 0},
    }
    os.makedirs("config", exist_ok=True)
    with open("config/experiment_config.yaml","w") as f:
        yaml.safe_dump(cfg, f)

    # Stub TaskGenerator to produce a single assignment
    def fake_init(self, config, dirty_ids):
        self.config = config
        self.dirty_ids = dirty_ids
        self.participants = {"P1": {"quality_pattern": "PQ1", "prompt_pattern": "PC1"}}
        self.quality_patterns = config["human_study"]["quality_patterns"]
        self.prompt_patterns = config["human_study"]["prompt_patterns"]
        self.task_templates = config["task_complexity"]
        self.valid_ids = {"easy":["OR1"],"medium":[],"hard":["OR1"]}
        self.clean_ids = {"easy":["OR1"],"medium":[],"hard":["OR1"]}
    monkeypatch.setattr(TaskGenerator, "__init__", fake_init)
    monkeypatch.setattr(
        TaskGenerator,
        "generate_all_assignments",
        lambda self: {"P1":[
            {
                "task_id":"P1_task_1",
                "participant_id":"P1",
                "complexity":"easy",
                "quality_condition":"Q0",
                "query_string":"Do task OR1",
                "dataset_path":""
            }
        ]}
    )
    monkeypatch.setattr(TaskGenerator, "save_assignments", lambda self, a: None)

    # Stub StudyPlatform.run_session to avoid interactive I/O
    monkeypatch.setattr(StudyPlatform, "run_session", lambda self: None)

    # 2) Run phase3
    phase3_main()

    # Should have participant_assignments.json
    asn = tmp_path / "experiments" / "human_study" / "participant_assignments.json"
    assert asn.exists()