import os
import json
import time
import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiment.llm_evaluation import LLMEvaluationRunner
from src.utils.llm_provider import MockLLMProvider
from src.utils.cost_tracker import CostTracker


# -- Stubs & fixtures -------------------------------------------------------

class DummyDataLoader:
    def __init__(self, base_path):
        pass
    def load_base_data(self):
        return {}

class StubMasterAgent:
    def __init__(self, llm_provider, datasets, config, cost_tracker):
        pass
    def run_query(self, query: str):
        # return the new dict form
        return {
            "final_report": "Results include gear 3DOR1",
            "reconciliation_summary": {"confidence": 1.0, "issues_found": []}
        }

@pytest.fixture
def runner_and_dirs(tmp_path, monkeypatch):
    # 1. Stub out JSON loading
    assignments = {
        "P1": [{"task_id": "T1", "complexity": "easy", "quality_condition": "Q0",
                "query_string": "Find gears for ORBOX0014", "dataset_path": "ignored"}]
    }
    ground_truth = [
        {"task_id": "T1", "complexity_level": "easy",
         "baseline_answer": {"gear_list": ["3DOR1"]}}
    ]
    monkeypatch.setattr(
        LLMEvaluationRunner, "_load_json",
        lambda self, path: assignments if "assignments" in path else ground_truth
    )

    # 2. Stub out heavy dependencies
    import src.experiment.llm_evaluation as m
    monkeypatch.setattr(m, "DataLoader", DummyDataLoader)
    monkeypatch.setattr(m, "MasterAgent", StubMasterAgent)

    # 3. Patch the MockLLMProvider's generate method to act like a judge
    #    when it sees the judge prompt.
    original_generate = MockLLMProvider.generate
    def mock_generate_with_judge(self, prompt):
        if "You are an impartial judge" in prompt:
            # If this is the judge prompt, return "Correct"
            return {"content": "Correct", "input_tokens": 10, "output_tokens": 1}
        else:
            # Otherwise, use the original mock logic (for planning/synthesis)
            return original_generate(self, prompt)

    monkeypatch.setattr(MockLLMProvider, "generate", mock_generate_with_judge)

    # 4. Create the runner instance correctly
    cfg = {
        "llm_evaluation": {"models_to_test": ["M1"]},
        "llm_providers": {
            "M1": {"cost_per_1m_tokens_input": 0.0, "cost_per_1m_tokens_output": 0.0}
        },
    }
    runner = LLMEvaluationRunner(config=cfg, use_mock=True)

    # 5. Point the runner's log directory to the temporary path
    runner.log_dir = str(tmp_path)
    return runner, tmp_path


# --- UPDATE the test_evaluate_answer_missing_gt function ---
def test_evaluate_answer_missing_gt(runner_and_dirs):
    runner, _ = runner_and_dirs
    # The method now requires an llm_provider instance as the third argument
    ok, gt = runner._evaluate_answer("NO_SUCH", "whatever", MockLLMProvider("M1"))
    assert ok is False
    assert "Ground truth for task_id 'NO_SUCH' not found" in gt


def test_run_evaluation_end_to_end(runner_and_dirs):
    runner, tmp_path = runner_and_dirs
    # Note: The runner should use seeded generation internally
    runner.run_evaluation()

    # One result row
    assert len(runner.results) == 1
    res = runner.results[0]

    # Check new keys are present
    for key in [
        "task_id", "model", "complexity", "quality_condition",
        "completion_time_sec", "is_correct",
        "total_cost_usd", "input_tokens", "output_tokens",
        "final_confidence", "reconciliation_issues",
        "llm_final_report", "ground_truth_answer"
    ]:
        assert key in res

    # CSV file was written
    csv_path = tmp_path / "llm_performance_results.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert df.loc[0, "task_id"] == "T1"
    assert df.loc[0, "model"] == "M1"
    
    # UPDATED: Use '==' for value comparison instead of 'is' for identity
    assert df.loc[0, "is_correct"] == True