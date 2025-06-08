# test_agents.py

import sys
import os
import pytest

# Ensure 'src' is on PYTHONPATH so we can import src.agents
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.agents.data_retrieval_agent import DataRetrievalAgent
from src.agents.reconciliation_agent import ReconciliationAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.agents.master_agent import MasterAgent

# -- Helpers & Fixtures -----------------------------------------------------

class DummyTool:
    """A fake tool that returns a preset value or raises if needed."""
    def __init__(self, result):
        self.result = result

    def run(self, _input):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

class DummyLLM:
    """A fake LLM client capturing prompts."""
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "DUMMY_LLM_RESPONSE"

@pytest.fixture
def tools():
    return {
        "tool_a": DummyTool("result_a"),
        "barcode_validator_tool": DummyTool(True),
    }

# -- Tests for DataRetrievalAgent -------------------------------------------

def test_data_retrieval_success(tools):
    agent = DataRetrievalAgent(tools)
    out = agent.retrieve("tool_a", "input_x")
    assert out == "result_a"

def test_data_retrieval_missing_tool(tools):
    agent = DataRetrievalAgent(tools)
    res = agent.retrieve("no_such_tool", "x")
    assert isinstance(res, dict)
    assert "error" in res

# -- Tests for ReconciliationAgent ------------------------------------------

def test_reconciliation_no_issues(tools):
    agent = ReconciliationAgent(tools)
    data = {"tool_a": "OK"}
    rec = agent.reconcile(data)
    assert rec["issues_found"] == []
    assert rec["validated_data"] == data
    assert rec["confidence"] == 1.0

def test_reconciliation_with_error(tools):
    agent = ReconciliationAgent(tools)
    data = {"tool_a": {"error": "failure"}}
    rec = agent.reconcile(data)
    assert len(rec["issues_found"]) == 1
    # confidence dropped by 0.25 per error
    assert pytest.approx(rec["confidence"], 0.01) == 0.75

# -- Tests for SynthesisAgent ------------------------------------------------

def test_synthesis_agent_formats_report_correctly():
    llm = DummyLLM()
    templates = {
        "easy_response": (
            "Order {order_id}; Count {gear_count}; "
            "Issues {data_quality_issues}; Conf {confidence_level}; "
            "Status {manufacturing_status}; Recs {recommendations}"
        )
    }
    syn = SynthesisAgent(llm, templates)

    reconciled = {
        "validated_data": {"relationship_tool": ["g1", "g2"]},
        "issues_found": [],
        "confidence": 1.0,
    }
    report = syn.synthesize(reconciled, "Find gears for OR123", "easy")
    assert "Order OR123" in report
    assert "Count 2" in report
    assert "Issues []" in report
    assert "Status Completed" in report

# -- Tests for MasterAgent ---------------------------------------------------

class StubMaster(MasterAgent):
    """
    Subclass MasterAgent to override _decompose_task with a fixed plan.
    """
    def _decompose_task(self, query: str):
        # Always return a one-step plan using our tool stub
        return ([{"tool": "relationship_tool", "input": "OR1"}], "easy")

def test_master_agent_end_to_end(monkeypatch, tools):
    # Add the tool that MasterAgent will call
    tools["relationship_tool"] = DummyTool(["g1", "g2", "g3"])
    llm = DummyLLM()
    config = {
        "system_prompts": {"master_agent_planning": ""},
        "response_formats": {"easy_response": "Count {gear_count} for {order_id}"},
    }

    master = StubMaster(llm, tools, config)
    report = master.run_query("Find gears for OR1")
    # Stub template: "Count {gear_count} for {order_id}"
    assert report.strip() == "Count 3 for OR1"

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main(["-v", __file__])