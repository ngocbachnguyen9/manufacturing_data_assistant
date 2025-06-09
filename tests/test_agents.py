# tests/test_agents.py

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.data_retrieval_agent import DataRetrievalAgent
from src.agents.reconciliation_agent import ReconciliationAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.agents.master_agent import MasterAgent


class DummyTool:
    def __init__(self, result):
        self.result = result

    def run(self, *_args, **_kwargs):
        return self.result


class DummyLLM:
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "LLM_OK"


# -- DataRetrievalAgent tests ----------------------------------------------

def test_data_retrieval_success():
    tools = {"foo": DummyTool("bar")}
    agent = DataRetrievalAgent(tools, config={})
    assert agent.retrieve("foo", "input") == "bar"


def test_data_retrieval_missing_tool():
    agent = DataRetrievalAgent({}, config={})
    res = agent.retrieve("missing", "x")
    assert isinstance(res, dict) and "error" in res


# -- ReconciliationAgent tests ---------------------------------------------

def test_reconcile_no_issues():
    agent = ReconciliationAgent(tools={})
    ctx = {"step1": [{"value": 1}]}
    rec = agent.reconcile(ctx)
    assert rec["confidence"] == 1.0
    assert rec["issues_found"] == []


def test_reconcile_detects_errors():
    agent = ReconciliationAgent(tools={})
    ctx = {"step1": [{"error": "oops"}]}
    rec = agent.reconcile(ctx)
    assert rec["issues_found"] == ["Error from step1: oops"]
    assert rec["confidence"] < 1.0


# -- SynthesisAgent tests --------------------------------------------------

def test_synthesis_agent_returns_mock_report():
    llm = DummyLLM()
    tmpl = {"medium_response": "RESP"}
    agent = SynthesisAgent(llm, tmpl)
    rec_data = {"issues_found": [], "validated_data": {}, "confidence": 1.0}
    report = agent.synthesize(rec_data, "Q", "medium")
    assert "GEAR IDENTIFICATION RESULTS" in report


# -- MasterAgent tests -----------------------------------------------------

@pytest.fixture
def basic_master(monkeypatch):
    llm = DummyLLM()
    cfg = {
        "system_prompts": {},
        "response_formats": {"medium_response": "OUT"},
        "master_agent": {"max_execution_attempts": 2, "confidence_threshold": 0.5},
        "specialist_agents": {"data_retrieval_agent": {}},
    }
    ma = MasterAgent(llm, datasets={}, config=cfg)
    # Stub out synthesis to return a sentinel
    monkeypatch.setattr(ma.synthesis_agent, "synthesize", lambda *_: "FINAL")
    return ma


def test_master_agent_success(basic_master, monkeypatch):
    ma = basic_master
    # First, plan yields one step
    fake_plan = ([{"step": 1, "tool": "foo", "input": "in", "output_key": "o1"}],
                 "medium")
    monkeypatch.setattr(ma, "_decompose_task", lambda q, e="": fake_plan)
    # Stub retrieval to return valid data
    class StubRetrieval:
        def retrieve(self, *_):
            return [{"foo": "bar"}]
    ma.retrieval_agent = StubRetrieval()
    # Stub reconcile to exceed threshold immediately
    monkeypatch.setattr(ma.reconciliation_agent, "reconcile",
                        lambda ctx: {"confidence": 0.8,
                                     "issues_found": [],
                                     "validated_data": ctx})
    out = ma.run_query("Q")
    assert out == "FINAL"


def test_master_agent_replanning_then_success(basic_master, monkeypatch):

    ma = basic_master
    fake_plan = ([{"step": 1, "tool": "foo", "input": "in", "output_key": "o"}],
                 "medium")
    monkeypatch.setattr(ma, "_decompose_task", lambda q, e="": fake_plan)

    class StubRetrievalAgent:
        def retrieve(self, tool_name, inp):
            return [{"x": "y"}]

    ma.retrieval_agent = StubRetrievalAgent()
    # First reconciliation below threshold, then above
    seq = [
        {"confidence": 0.1, "issues_found": ["err"], "validated_data": {}},
        {"confidence": 0.6, "issues_found": [], "validated_data": {}},
    ]
    monkeypatch.setattr(
        ma.reconciliation_agent,
        "reconcile",
        lambda ctx, seq=seq: seq.pop(0),
    )
    out = ma.run_query("Q")
    assert out == "FINAL"


def test_master_agent_failure_after_max_attempts(basic_master, monkeypatch):
    ma = basic_master
    ma.config["master_agent"]["max_execution_attempts"] = 1
    monkeypatch.setattr(
        ma,
        "_decompose_task",
        lambda q, e="": ([{"tool": "t", "input": "i", "output_key": "o"}],
                        "medium"),
    )

    class StubRetrievalAgent:
        def retrieve(self, tool_name, inp):
            return [{"x": "y"}]

    ma.retrieval_agent = StubRetrievalAgent()
    # Always below threshold
    monkeypatch.setattr(
        ma.reconciliation_agent,
        "reconcile",
        lambda ctx: {"confidence": 0.0,
                     "issues_found": ["bad"],
                     "validated_data": {}},
    )
    out = ma.run_query("Q")
    assert "Could not resolve query with high confidence" in out
    

def test_post_process_data_deduplicates_gears():
    # Create a dummy MasterAgent just to call _post_process_data
    from src.agents.master_agent import MasterAgent

    dummy_llm = DummyLLM()
    cfg = {
        "system_prompts": {},
        "response_formats": {},
        "master_agent": {"max_execution_attempts": 1, "confidence_threshold": 0.0},
        "specialist_agents": {"data_retrieval_agent": {}},
    }
    # empty datasets is fine for this test
    ma = MasterAgent(dummy_llm, datasets={}, config=cfg)

    # Build a fake reconciliation dict
    # 'relationship_tool_step1' simulates the key name you'd get back
    raw_list = [
        {"child": "3DOR1"},
        {"child": "3DOR1"},   # duplicate
        {"child": "3DOR2"},
        {"child": "XNOTGEAR"} # should be filtered out
    ]
    reconciliation = {
        "validated_data": {"relationship_tool_step1": raw_list},
        "issues_found": [],
        "confidence": 1.0,
    }

    # Now post‐process for 'easy'
    out = ma._post_process_data(
        data=reconciliation.copy(), complexity="easy"
    )
    # after dedupe + filter + sort, we expect ['3DOR1','3DOR2']
    cleaned = out["validated_data"]["relationship_tool_step1"]
    assert cleaned == ["3DOR1", "3DOR2"]


def test_run_query_uses_post_processing(monkeypatch):
    from src.agents.master_agent import MasterAgent

    # Set up a master with a low threshold so synthesis is called immediately
    dummy_llm = DummyLLM()
    # Stub synthesis so we can capture the argument it receives
    captured = {}
    def fake_synth(data, query, complexity):
        captured["data"] = data
        captured["complexity"] = complexity
        return "SYNTH_RESP"

    cfg = {
        "system_prompts": {},
        "response_formats": {},
        "master_agent": {"max_execution_attempts": 1, "confidence_threshold": 0.0},
        "specialist_agents": {"data_retrieval_agent": {}},
    }
    ma = MasterAgent(dummy_llm, datasets={}, config=cfg)
    monkeypatch.setattr(ma.synthesis_agent, "synthesize", fake_synth)

    # Stub decomposition to return a single-step plan
    plan = ([{"step": 1, "tool": "relationship_tool", "input": "IGNORED", "output_key": "relationship_tool_step1"}], "easy")
    monkeypatch.setattr(ma, "_decompose_task", lambda q, e="": plan)

    # Stub execution to put our raw list into context
    raw = [
        {"child": "3DOR1"},
        {"child": "3DOR1"},
        {"child": "3DOR3"},
    ]
    class StubRetrieval:
        def retrieve(self, tool, inp):
            return raw
    ma.retrieval_agent = StubRetrieval()

    # Stub reconciliation to wrap our raw list as validated_data
    monkeypatch.setattr(
        ma.reconciliation_agent,
        "reconcile",
        lambda ctx: {
            "validated_data": {"relationship_tool_step1": ctx["relationship_tool_step1"]},
            "issues_found": [],
            "confidence": 1.0,
        }
    )

    # Call run_query
    out = ma.run_query("ANY QUERY")
    assert out == "SYNTH_RESP"

    # Now inspect what we passed to synthesize:
    assert captured["complexity"] == "easy"
    # The raw list had duplicates; post_process should dedupe to ['3DOR1','3DOR3']
    proc = captured["data"]["validated_data"]["relationship_tool_step1"]
    assert proc == ["3DOR1", "3DOR3"]


if __name__ == "__main__":
    pytest.main(["-q", __file__])
