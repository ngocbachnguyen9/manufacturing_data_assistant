# tests/test_agents.py

import os
import sys
import json
import pytest

# Ensure src/ is on PYTHONPATH so we can import agents and utils
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.agents.data_retrieval_agent import DataRetrievalAgent
from src.agents.reconciliation_agent import ReconciliationAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.agents.master_agent import MasterAgent
from src.utils.cost_tracker import CostTracker


# -- Dummy collaborators ---------------------------------------------------

class DummyTool:
    """A fake tool that returns a preset value or raises if needed."""
    def __init__(self, result):
        self.result = result

    def run(self, *_args, **_kwargs):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class DummyCostTracker:
    """Captures log_transaction calls for verification."""
    def __init__(self):
        self.calls = []

    def log_transaction(self, input_tokens, output_tokens, model_name):
        self.calls.append((input_tokens, output_tokens, model_name))


class DummyLLM:
    """
    Fake LLM client: returns a dict with 'content',
    'input_tokens', 'output_tokens'; has a .model_name.
    """
    def __init__(self, content="{}", in_toks=1, out_toks=1, model="dummy-model"):
        self.last_prompt = None
        self.content = content
        self.input_tokens = in_toks
        self.output_tokens = out_toks
        self.model_name = model

    def generate(self, prompt: str) -> dict:
        self.last_prompt = prompt
        return {
            "content": self.content,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens
        }


# -- DataRetrievalAgent tests ----------------------------------------------

def test_data_retrieval_success_and_error():
    tools = {"tool_a": DummyTool("result_a")}
    agent = DataRetrievalAgent(tools, config={})
    out = agent.retrieve("tool_a", "input_x")
    assert out == "result_a"

    agent = DataRetrievalAgent({}, config={})
    res = agent.retrieve("missing_tool", "in")
    assert isinstance(res, dict)
    assert "error" in res


# -- ReconciliationAgent tests ---------------------------------------------

def test_reconcile_no_issues_and_with_error():
    agent = ReconciliationAgent(tools={})
    ctx = {"step1": [{"value": 1}]}
    rec = agent.reconcile(ctx)
    assert rec["confidence"] == 1.0
    assert rec["issues_found"] == []

    ctx_err = {"step1": [{"error": "failure"}]}
    rec2 = agent.reconcile(ctx_err)
    assert rec2["issues_found"] == ["Error from step1: failure"]
    assert rec2["confidence"] < 1.0


# -- SynthesisAgent tests --------------------------------------------------

def test_synthesis_logs_cost_and_returns_content():
    # Prepare dummy LLM and cost tracker
    llm = DummyLLM(content="REPLY", in_toks=5, out_toks=10, model="mymodel")
    tracker = DummyCostTracker()
    templates = {"easy_response": "Template"}
    agent = SynthesisAgent(llm, templates, tracker)

    result = agent.synthesize(
        reconciled_data={"foo": "bar"},
        original_query="Q",
        complexity="easy"
    )
    # Should return the LLM content
    assert result == "REPLY"
    # And log exactly one transaction
    assert len(tracker.calls) == 1
    in_t, out_t, mname = tracker.calls[0]
    assert in_t == 5 and out_t == 10 and mname == "mymodel"


# -- MasterAgent internal tests --------------------------------------------

def test_decompose_logs_cost_and_parses_plan(dummy_llm=None, dummy_tracker=None):
    # Setup dummy LLM(plan) and cost tracker
    plan = {"complexity": "easy",
            "plan": [{"step": 1, "tool": "foo", "input": "bar", "output_key": "o1"}]}
    llm = DummyLLM(content=json.dumps(plan), in_toks=3, out_toks=7, model="modelX")
    tracker = DummyCostTracker()
    cfg = {
        "system_prompts": {},
        "master_agent": {"max_execution_attempts": 1,
                         "confidence_threshold": 0.0},
        "specialist_agents": {"data_retrieval_agent": {}},
    }
    ma = MasterAgent(llm, datasets={}, config=cfg, cost_tracker=tracker)

    plan_out, comp = ma._decompose_task("QUERY", "")
    assert comp == "easy"
    assert isinstance(plan_out, list)
    # cost logged once
    assert tracker.calls == [(3, 7, "modelX")]


def test_post_process_deduplicates_gears():
    # Test the easy‐task post‐processing logic
    llm = DummyLLM()
    tracker = DummyCostTracker()
    cfg = {
        "system_prompts": {},
        "master_agent": {"max_execution_attempts": 1,
                         "confidence_threshold": 0.0},
        "specialist_agents": {"data_retrieval_agent": {}},
    }
    ma = MasterAgent(llm, datasets={}, config=cfg, cost_tracker=tracker)

    reconciliation = {
        "validated_data": {
            "relationship_tool_step": [
                {"child": "3DOR1"}, {"child": "3DOR1"}, {"child": "X"}
            ]
        },
        "issues_found": [],
        "confidence": 1.0
    }
    processed = ma._post_process_data(reconciliation, "easy")
    cleaned = processed["validated_data"]["relationship_tool_step"]
    assert cleaned == ["3DOR1"]


# -- MasterAgent orchestration tests ---------------------------------------

def test_run_query_triggers_post_and_synthesis(monkeypatch):
    # Setup MasterAgent with dummy LLM & tracker
    llm = DummyLLM(content="{}", in_toks=1, out_toks=1, model="m")
    tracker = DummyCostTracker()
    cfg = {
        "system_prompts": {},
        "master_agent": {"max_execution_attempts": 1,
                         "confidence_threshold": 0.0},
        "specialist_agents": {"data_retrieval_agent": {}},
    }
    ma = MasterAgent(llm, datasets={}, config=cfg, cost_tracker=tracker)

    # Stub _decompose_task to return a single easy step
    monkeypatch.setattr(
        ma,
        "_decompose_task",
        lambda q, e="": (
            [{"step": 1,
              "tool": "relationship_tool",
              "input": "X",
              "output_key": "relationship_tool"}],
            "easy"
        )
    )
    # Stub retrieval to return duplicate gears
    class StubRetrieval:
        def retrieve(self, t, inp):
            return [{"child": "3DOR1"}, {"child": "3DOR1"}]
    ma.retrieval_agent = StubRetrieval()

    # Stub reconcile to wrap that into validated_data
    monkeypatch.setattr(
        ma.reconciliation_agent,
        "reconcile",
        lambda ctx: {
            "validated_data": {"relationship_tool": ctx["relationship_tool"]},
            "issues_found": [],
            "confidence": 1.0
        }
    )
    # Capture what synthesize receives
    captured = {}
    def fake_synth(data, query, complexity):
        captured["data"] = data
        captured["complexity"] = complexity
        return "DONE"
    monkeypatch.setattr(ma.synthesis_agent, "synthesize", fake_synth)

    result = ma.run_query("any")
    assert isinstance(result, dict)
    assert result["final_report"] == "DONE"
    # Post‐processing should have deduped to ['3DOR1']
    assert captured["data"]["validated_data"]["relationship_tool"] == ["3DOR1"]