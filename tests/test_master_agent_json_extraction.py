import re
import json
from src.agents.master_agent import MasterAgent
import yaml
from unittest.mock import MagicMock

# Mock dependencies for MasterAgent
mock_llm_provider = MagicMock()
mock_datasets = MagicMock()

# Load minimal configuration
mock_config = {
    "master_agent_planning_prompt": "Test planning prompt"
}

mock_cost_tracker = MagicMock()

# Create agent instance
agent = MasterAgent(
    llm_provider=mock_llm_provider,
    datasets=mock_datasets,
    config=mock_config,
    cost_tracker=mock_cost_tracker
)

# Test cases
test_cases = [
    ("```json***REMOVED***n{***REMOVED***"plan***REMOVED***": ***REMOVED***"test***REMOVED***"}***REMOVED***n```", {"plan": "test"}),
    ("```***REMOVED***n{***REMOVED***"plan***REMOVED***": ***REMOVED***"test***REMOVED***"}***REMOVED***n```", {"plan": "test"}),
    ("{***REMOVED***"plan***REMOVED***": ***REMOVED***"test***REMOVED***"}", {"plan": "test"}),
    ("No JSON here", None),
    ("```json***REMOVED***n{invalid json}***REMOVED***n```", None),
]

# Run tests
for i, (input_text, expected) in enumerate(test_cases, 1):
    result = agent._extract_json_from_string(input_text)
    status = "PASS" if result == expected else "FAIL"
    print(f"Test {i}: {status}")
    print(f"Input: {repr(input_text)}")
    print(f"Expected: {expected}")
    print(f"Actual: {result}***REMOVED***n")