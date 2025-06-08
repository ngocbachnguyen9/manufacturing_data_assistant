import json
from typing import Dict, Any, List, Tuple

from .data_retrieval_agent import DataRetrievalAgent
from .reconciliation_agent import ReconciliationAgent
from .synthesis_agent import SynthesisAgent
from src.tools import TOOL_CLASSES


class MasterAgent:
    """
    Orchestrates the entire query resolution process by managing specialist agents.
    """

    def __init__(
        self, llm_provider: Any, datasets: Dict[str, Any], config: Dict
    ):
        self.llm = llm_provider
        self.config = config
        self.prompts = config.get("system_prompts", {})

        self.tools = {
            name: cls(datasets) for name, cls in TOOL_CLASSES.items()
        }
        print("MasterAgent initialized tools:", list(self.tools.keys()))

        # UPDATED: Pass agent-specific config to DataRetrievalAgent
        agent_configs = self.config.get("specialist_agents", {})
        self.retrieval_agent = DataRetrievalAgent(
            self.tools, agent_configs.get("data_retrieval_agent", {})
        )
        self.reconciliation_agent = ReconciliationAgent(self.tools)
        self.synthesis_agent = SynthesisAgent(
            llm_provider, config.get("response_formats", {})
        )
        print("MasterAgent and specialist agents initialized.")

    def run_query(self, query: str) -> str:
        """
        Manages the end-to-end process of handling a user query.

        Args:
            query: The natural language query from the user.

        Returns:
            A string containing the final, synthesized report.
        """
        print(f"***REMOVED***n[MasterAgent] Received query: '{query}'")
        
        # 1. Decompose task into a plan
        plan, complexity = self._decompose_task(query)
        if not plan:
            return "Failed to create a valid execution plan."

        # 2. Execute the plan to retrieve data
        retrieved_data = self._execute_plan(plan)

        # 3. Reconcile and validate the data
        reconciled_data = self.reconciliation_agent.reconcile(retrieved_data)

        # 4. Synthesize the final report
        final_report = self.synthesis_agent.synthesize(reconciled_data, query, complexity)
        
        print("[MasterAgent] Query processing complete.")
        return final_report

    def _decompose_task(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Uses an LLM to decompose the user query into a structured, step-by-step plan.
        """
        print("- [MasterAgent] Decomposing task into a plan...")
        prompt = self.prompts.get("master_agent_planning", "").format(query=query)
        
        # Mock LLM response for demonstration
        mock_llm_response = """
        {
            "complexity": "easy",
            "plan": [
                {
                    "step": 1,
                    "tool": "relationship_tool",
                    "input": "ORBOX0011",
                    "reason": "Find all gears associated with the order."
                }
            ]
        }
        """
        
        try:
            # In a real implementation: response = self.llm.generate(prompt)
            parsed_response = json.loads(mock_llm_response)
            plan = parsed_response.get("plan", [])
            complexity = parsed_response.get("complexity", "unknown")
            print(f"- [MasterAgent] Plan created successfully. Complexity: {complexity}.")
            return plan, complexity
        except json.JSONDecodeError as e:
            print(f"- [MasterAgent] Error: Failed to parse LLM plan response. {e}")
            return None, "unknown"

    def _execute_plan(self, plan: List[Dict]) -> Dict[str, Any]:
        """
        Executes the steps in the plan using the DataRetrievalAgent.

        Args:
            plan: A list of steps, each specifying a tool and its input.

        Returns:
            A dictionary containing the collected results from all tool executions.
        """
        print("- [MasterAgent] Executing plan...")
        execution_results = {}
        for step in plan:
            tool_name = step.get("tool")
            tool_input = step.get("input")
            if tool_name and tool_input is not None:
                result = self.retrieval_agent.retrieve(tool_name, tool_input)
                execution_results[tool_name] = result
            else:
                print(f"  - Warning: Skipping invalid plan step: {step}")
        
        print("- [MasterAgent] Plan execution complete.")
        return execution_results