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
        self.max_attempts = config.get("master_agent", {}).get(
            "max_execution_attempts", 2
        )
        self.confidence_threshold = config.get("master_agent", {}).get(
            "confidence_threshold", 0.7
        )

        self.tools = {
            name: cls(datasets) for name, cls in TOOL_CLASSES.items()
        }
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
        # ... (re-planning loop remains unchanged) ...
        attempts = 0
        error_context = ""
        final_report = "Failed to generate a satisfactory report."

        while attempts < self.max_attempts:
            attempts += 1
            print(f"***REMOVED***n[MasterAgent] Execution Attempt: {attempts}")

            plan, complexity = self._decompose_task(query, error_context)
            if not plan:
                return "Failed to create a valid execution plan."

            context = self._execute_plan(plan)
            reconciliation = self.reconciliation_agent.reconcile(context)

            if reconciliation["confidence"] >= self.confidence_threshold:
                print(
                    "[MasterAgent] Confidence threshold met. Proceeding to synthesis."
                )
                # NEW: Add a post-processing step before synthesis
                processed_data = self._post_process_data(
                    reconciliation, complexity
                )
                final_report = self.synthesis_agent.synthesize(
                    processed_data, query, complexity
                )
                break
            else:
                print(
                    "[MasterAgent] Confidence threshold not met. Triggering re-planning."
                )
                error_context = f"Previous attempt failed with low confidence. Issues found: {reconciliation['issues_found']}. Please create a new plan to address these issues."
                final_report = f"Could not resolve query with high confidence. Last issues found: {reconciliation['issues_found']}"

        print("[MasterAgent] Query processing complete.")
        return final_report
    
    def _post_process_data(
        self, data: Dict[str, Any], complexity: str
    ) -> Dict[str, Any]:
        """
        Cleans and transforms data based on task complexity before synthesis.
        """
        if complexity == "easy":
            print("  - [MasterAgent] Post-processing for 'easy' task: Deduplicating gears.")
            # Find the output from the relationship tool
            for key, value in data["validated_data"].items():
                if "relationship_tool" in key and isinstance(value, list):
                    # Extract just the 'child' IDs which are the gears
                    gear_ids = [
                        item["child"]
                        for item in value
                        if "child" in item and item["child"].startswith("3DOR")
                    ]
                    # Deduplicate and sort the list
                    unique_gears = sorted(list(set(gear_ids)))
                    # Replace the raw tool output with the clean list
                    data["validated_data"][key] = unique_gears
                    break  # Assume only one relationship step for easy tasks
        return data

    def _decompose_task(
        self, query: str, error_context: str = ""
    ) -> Tuple[List[Dict], str]:
        """
        Uses an LLM to decompose the query into a plan, considering past errors.
        """
        print("- [MasterAgent] Decomposing task into a plan...")
        # NEW: A more realistic mock plan showing dependency
        mock_llm_response = """
        {
            "complexity": "medium",
            "plan": [
                {
                    "step": 1,
                    "tool": "relationship_tool",
                    "input": "ORBOX0014",
                    "output_key": "order_gears",
                    "reason": "Find all gears for the order."
                },
                {
                    "step": 2,
                    "tool": "location_query_tool",
                    "input": "{order_gears[0][child]}",
                    "output_key": "gear_location_history",
                    "reason": "Find the location history of the first gear found."
                }
            ]
        }
        """
        try:
            parsed_response = json.loads(mock_llm_response)
            return parsed_response.get("plan", []), parsed_response.get(
                "complexity", "unknown"
            )
        except json.JSONDecodeError:
            return [], "unknown"

    def _execute_plan(self, plan: List[Dict]) -> Dict[str, Any]:
        """
        Executes the plan, passing context between dependent steps.
        """
        print("- [MasterAgent] Executing stateful plan...")
        context = {}
        for step in plan:
            tool_name = step.get("tool")
            raw_input = step.get("input")
            if "output_key" in step:
                output_key = step["output_key"]
            else:
                #use .get here so missing 'step' just yields '0'
                output_key = f"step_{step.get('step', 0)}_output"
            try:
                # NEW: Resolve input from context if it's a template
                resolved_input = str(raw_input).format(**context)
            except KeyError as e:
                print(
                    f"  - Error: Could not resolve input for step {step['step']}. Missing key: {e}"
                )
                context[
                    output_key
                ] = [{"error": f"Missing context key {e}"}]
                continue

            if tool_name:
                result = self.retrieval_agent.retrieve(
                    tool_name, resolved_input
                )
                context[output_key] = result
            else:
                print(f"  - Warning: Skipping invalid plan step: {step}")

        print("- [MasterAgent] Plan execution complete.")
        return context