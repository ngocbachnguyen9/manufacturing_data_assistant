import json
from typing import Dict, Any, List, Tuple

# Import all required classes
from .data_retrieval_agent import DataRetrievalAgent
from .reconciliation_agent import ReconciliationAgent
from .synthesis_agent import SynthesisAgent
from src.tools import TOOL_CLASSES
from src.utils.cost_tracker import CostTracker


class MasterAgent:
    """
    Orchestrates the entire query resolution process by managing specialist agents.
    """

    def __init__(
        self,
        llm_provider: Any,
        datasets: Dict[str, Any],
        config: Dict,
        cost_tracker: CostTracker,  # UPDATED: Now accepts a CostTracker instance
    ):
        """
        Initializes the MasterAgent and all its specialist agents and tools.
        """
        self.llm = llm_provider
        self.config = config
        self.prompts = config.get("system_prompts", {})
        self.max_attempts = config.get("master_agent", {}).get(
            "max_execution_attempts", 2
        )
        self.confidence_threshold = config.get("master_agent", {}).get(
            "confidence_threshold", 0.7
        )
        self.cost_tracker = cost_tracker  # NEW: Store the tracker instance

        # Instantiate tools
        self.tools = {
            name: cls(datasets) for name, cls in TOOL_CLASSES.items()
        }

        # Instantiate specialist agents, passing down necessary components
        agent_configs = self.config.get("specialist_agents", {})
        self.retrieval_agent = DataRetrievalAgent(
            self.tools, agent_configs.get("data_retrieval_agent", {})
        )
        self.reconciliation_agent = ReconciliationAgent(self.tools)
        # UPDATED: Pass the cost_tracker to the SynthesisAgent
        self.synthesis_agent = SynthesisAgent(
            llm_provider,
            config.get("response_formats", {}),
            self.cost_tracker,
        )
        print("MasterAgent and specialist agents initialized.")

    def run_query(self, query: str) -> Dict[str, Any]:
        """
        UPDATED: Manages the end-to-end process and returns a full execution trace.
        """
        attempts = 0
        error_context = ""
        final_report = "Failed to generate a satisfactory report."
        reconciliation = {}

        while attempts < self.max_attempts:
            attempts += 1
            print(f"***REMOVED***n[MasterAgent] Execution Attempt: {attempts}")

            plan, complexity = self._decompose_task(query, error_context)
            if not plan:
                final_report = "Failed to create a valid execution plan."
                break

            context = self._execute_plan(plan)
            reconciliation = self.reconciliation_agent.reconcile(context)

            if reconciliation["confidence"] >= self.confidence_threshold:
                processed_data = self._post_process_data(
                    reconciliation, complexity
                )
                final_report = self.synthesis_agent.synthesize(
                    processed_data, query, complexity
                )
                break
            else:
                error_context = f"Previous attempt failed. Issues: {reconciliation['issues_found']}"
                final_report = f"Could not resolve query. Last issues: {reconciliation['issues_found']}"

        print("[MasterAgent] Query processing complete.")
        # NEW: Return the full trace
        return {
            "final_report": final_report,
            "reconciliation_summary": reconciliation,
        }

    def _decompose_task(
        self, query: str, error_context: str = ""
    ) -> Tuple[List[Dict], str]:
        """
        Uses an LLM to decompose the query into a plan, and logs the cost.
        """
        print("- [MasterAgent] Decomposing task into a plan...")
        prompt = f"Based on the query '{query}', create a JSON plan. Tool list: {list(self.tools.keys())}"
        if error_context:
            prompt += f"***REMOVED***nPREVIOUS ATTEMPT FAILED: {error_context}"

        # UPDATED: This now represents a real call to the LLM provider
        response = self.llm.generate(prompt)

        # NEW: Log the cost of this transaction using the tracker
        self.cost_tracker.log_transaction(
            response["input_tokens"], response["output_tokens"], self.llm.model_name
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                parsed_response = json.loads(response["content"])
                plan = parsed_response.get("plan", [])
                complexity = parsed_response.get("complexity", "unknown")
                print(
                    f"- [MasterAgent] Plan created successfully. Complexity: {complexity}."
                )
                return plan, complexity
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(
                        f"- [MasterAgent] JSON parse error (attempt {attempt+1}): {e}. Retrying..."
                    )
                else:
                    print(
                        f"- [MasterAgent] Error: Failed to parse LLM plan response after {max_retries} attempts. {e}"
                    )
                    return [], "unknown"

    def _execute_plan(self, plan: List[Dict]) -> Dict[str, Any]:
        # ... (this method is unchanged) ...
        print("- [MasterAgent] Executing stateful plan...")
        context = {}
        for step in plan:
            tool_name = step.get("tool")
            raw_input = step.get("input")
            output_key = step.get("output_key", f"step_{step['step']}_output")
            try:
                resolved_input = str(raw_input).format(**context)
            except KeyError as e:
                print(
                    f"  - Error: Could not resolve input for step {step['step']}. Missing key: {e}"
                )
                context[output_key] = [{"error": f"Missing context key {e}"}]
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

    def _post_process_data(
        self, data: Dict[str, Any], complexity: str
    ) -> Dict[str, Any]:
        # ... (this method is unchanged) ...
        if complexity == "easy":
            print(
                "  - [MasterAgent] Post-processing for 'easy' task: Deduplicating gears."
            )
            for key, value in data["validated_data"].items():
                if "relationship_tool" in key and isinstance(value, list):
                    gear_ids = [
                        item["child"]
                        for item in value
                        if "child" in item and item["child"].startswith("3DOR")
                    ]
                    unique_gears = sorted(list(set(gear_ids)))
                    data["validated_data"][key] = unique_gears
                    break
        return data