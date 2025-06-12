import json
from typing import Dict, Any, List, Tuple

# Import all required classes
from .data_retrieval_agent import DataRetrievalAgent
from .reconciliation_agent import ReconciliationAgent
from .synthesis_agent import SynthesisAgent
from src.tools import TOOL_CLASSES
from src.utils.cost_tracker import CostTracker


class MasterAgent:
    def __init__(
        self,
        llm_provider: Any,
        datasets: Dict[str, Any],
        config: Dict,
        cost_tracker: CostTracker,
    ):
        self.llm = llm_provider
        self.config = config
        # UPDATED: Load the specific planning prompt from the combined config
        self.planning_prompt_template = config.get("master_agent_planning_prompt", "")
        if not self.planning_prompt_template:
            raise ValueError("Master agent planning prompt not found in configuration.")

        self.max_attempts = config.get("master_agent", {}).get("max_execution_attempts", 2)
        self.confidence_threshold = config.get("master_agent", {}).get("confidence_threshold", 0.7)
        self.cost_tracker = cost_tracker

        self.tools = {name: cls(datasets) for name, cls in TOOL_CLASSES.items()}
        agent_configs = self.config.get("specialist_agents", {})
        self.retrieval_agent = DataRetrievalAgent(self.tools, agent_configs.get("data_retrieval_agent", {}))
        self.reconciliation_agent = ReconciliationAgent(self.tools)
        self.synthesis_agent = SynthesisAgent(llm_provider, config.get("response_formats", {}), self.cost_tracker)
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

            # Always attempt synthesis even with low confidence
            processed_data = self._post_process_data(reconciliation, complexity)
            try:
                base_report = self.synthesis_agent.synthesize(
                    processed_data, query, complexity
                )
                
                # Include reconciliation issues in final report if confidence is low
                if reconciliation["confidence"] < self.confidence_threshold:
                    final_report = (
                        f"⚠️ Low Confidence Report (confidence: {reconciliation['confidence']:.2f}) ⚠️***REMOVED***n"
                        f"Issues found during reconciliation:***REMOVED***n"
                        + "***REMOVED***n".join([f" - {issue}" for issue in reconciliation["issues_found"]])
                        + "***REMOVED***n***REMOVED***n"
                        + base_report
                    )
                else:
                    final_report = base_report
                    
                break  # Break out of retry loop on successful synthesis
            except Exception as e:
                print(f"  - Synthesis failed: {str(e)}")
                # Include reconciliation issues in error context
                error_context = f"Synthesis error: {str(e)}"
                if reconciliation["issues_found"]:
                    error_context += f". Reconciliation issues: {reconciliation['issues_found']}"
                final_report = f"Synthesis failed: {str(e)}"

        print("[MasterAgent] Query processing complete.")
        # NEW: Return the full trace
        return {
            "final_report": final_report,
            "reconciliation_summary": reconciliation,
        }
    
    # NEW HELPER METHOD
    def _get_tool_descriptions(self) -> str:
        """Generates a formatted string of tool descriptions for the prompt."""
        descriptions = []
        for name, tool_class in TOOL_CLASSES.items():
            # Get the first line of the class docstring as the description
            docstring = (tool_class.__doc__ or "No description.").strip().split('***REMOVED***n')[0]
            descriptions.append(f"- `{name}`: {docstring}")
        return "***REMOVED***n".join(descriptions)

    # REPLACED METHOD
    def _decompose_task(
        self, query: str, error_context: str = ""
    ) -> Tuple[List[Dict], str]:
        """
        Uses a detailed few-shot prompt to get a reliable JSON plan from the LLM.
        """
        print("- [MasterAgent] Decomposing task into a plan...")

        tool_descriptions = self._get_tool_descriptions()
        prompt = self.planning_prompt_template.format(
            tool_descriptions=tool_descriptions,
            query=query
        )

        if error_context:
            prompt += f"***REMOVED***n***REMOVED***nIMPORTANT: Your previous attempt failed with these issues: {error_context}. Create a new, corrected plan."

        response = self.llm.generate(prompt)
        self.cost_tracker.log_transaction(
            response["input_tokens"], response["output_tokens"], self.llm.model_name
        )

        try:
            # The LLM might return the JSON inside a markdown code block, so we clean it.
            content = response["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            parsed_response = json.loads(content)
            plan = parsed_response.get("plan", [])
            complexity = parsed_response.get("complexity", "unknown")
            print(f"- [MasterAgent] Plan created successfully. Complexity: {complexity}.")
            return plan, complexity
        except (json.JSONDecodeError, IndexError) as e:
            print(f"- [MasterAgent] Error: Failed to parse LLM plan response. Content was: {response['content']}. Error: {e}")
            return [], "unknown"

    def _execute_plan(self, plan: List[Dict]) -> Dict[str, Any]:
        print("- [MasterAgent] Executing stateful plan...")
        context = {}
        for step in plan:
            tool_name = step.get("tool")
            raw_input = step.get("input")
            output_key = step.get("output_key", f"step_{step['step']}_output")
            
            # Handle missing parameters by trying raw input first
            try:
                resolved_input = str(raw_input).format(**context) if raw_input else ""
            except KeyError:
                # Use raw input as fallback if context is missing
                resolved_input = raw_input
                print(f"  - Warning: Using raw input for step {step['step']} due to missing context keys")
            
            if tool_name:
                try:
                    result = self.retrieval_agent.retrieve(tool_name, resolved_input)
                    context[output_key] = result
                except Exception as e:
                    print(f"  - Error in {tool_name}: {str(e)}")
                    context[output_key] = [{"error": str(e)}]
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