import json
import re
import yaml
from typing import Dict, Any, List, Tuple, Optional

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
        # Load planning prompt from config
        planning_prompt = config.get("master_agent_planning_prompt", "")
        if not planning_prompt:
            raise ValueError("Master agent planning prompt not found in configuration.")
        
        # Handle case where prompt might be a dict (if loaded from YAML)
        if isinstance(planning_prompt, dict):
            self.planning_prompt_template = planning_prompt.get("master_agent_planning_prompt", "")
        else:
            self.planning_prompt_template = planning_prompt
            
        if not self.planning_prompt_template:
            raise ValueError("Master agent planning prompt is empty.")
            
        # Load task-specific prompts
        self.task_prompts = self._load_task_prompts()

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
        import time
        start_time = time.time()
        max_execution_time = 300  # 5 minutes timeout per task

        attempts = 0
        error_context = ""
        final_report = "Failed to generate a satisfactory report."
        reconciliation = {}

        while attempts < self.max_attempts:
            # Check timeout
            if time.time() - start_time > max_execution_time:
                print(f"[MasterAgent] ⏰ Task timeout after {max_execution_time}s")
                final_report = f"Task timed out after {max_execution_time} seconds"
                break

            attempts += 1
            print(f"***REMOVED***n[MasterAgent] Execution Attempt: {attempts}")

            plan, complexity = self._decompose_task(query, error_context)
            if not plan:
                final_report = "Failed to create a valid execution plan."
                break

            context = self._execute_plan(plan)
            reconciliation = self.reconciliation_agent.reconcile(context)
            
            # Handle critical data issues by retrying with alternative tools
            if reconciliation.get("critical_issue", False) and attempts < self.max_attempts:
                error_context = "Critical data issue detected: " + "; ".join(reconciliation["issues_found"])
                print(f"  - Critical issue found. Retrying with context: {error_context}")
                # Skip synthesis and retry immediately
                continue

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

    def _extract_json_from_string(self, text: str) -> Optional[Dict]:
        """
        Finds and parses the first valid JSON object within a string.
        Handles markdown code blocks ```json ... ``` and other text.
        """
        # Pattern to find a JSON object enclosed in triple backticks (with or without language specifier)
        # We'll look for either ```json ... ``` or just ``` ... ``` containing a JSON object
        # Also handle cases without backticks
        patterns = [
            r"```json***REMOVED***s*(***REMOVED***{.*?***REMOVED***})***REMOVED***s*```",  # ```json { ... } ```
            r"```***REMOVED***s*(***REMOVED***{.*?***REMOVED***})***REMOVED***s*```",      # ``` { ... } ```
            r"(***REMOVED***{.****REMOVED***})"                     # raw JSON without backticks
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try the next pattern
                    continue

        return None

    def _decompose_task(
        self, query: str, error_context: str = ""
    ) -> Tuple[List[Dict], str]:
        """
        Uses a detailed few-shot prompt and robust JSON extraction to get a
        reliable plan from the LLM.
        """
        print("- [MasterAgent] Decomposing task into a plan...")

        base_prompt = self.planning_prompt_template.format(
            tool_descriptions=self._get_tool_descriptions()
        )
        final_prompt = f"{base_prompt}***REMOVED***n***REMOVED***n--- USER QUERY ---***REMOVED***n{query}"

        if error_context:
            final_prompt += f"***REMOVED***n***REMOVED***nIMPORTANT: Your previous attempt failed with these issues: {error_context}. Create a new, corrected plan."

        print(f"- [MasterAgent] DEBUG: Final planning prompt sent to LLM:***REMOVED***n{final_prompt}")

        response = self.llm.generate(final_prompt)
        self.cost_tracker.log_transaction(
            response["input_tokens"], response["output_tokens"], self.llm.model_name
        )
        content = response["content"]
        print(f"- [MasterAgent] DEBUG: Raw LLM planning response: {content}")

        # Use the new robust JSON extraction method
        parsed_response = self._extract_json_from_string(content)

        if not parsed_response:
            print(f"- [MasterAgent] Error: Failed to parse LLM plan. No JSON object found in the response.")
            return [], "unknown"

        plan = parsed_response.get("plan", [])
        complexity = parsed_response.get("complexity", "unknown")

        if not plan or not isinstance(plan, list):
            print(f"- [MasterAgent] Error: LLM response contained JSON but no valid 'plan' key.")
            return [], complexity

        print(f"- [MasterAgent] Plan created successfully. Complexity: {complexity}.")
        return plan, complexity

    def _execute_plan(self, plan: List[Dict]) -> Dict[str, Any]:
        print("- [MasterAgent] Executing stateful plan...")
        context = {}
        for step in plan:
            tool_name = step.get("tool")
            raw_input = step.get("input")
            output_key = step.get("output_key", f"step_{step['step']}_output")
            step_num = step.get("step", "unknown")

            # Check if this step has unresolved dependencies
            if self._has_unresolved_dependencies(raw_input, context):
                print(f"  - Skipping step {step_num} ({tool_name}): Missing required dependencies")
                context[output_key] = [{"error": "Skipped due to missing dependencies", "dependency_failed": True}]
                continue

            # Resolve input by substituting context variables
            resolved_input, resolution_success = self._resolve_input_variables_enhanced(raw_input, context)

            # If resolution failed and this step depends on previous steps, try fallback
            if not resolution_success and self._step_has_dependencies(raw_input):
                fallback_input = self._get_fallback_input(raw_input, step, context)
                if fallback_input:
                    print(f"  - Using fallback input for step {step_num}: {fallback_input}")
                    resolved_input = fallback_input
                else:
                    print(f"  - Skipping step {step_num} ({tool_name}): No fallback available")
                    context[output_key] = [{"error": "No fallback input available", "dependency_failed": True}]
                    continue

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

    def _resolve_input_variables_enhanced(self, raw_input: str, context: Dict[str, Any]) -> tuple[str, bool]:
        """
        Enhanced variable resolution that returns both resolved input and success status
        """
        if not raw_input or not isinstance(raw_input, str):
            return raw_input, True

        import re

        # Pattern to match {variable_name['key']} or {variable_name["key"]}
        pattern = r'***REMOVED***{([^}]+)***REMOVED***[[***REMOVED***'"](.*?)[***REMOVED***'"]***REMOVED***]***REMOVED***}'
        resolution_success = True

        def replace_variable(match):
            nonlocal resolution_success
            var_name = match.group(1)
            key = match.group(2)

            if var_name in context:
                var_value = context[var_name]

                # Handle different data structures returned by tools
                if isinstance(var_value, dict) and key in var_value:
                    return str(var_value[key])
                elif isinstance(var_value, list) and len(var_value) > 0:
                    if isinstance(var_value[0], dict) and key in var_value[0]:
                        return str(var_value[0][key])
                    # Handle case where list contains non-dict items
                    elif len(var_value) == 1 and not isinstance(var_value[0], dict):
                        return str(var_value[0])

            print(f"  - Warning: Could not resolve variable {var_name}['{key}'] in context")
            resolution_success = False
            return match.group(0)  # Return original if can't resolve

        try:
            resolved = re.sub(pattern, replace_variable, raw_input)
            if resolved != raw_input and resolution_success:
                print(f"  - Resolved input: '{raw_input}' -> '{resolved}'")
            return resolved, resolution_success
        except Exception as e:
            print(f"  - Warning: Error resolving variables in '{raw_input}': {str(e)}")
            return raw_input, False

    def _resolve_input_variables(self, raw_input: str, context: Dict[str, Any]) -> str:
        """
        Legacy method for backward compatibility
        """
        resolved, _ = self._resolve_input_variables_enhanced(raw_input, context)
        return resolved

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

    def _load_task_prompts(self) -> dict:
        """Load task-specific prompts from task_prompts_variations.yaml"""
        path = self.config.get("task_prompts_path", "config/task_prompts_variations.yaml")
        with open(path, 'r') as f:
            return yaml.safe_load(f).get("prompt_variations", {})

    def _determine_task_type(self, query: str) -> str:
        """Determine task complexity based on query content"""
        query = query.lower()
        if "verify" in query or "compliance" in query or "certificate" in query:
            return "hard"
        elif "printer" in query or "count" in query or "machine" in query:
            return "medium"
        return "easy"

    def _select_prompt_template(self, task_type: str, has_data_issues: bool) -> str:
        """Select appropriate prompt template based on task type and data quality"""
        if task_type not in self.task_prompts:
            return self.task_prompts.get("easy", {}).get("base_prompt", "")
            
        task_prompts = self.task_prompts[task_type]
        key = "with_data_quality_issues" if has_data_issues else "base_prompt"
        return task_prompts.get(key, task_prompts.get("base_prompt", ""))

    def _format_task_prompt(self, template: str, query: str) -> str:
        """Extract parameters from query and format prompt template"""
        # Extract parameters from query using regex
        params = {}
        
        # Packing list ID
        if "packing_list_id" in template:
            if match := re.search(r"packing***REMOVED***s*list***REMOVED***s*(***REMOVED***w+)", query, re.IGNORECASE):
                params["packing_list_id"] = match.group(1)
                
        # Part ID
        if "part_id" in template:
            if match := re.search(r"part***REMOVED***s*(***REMOVED***w+)", query, re.IGNORECASE):
                params["part_id"] = match.group(1)
                
        # Order ID
        if "order_id" in template:
            if match := re.search(r"order***REMOVED***s*(***REMOVED***w+)", query, re.IGNORECASE):
                params["order_id"] = match.group(1)
        
        return template.format(**params)

    def _has_unresolved_dependencies(self, raw_input: str, context: Dict[str, Any]) -> bool:
        """
        Check if the input has variable dependencies that cannot be resolved
        """
        if not raw_input or not isinstance(raw_input, str):
            return False

        import re
        pattern = r'***REMOVED***{([^}]+)***REMOVED***[[***REMOVED***'"](.*?)[***REMOVED***'"]***REMOVED***]***REMOVED***}'
        matches = re.findall(pattern, raw_input)

        for var_name, key in matches:
            if var_name not in context:
                return True
            var_value = context[var_name]
            # Check if the variable exists but contains error data
            if isinstance(var_value, list) and len(var_value) > 0:
                if isinstance(var_value[0], dict) and "error" in var_value[0]:
                    # Check if it's a dependency failure (should skip) vs recoverable error
                    if var_value[0].get("dependency_failed", False):
                        return True
                    # For other errors, check if we have the required key anyway
                    if key not in var_value[0]:
                        return True
        return False

    def _step_has_dependencies(self, raw_input: str) -> bool:
        """
        Check if a step input contains variable references
        """
        if not raw_input or not isinstance(raw_input, str):
            return False
        import re
        pattern = r'***REMOVED***{([^}]+)***REMOVED***[[***REMOVED***'"](.*?)[***REMOVED***'"]***REMOVED***]***REMOVED***}'
        return bool(re.search(pattern, raw_input))

    def _get_fallback_input(self, raw_input: str, step: Dict, context: Dict[str, Any]) -> Optional[str]:
        """
        Generate fallback input when variable resolution fails
        """
        tool_name = step.get("tool", "")

        # Extract any direct IDs from the original raw_input
        import re

        # Look for order IDs like ORBOX00117
        order_match = re.search(r'ORBOX***REMOVED***d+', raw_input)
        if order_match:
            order_id = order_match.group(0)

            # Tool-specific fallback strategies
            if tool_name == "document_parser_tool":
                return order_id  # Use order ID directly
            elif tool_name == "location_query_tool":
                return order_id  # Use order ID directly
            elif tool_name == "relationship_tool":
                return order_id  # Use order ID directly
            elif tool_name == "worker_data_tool":
                return order_id  # Use order ID directly

        # Look for part IDs like 3DOR100xxx
        part_match = re.search(r'3DOR***REMOVED***d+', raw_input)
        if part_match and tool_name in ["relationship_tool", "worker_data_tool"]:
            return part_match.group(0)

        # Look for any successful previous step outputs that might be usable
        for key, value in context.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    # Check for structured error with order_id
                    if "error" in value[0] and "order_id" in value[0]:
                        return value[0]["order_id"]
                    # Check for successful data
                    elif "error" not in value[0]:
                        if "order_id" in value[0]:
                            return value[0]["order_id"]
                        elif "id" in value[0]:
                            return value[0]["id"]

        return None