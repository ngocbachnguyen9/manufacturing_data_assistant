from typing import Dict, Any


class DataRetrievalAgent:
    """
    Executes data retrieval tasks by dispatching to the correct tool,
    passing along relevant search strategy configurations.
    """

    def __init__(self, tools: Dict[str, Any], config: Dict[str, Any]):
        """
        Initializes the agent with available tools and its specific config.

        Args:
            tools: A dictionary mapping tool names to instantiated tool objects.
            config: The configuration specific to this agent (e.g., from
                    agent_config.yaml['specialist_agents']['data_retrieval_agent'])
        """
        self.tools = tools
        self.config = config
        self.fuzzy_enabled = self.config.get("fuzzy_search_enabled", False)
        self.search_threshold = self.config.get("search_threshold", 0.8)
        self.force_fuzzy_on_failure = self.config.get("force_fuzzy_on_failure", False)
        print(
            f"DataRetrievalAgent initialized. Fuzzy Search: {self.fuzzy_enabled}, Force on Failure: {self.force_fuzzy_on_failure}"
        )

    def retrieve(self, tool_name: str, tool_input: Any) -> Any:
        """
        Finds and executes the specified tool, passing search parameters.

        Args:
            tool_name: The name of the tool to execute.
            tool_input: The primary input for the tool.

        Returns:
            The result from the tool's execution.
        """
        print(
            f"  - [DataRetrievalAgent] Executing tool: {tool_name} with input: {tool_input}"
        )
        try:
            if tool_name not in self.tools:
                raise ValueError(f"Tool '{tool_name}' not found.")

            tool = self.tools[tool_name]

            # Pass search strategy parameters to the tool's run method
            result = tool.run(
                tool_input,
                fuzzy_enabled=self.fuzzy_enabled,
                threshold=self.search_threshold,
            )

            # If result indicates failure and force_fuzzy_on_failure is enabled, retry with fuzzy
            if self.force_fuzzy_on_failure and not self.fuzzy_enabled:
                if (isinstance(result, list) and len(result) == 1 and
                    isinstance(result[0], dict) and "error" in result[0]):
                    print(f"  - Exact match failed, retrying with fuzzy matching...")
                    result = tool.run(
                        tool_input,
                        fuzzy_enabled=True,
                        threshold=self.search_threshold,
                    )

            return result
        except Exception as e:
            error_message = f"Error executing tool '{tool_name}': {e}"
            print(f"  - [DataRetrievalAgent] {error_message}")
            return {"error": error_message}