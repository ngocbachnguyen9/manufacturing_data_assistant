from typing import Dict, Any


class DataRetrievalAgent:
    """
    Executes specific data retrieval tasks by dispatching to the correct tool.
    """

    def __init__(self, tools: Dict[str, Any]):
        """
        Initializes the agent with a dictionary of available tools.

        Args:
            tools: A dictionary mapping tool names to instantiated tool objects.
        """
        self.tools = tools
        print("DataRetrievalAgent initialized with tools:", list(self.tools.keys()))

    def retrieve(self, tool_name: str, tool_input: Any) -> Any:
        """
        Finds and executes the specified tool with the given input.

        Args:
            tool_name: The name of the tool to execute (e.g., 'location_query_tool').
            tool_input: The input to pass to the tool's run method.

        Returns:
            The result from the tool's execution.
        """
        print(f"  - [DataRetrievalAgent] Executing tool: {tool_name} with input: {tool_input}")
        try:
            if tool_name not in self.tools:
                raise ValueError(f"Tool '{tool_name}' not found.")
            
            tool = self.tools[tool_name]
            result = tool.run(tool_input)
            return result
        except Exception as e:
            error_message = f"Error executing tool '{tool_name}': {e}"
            print(f"  - [DataRetrievalAgent] {error_message}")
            return {"error": error_message}