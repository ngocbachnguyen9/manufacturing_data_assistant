import json
from typing import Dict, Any


class SynthesisAgent:
    """
    Generates a comprehensive, human-readable report from reconciled data.
    """

    def __init__(self, llm_provider: Any, response_templates: Dict[str, str]):
        """
        Initializes the agent with an LLM provider and response templates.

        Args:
            llm_provider: An instantiated LLM client object.
            response_templates: A dictionary of response format templates.
        """
        self.llm = llm_provider
        self.templates = response_templates

    def synthesize(self, reconciled_data: Dict[str, Any], original_query: str, complexity: str) -> str:
        """
        Uses an LLM to format the reconciled data into a structured report.

        Args:
            reconciled_data: The verified data from the ReconciliationAgent.
            original_query: The user's original query.
            complexity: The determined complexity of the task ('easy', 'medium', 'hard').

        Returns:
            A formatted string containing the final report.
        """
        print("  - [SynthesisAgent] Synthesizing final report...")
        
        template = self.templates.get(f"{complexity}_response", "No template found for {complexity}")
        
        # Prepare data for the LLM prompt
        data_summary = json.dumps(reconciled_data, indent=2)

        prompt = f"""
        Based on the following data, answer the original user query.
        Format your response EXACTLY according to the provided template.

        Original Query: "{original_query}"

        Reconciled Data:
        {data_summary}

        Response Template:
        {template}
        """

        # In a real implementation, this would be an API call
        # response = self.llm.generate(prompt)
        
        # Mock response for demonstration
        print("  - [SynthesisAgent] LLM call would be made here.")
        mock_report = template.format(
            order_id=original_query.split(" ")[-1],
            gear_count=len(reconciled_data.get("validated_data", {}).get("relationship_tool", [])),
            gear_list=reconciled_data.get("validated_data", {}).get("relationship_tool", []),
            data_quality_issues=reconciled_data.get("issues_found", "None"),
            confidence_level=reconciled_data.get("confidence", "N/A"),
            manufacturing_status="Completed",
            recommendations="None"
        )
        
        return mock_report