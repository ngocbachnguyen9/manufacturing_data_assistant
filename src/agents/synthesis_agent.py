import json
from typing import Dict, Any


class SynthesisAgent:
    """
    Generates a comprehensive, human-readable report from reconciled data.
    """

    def __init__(self, llm_provider: Any, response_templates: Dict[str, str]):
        self.llm = llm_provider
        self.templates = response_templates

    def synthesize(
        self,
        reconciled_data: Dict[str, Any],
        original_query: str,
        complexity: str,
    ) -> str:
        """
        Uses an LLM to format the reconciled data into a structured report.
        """
        print("  - [SynthesisAgent] Synthesizing final report...")

        template = self.templates.get(
            f"{complexity}_response",
            "Please provide a summary of your findings.",
        )
        data_summary = json.dumps(reconciled_data, indent=2)

        # NEW: A robust prompt that asks the LLM to perform the synthesis
        prompt = f"""
        You are a manufacturing data analyst. Your task is to answer the user's query based on the provided data and format the response using the given template.

        **Original Query:**
        {original_query}

        **Reconciled Data from Tools (JSON format):**
        {data_summary}

        **Instructions:**
        1. Analyze the "validated_data" to find the direct answer to the query.
        2. Review the "issues_found" list to identify any data quality problems.
        3. Note the overall "confidence" score.
        4. Populate the response template below with this information in a clear, human-readable format. If data is missing, state that clearly.

        **Response Template:**
        {template}
        """

        # In a real implementation, this would be an API call to the LLM
        # response = self.llm.generate(prompt)
        # return response

        # NEW: A more realistic mock response demonstrating synthesis
        print("  - [SynthesisAgent] LLM call would be made with a robust prompt.")
        mock_llm_report = f"""
## GEAR IDENTIFICATION RESULTS

**Order ID:** ORBOX0014
**Total Gears Found:** 10

**Gear List:**
- 3DOR10001
- 3DOR10003
- 3DOR10004
- 3DOR10005
- 3DOR10008
- ... (and 5 more)

**Data Quality Assessment:**
- Issues Detected: {reconciled_data.get('issues_found', 'None')}
- Confidence Level: {reconciled_data.get('confidence', 1.0):.2f}

**Manufacturing Status:**
All gears appear to have reached the 'Parts Warehouse' or 'Goods Out' stations, indicating completion.

**Recommendations:**
No issues detected that require action.
"""
        return mock_llm_report