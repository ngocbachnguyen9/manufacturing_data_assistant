import json
import random
import re
import os
import openai
import anthropic
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

class MockLLMProvider:
    """
    A mock LLM provider that simulates API calls, returning pre-defined
    responses and token counts for testing and cost tracking.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Simulates a call to an LLM API.
        It dynamically generates a mock plan or a mock synthesis report
        based on the prompt's content.
        """
        # Simulate token usage
        input_tokens = len(prompt.split())
        output_tokens = random.randint(50, 250)

        # Determine if the prompt is for planning or synthesis based on keywords
        if "create a precise, step-by-step execution plan" in prompt:
            # This is a planning prompt -> return a mock JSON plan
            content = self._get_mock_plan(prompt)
        elif "You are a manufacturing data analyst. Your task is to answer" in prompt:
            # This is a synthesis prompt -> return a mock synthesized report
            content = self._get_mock_synthesis_report(prompt)
        else:
            # Fallback for unexpected prompts (e.g., judge prompt in evaluation)
            content = "Mock response for unknown prompt type."


        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _get_mock_plan(self, prompt: str) -> str:
        """
        Returns a mock plan dynamically, extracting the entity ID and complexity
        from the prompt to ensure consistency. Always returns a valid JSON object.
        """
        # Try to match known query types
        # 1. Packing List
        match = re.search(r"Packing List***REMOVED***s*(PL***REMOVED***w+)", prompt)
        if match:
            entity_id = match.group(1)
            return json.dumps({
                "complexity": "easy",
                "plan": [
                    {
                        "step": 1,
                        "tool": "packing_list_parser_tool",
                        "input": entity_id,
                        "output_key": "step_1_order_id",
                        "reason": "Parse the packing list to find the corresponding Order ID."
                    },
                    {
                        "step": 2,
                        "tool": "relationship_tool",
                        "input": "{step_1_order_id['order_id']}",
                        "output_key": "step_2_gear_list",
                        "reason": "Use the Order ID to find all related gear parts."
                    }
                ]
            })

        # 2. Part/Printer
        match = re.search(r"Part***REMOVED***s*(3DOR***REMOVED***w+)", prompt)
        if match:
            entity_id = match.group(1)
            return json.dumps({
                "complexity": "medium",
                "plan": [
                    {
                        "step": 1,
                        "tool": "relationship_tool",
                        "input": entity_id,
                        "output_key": "step_1_printer_info",
                        "reason": "Find the parent printer associated with the given part ID."
                    },
                    {
                        "step": 2,
                        "tool": "relationship_tool",
                        "input": "{step_1_printer_info['parent']}",
                        "output_key": "step_2_all_parts_on_printer",
                        "reason": "Use the found printer ID to query for all other parts associated with it to get a total count."
                    }
                ]
            })

        # 3. Order/ARC document
        match = re.search(r"Order***REMOVED***s*(ORBOX***REMOVED***w+)", prompt)
        if match:
            entity_id = match.group(1)
            return json.dumps({
                "complexity": "hard",
                "plan": [
                    {
                        "step": 1,
                        "tool": "document_parser_tool",
                        "input": entity_id,
                        "output_key": "step_1_arc_date",
                        "reason": "Parse the ARC document to get the certificate completion date."
                    },
                    {
                        "step": 2,
                        "tool": "location_query_tool",
                        "input": entity_id,
                        "output_key": "step_2_warehouse_arrival",
                        "reason": "Find the arrival date of the order at the Parts Warehouse."
                    }
                ]
            })

        # Default fallback: always return a valid JSON object
        return json.dumps({
            "complexity": "unknown",
            "plan": [],
            "reason": "Mock plan not defined for this specific query type."
        })

    def _get_mock_synthesis_report(self, prompt: str) -> str:
        """Returns a generic synthesized report based on the prompt content."""
        # Extract original query from synthesis prompt if possible
        query_match = re.search(r"***REMOVED*******REMOVED****Original Query:***REMOVED*******REMOVED*******REMOVED***n(.*?)***REMOVED***n***REMOVED***n***REMOVED*******REMOVED****Reconciled Data from Tools", prompt, re.DOTALL)
        original_query = query_match.group(1).strip() if query_match else "Unknown query"

        # Simple mock logic: if the original query is about gears, synthesize a gear report
        if "gears" in original_query.lower():
            return f"""## GEAR IDENTIFICATION RESULTS (Mock Synthesis)

**Original Query:** {original_query}
**Total Gears Found:** 5 (Mock Data)
**Gear List:**
- MOCKGEAR001
- MOCKGEAR002
**Data Quality Assessment:**
- Issues Detected: None (Mock)
- Confidence Level: 1.0 (Mock)
"""
        elif "printer" in original_query.lower() or "part" in original_query.lower():
             return f"""## PRINTER ANALYSIS RESULTS (Mock Synthesis)
**Original Query:** {original_query}
**Assigned Printer:** Printer_X (Mock Data)
**Total Parts on Printer:** 10 (Mock Data)
**Data Quality Assessment:**
- Issues Detected: None (Mock)
- Confidence Level: 1.0 (Mock)
"""
        elif "verify" in original_query.lower() or "order" in original_query.lower():
             return f"""## COMPLIANCE VERIFICATION RESULTS (Mock Synthesis)
**Original Query:** {original_query}
**Date Match Status:** True (Mock Data)
**Data Quality Assessment:**
- Issues Detected: None (Mock)
- Confidence Level: 1.0 (Mock)
"""
        else:
            return f"""## SYNTHESIZED REPORT (Mock Synthesis)
**Answer:** This is a generic mock answer for: {original_query}.
**Confidence:** 0.95
**Data Quality Issues:** None detected.
"""
    
class OpenAIProvider:
    """A real LLM provider that connects to the OpenAI API."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        # The client automatically reads the OPENAI_API_KEY env variable
        self.client = openai.OpenAI()

    @retry(
        stop=stop_after_attempt(3), # From retry_config.max_retries
        wait=wait_exponential(multiplier=1, min=2, max=60) # Exponential backoff
    )

    def generate(self, prompt: str) -> Dict[str, Any]:
        """Makes a real API call to OpenAI and standardizes the response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                # For JSON output, it's better to enable JSON mode
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        except Exception as e:
            print(f"ERROR: OpenAI API call failed: {e}")
            return {"content": f'{{"error": "API call failed: {e}"}}', "input_tokens": 0, "output_tokens": 0}

class AnthropicProvider:
    """A real LLM provider that connects to the Anthropic (Claude) API."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        # The client automatically reads the ANTHROPIC_API_KEY env variable
        self.client = anthropic.Anthropic()

    @retry(
        stop=stop_after_attempt(3),  # From retry_config.max_retries
        wait=wait_exponential(multiplier=1, min=2, max=60),  # Exponential backoff
    )
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Makes a real API call to Anthropic and standardizes the response."""
        try:
            # Prepend a system instruction to force pure JSON output
            system_prompt = (
                "You are a JSON generator. Respond with exactly one valid JSON "
                "object and no additional text or formatting."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,  # Required by Claude API
                messages=messages,
            )
            content = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        except Exception as e:
            print(f"ERROR: Anthropic API call failed: {e}")
            return {
                "content": f'{{"error": "API call failed: {e}"}}',
                "input_tokens": 0,
                "output_tokens": 0,
            }
        
class DeepSeekProvider:
    """A real LLM provider for DeepSeek's OpenAI-compatible API."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Initializes the client with the specific DeepSeek API key and base URL
        self.client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
    
    @retry(
        stop=stop_after_attempt(3), # From retry_config.max_retries
        wait=wait_exponential(multiplier=1, min=2, max=60) # Exponential backoff
    )

    def generate(self, prompt: str) -> Dict[str, Any]:
        """Makes a real API call to DeepSeek and standardizes the response."""
        # DeepSeek requires the word "json" in the prompt when using JSON response format
        adjusted_prompt = prompt
        if "json" not in prompt.lower():
            adjusted_prompt = "Please respond in JSON format.***REMOVED***n***REMOVED***n" + prompt
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": adjusted_prompt}],
                response_format={"type": "json_object"},
            )
            return {
                "content": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        except Exception as e:
            print(f"ERROR: DeepSeek API call failed: {e}")
            return {"content": f'{{"error": "API call failed: {e}"}}', "input_tokens": 0, "output_tokens": 0}