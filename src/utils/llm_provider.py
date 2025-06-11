import json
import random
from typing import Dict, Any
import openai      # NEW: Import real libraries
import anthropic   # NEW
from tenacity import retry, stop_after_attempt, wait_exponential
import os


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
        """
        # Simulate token usage
        input_tokens = len(prompt.split())
        output_tokens = random.randint(50, 250)

        # Simulate response content based on prompt type
        if "create a JSON plan" in prompt:
            content = self._get_mock_plan(prompt)
        else:
            content = self._get_mock_synthesis(prompt)

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _get_mock_plan(self, prompt: str) -> str:
        """Returns a mock plan based on keywords in the prompt."""
        if "ORBOX0014" in prompt:  # Easy task
            return """
            {
                "complexity": "easy",
                "plan": [{"step": 1, "tool": "relationship_tool", "input": "ORBOX0014", "output_key": "gears"}]
            }
            """
        # Default to a more complex plan
        return """
        {
            "complexity": "medium",
            "plan": [
                {"step": 1, "tool": "relationship_tool", "input": "3DOR10001", "output_key": "printer_rel"},
                {"step": 2, "tool": "machine_log_tool", "input": "{printer_rel[0][parent]}", "output_key": "printer_logs"}
            ]
        }
        """

    def _get_mock_synthesis(self, prompt: str) -> str:
        """Returns a generic synthesized report."""
        return """ ## SYNTHESIZED REPORT
        **Answer:** Based on the data, the primary entity is associated with Printer_1.
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
        stop=stop_after_attempt(3), # From retry_config.max_retries
        wait=wait_exponential(multiplier=1, min=2, max=60) # Exponential backoff
    )

    def generate(self, prompt: str) -> Dict[str, Any]:
        """Makes a real API call to Anthropic and standardizes the response."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096, # Required by Claude API
                messages=[{"role": "user", "content": prompt}],
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
            return {"content": f'{{"error": "API call failed: {e}"}}', "input_tokens": 0, "output_tokens": 0}

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
        # This logic is identical to the OpenAIProvider because the API is compatible
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
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