# In tests/test_llm_providers.py

import pytest
from unittest.mock import MagicMock
from src.utils.llm_provider import OpenAIProvider

def test_openai_provider_parses_response_correctly(monkeypatch):
    """
    Tests that the OpenAIProvider correctly handles a mock API response.
    """
    # 1. Create a fake API response object that mimics the real one
    mock_api_response = MagicMock()
    mock_api_response.choices[0].message.content = '{"key": "value"}'
    mock_api_response.usage.prompt_tokens = 10
    mock_api_response.usage.completion_tokens = 20

    # 2. Create a fake client that returns our fake response
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_api_response

    # 3. Patch the OpenAI() constructor to return our fake client
    monkeypatch.setattr("openai.OpenAI", lambda: mock_client)

    # 4. Run the provider and check its output
    provider = OpenAIProvider(model_name="gpt-4-turbo")
    result = provider.generate("any prompt")

    # Assert that the provider correctly standardized the mock response
    assert result["content"] == '{"key": "value"}'
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20