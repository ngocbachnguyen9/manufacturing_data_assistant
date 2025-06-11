# In tests/test_llm_providers.py

import pytest
from unittest.mock import MagicMock
import openai  # Import the libraries we will be patching

# Import all the provider classes to be tested
from src.utils.llm_provider import (
    OpenAIProvider,
    AnthropicProvider,
    DeepSeekProvider,
)


# --- Test for OpenAIProvider ---

def test_openai_provider_parses_success_response(monkeypatch):
    """
    Tests that the OpenAIProvider correctly handles a successful mock API response.
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

    # 4. Run the provider and check its standardized output
    provider = OpenAIProvider(model_name="gpt-4-turbo")
    result = provider.generate("any prompt")

    assert result["content"] == '{"key": "value"}'
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20


def test_openai_provider_handles_api_error(monkeypatch):
    """
    Tests that the OpenAIProvider returns a standardized error dict when the API fails.
    """
    # 1. Configure the mock client to raise an exception when called
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.APIError(
        "Service unavailable", request=None, body=None
    )

    # 2. Patch the constructor
    monkeypatch.setattr("openai.OpenAI", lambda: mock_client)

    # 3. Run the provider and check the error output
    provider = OpenAIProvider(model_name="gpt-4-turbo")
    result = provider.generate("any prompt")

    assert "error" in result["content"]
    assert "API call failed" in result["content"]


# --- Test for AnthropicProvider ---

def test_anthropic_provider_parses_success_response(monkeypatch):
    """
    Tests that the AnthropicProvider correctly handles a successful mock API response.
    """
    mock_api_response = MagicMock()
    mock_api_response.content[0].text = '{"key": "value"}'
    mock_api_response.usage.input_tokens = 30
    mock_api_response.usage.output_tokens = 40

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_api_response

    monkeypatch.setattr("anthropic.Anthropic", lambda: mock_client)

    provider = AnthropicProvider(model_name="claude-3-5-sonnet")
    result = provider.generate("any prompt")

    assert result["content"] == '{"key": "value"}'
    assert result["input_tokens"] == 30
    assert result["output_tokens"] == 40


# --- Test for DeepSeekProvider ---

def test_deepseek_provider_parses_success_response(monkeypatch):
    """
    Tests that the DeepSeekProvider correctly handles a successful mock API response.
    (This is identical to the OpenAI test because the API is compatible).
    """
    mock_api_response = MagicMock()
    mock_api_response.choices[0].message.content = '{"key": "value"}'
    mock_api_response.usage.prompt_tokens = 50
    mock_api_response.usage.completion_tokens = 60

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_api_response

    # Patch the OpenAI constructor, which DeepSeekProvider uses
    monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: mock_client)

    provider = DeepSeekProvider(model_name="deepseek-r1")
    result = provider.generate("any prompt")

    assert result["content"] == '{"key": "value"}'
    assert result["input_tokens"] == 50
    assert result["output_tokens"] == 60