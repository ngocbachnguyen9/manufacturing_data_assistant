# In run_phase4_llm_evaluation.py
import yaml
import os
from dotenv import load_dotenv
from src.experiment.llm_evaluation import LLMEvaluationRunner
from src.utils.llm_provider import (
    MockLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    DeepSeekProvider,
)

# Load API keys from .env file into the environment
load_dotenv()


def load_all_configs(config_dir: str) -> dict:
    """Loads all YAML files from the config directory into one dictionary."""
    combined_config = {}
    for filename in os.listdir(config_dir):
        if filename.endswith(".yaml"):
            with open(os.path.join(config_dir, filename), "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:  # Ensure file is not empty
                    combined_config.update(config_data)
    return combined_config


def select_llm_provider(model_name: str, use_mock: bool):
    """Factory function to select the correct LLM provider."""
    if use_mock:
        print(f"  - Using MockLLMProvider for {model_name}")
        return MockLLMProvider(model_name)

    print(f"  - Using LIVE provider for {model_name}")
    if "o4" in model_name or "o3" in model_name:
        return OpenAIProvider(model_name)
    elif "claude" in model_name:
        return AnthropicProvider(model_name)
    elif "deepseek" in model_name:
        return DeepSeekProvider(model_name)
    else:
        raise ValueError(f"No real provider found for model: {model_name}")


def main():
    """Orchestrates the entire process for Phase 4: LLM Evaluation."""
    print("--- Starting Phase 4: LLM Evaluation ---")
    config_dir = "config"

    # --- THIS IS THE MASTER SWITCH ---
    # Set to False to use your real API keys and spend money.
    # Set to True to use the mock provider for fast, free testing.
    USE_MOCK_LLM = True

    try:
        config = load_all_configs(config_dir)
        # Pass the factory function to the runner
        runner = LLMEvaluationRunner(config, select_llm_provider, use_mock=USE_MOCK_LLM)
        runner.run_evaluation()
    except Exception as e:
        print(f"***REMOVED***nAn unexpected error occurred in the evaluation runner: {e}")
        # For debugging, you might want to see the full traceback
        # import traceback
        # traceback.print_exc()


if __name__ == "__main__":
    main()