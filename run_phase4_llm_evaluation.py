# In run_phase4_llm_evaluation.py

import yaml
import os
from dotenv import load_dotenv
from argparse import ArgumentParser # NEW: For command-line arguments
from src.experiment.llm_evaluation import LLMEvaluationRunner

# Load API keys from .env file into the environment
load_dotenv()

def load_specific_configs(config_dir: str) -> dict:
    """Loads and merges a specific, known set of configuration files."""
    combined_config = {}
    # Define the exact files to load, in order of precedence
    files_to_load = [
        "experiment_config.yaml",
        "llm_config.yaml", # Assuming you have this
        "task_prompts.yaml",
    ]
    for filename in files_to_load:
        path = os.path.join(config_dir, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    combined_config.update(config_data)
        else:
            print(f"Warning: Config file not found, skipping: {path}")
    return combined_config

def main():
    """Orchestrates the entire process for Phase 4: LLM Evaluation."""
    # NEW: Use argparse to handle the --live flag
    parser = ArgumentParser(description="Run Phase 4 LLM Evaluation.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run with live API keys. Default is to use the mock provider.",
    )
    args = parser.parse_args()

    print("--- Starting Phase 4: LLM Evaluation ---")
    # The USE_MOCK_LLM flag is now controlled from the command line
    USE_MOCK_LLM = not args.live

    if not USE_MOCK_LLM:
        print("***REMOVED***n" + "="*50)
        print("!! RUNNING IN LIVE MODE. REAL API CALLS WILL BE MADE. !!")
        print("="*50 + "***REMOVED***n")

    try:
        config = load_specific_configs("config")
        runner = LLMEvaluationRunner(config, use_mock=USE_MOCK_LLM)
        runner.run_evaluation()
    except Exception as e:
        print(f"***REMOVED***nAn unexpected error occurred in the evaluation runner: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()