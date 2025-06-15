# In run_phase4_llm_evaluation.py

import yaml
import os
from dotenv import load_dotenv
from argparse import ArgumentParser
from src.experiment.llm_evaluation import LLMEvaluationRunner
import re

# Load API keys from .env file into the environment
load_dotenv()

def load_specific_configs(config_dir: str) -> dict:
    """Loads and merges a specific, known set of configuration files."""
    # ... (this function is correct and does not need changes) ...
    combined_config = {}
    # Include agent settings, experiment, llm, task prompts, and master-agent planning prompt
    files_to_load = [
        "agent_config.yaml",
        "experiment_config.yaml",
        "llm_config.yaml",
        "task_prompts.yaml",
        "master_agent_planning_prompt.yaml",
    ]
    for filename in files_to_load:
        # first try config/<filename>, then config/prompts/<filename>
        path = os.path.join(config_dir, filename)
        if not os.path.exists(path):
            alt = os.path.join(config_dir, "prompts", filename)
            if os.path.exists(alt):
                path = alt
            else:
                print(f"Warning: Config file not found, skipping: {filename}")
                continue
        with open(path, "r") as f:
            config_data = yaml.safe_load(f)
            if config_data:
                combined_config.update(config_data)
    return combined_config


def main():
    """Orchestrates the entire process for Phase 4: LLM Evaluation."""
    # --- UPDATED: Add a new argument for model selection ---
    parser = ArgumentParser(description="Run Phase 4 LLM Evaluation.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run with live API keys. Default is to use the mock provider.",
    )
    parser.add_argument(
        "--models",
        nargs="+",  # This allows one or more arguments
        help="Specify one or more model names to test (e.g., gpt-4o claude-3-5-sonnet). Overrides config.",
    )
    args = parser.parse_args()
    # --- END UPDATE ---

    print("--- Starting Phase 4: LLM Evaluation ---")
    USE_MOCK_LLM = not args.live

    if not USE_MOCK_LLM:
        print("***REMOVED***n" + "="*50)
        print("!! RUNNING IN LIVE MODE. REAL API CALLS WILL BE MADE. !!")
        print("="*50 + "***REMOVED***n")

    try:
        config = load_specific_configs("config")

        # --- NEW: Override models_to_test if provided via command line ---
        if args.models:
            print(f"Overriding config. Running evaluation for specified models: {args.models}")
            config["llm_evaluation"]["models_to_test"] = args.models

        # Use corrected ground truth by default to fix the evaluation issues
        runner = LLMEvaluationRunner(config, use_mock=USE_MOCK_LLM, use_corrected_ground_truth=True)
        runner.run_evaluation()
    except Exception as e:
        print(f"***REMOVED***nAn unexpected error occurred in the evaluation runner: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()