import yaml
import os
from typing import Dict, Any

# Import the necessary classes from the src directory
from src.experiment.task_generator import TaskGenerator
from src.human_study.study_platform import StudyPlatform


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the main experiment configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Orchestrates the entire process for Phase 3: Human Study Execution.
    1. Loads the experiment configuration.
    2. Generates the counterbalanced task assignments for all participants.
    3. Launches the interactive study platform for data collection.
    """
    print("--- Starting Phase 3: Human Study Execution ---")
    config_path = "config/experiment_config.yaml"
    assignments_path = "experiments/human_study/participant_assignments.json"

    try:
        # --- Step 1: Load Configuration ---
        print(f"Loading configuration from {config_path}...")
        config = load_config(config_path)

        # --- Step 2: Generate Task Assignments ---
        print("***REMOVED***n--- Generating Participant Task Assignments ---")
        task_generator = TaskGenerator(config)
        assignments = task_generator.generate_all_assignments()
        task_generator.save_assignments(assignments)

        # --- Step 3: Launch Study Platform ---
        print("***REMOVED***n--- Launching Human Study Platform ---")
        print(
            "The platform will now start. Please enter a participant ID to begin a session."
        )
        input("Press Enter to continue...")

        study_platform = StudyPlatform(assignments_path)
        study_platform.run_session()

    except FileNotFoundError as e:
        print(f"***REMOVED***nERROR: A required file was not found. {e}")
        print("Please ensure your configuration and data paths are correct.")
    except Exception as e:
        print(f"***REMOVED***nAn unexpected error occurred: {e}")

    print("***REMOVED***n--- Phase 3 Script Finished ---")


if __name__ == "__main__":
    main()