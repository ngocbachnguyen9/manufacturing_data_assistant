import yaml
import os
from typing import Dict, Any

from src.data_generation.data_quality_controller import DataQualityController
from src.experiment.task_generator import TaskGenerator
from src.data_generation.manufacturing_environment import ManufacturingEnvironment


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the main experiment configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """
    Orchestrates the entire tightly coupled setup for the experiment.
    1. Generates Q0 baseline data.
    2. Generates corrupted datasets (Q1, Q2, Q3) and captures the IDs of
       the entities that were corrupted.
    3. Feeds these "dirty" IDs into the TaskGenerator to ensure all
       experimental tasks are valid and test the intended data conditions.
    """
    print("--- Running Full Experiment Setup ---")
    config = load_config("config/experiment_config.yaml")

    # Step 1: Setup Q0 Baseline Environment
    print("***REMOVED***n--- Phase 1a: Setting up Q0 Baseline Environment ---")
    env = ManufacturingEnvironment()
    env.setup_baseline_environment()

    # Step 2: Generate Corrupted Datasets and Capture Dirty IDs
    print("***REMOVED***n--- Phase 1b: Generating Corrupted Datasets ---")
    seed = config["experiment"]["random_seed"]
    print(f"🌱 Using random seed {seed} for reproducible data generation")
    controller = DataQualityController(random_seed=seed)
    dirty_entity_ids = {}

    for qc in ["Q1", "Q2", "Q3"]:
        corrupted_data, error_tracker, targeted_ids = controller.apply_corruption(qc)
        controller.save_corrupted_data(corrupted_data, error_tracker, qc)
        dirty_entity_ids[qc] = targeted_ids
        print(f"  - Captured {len(targeted_ids)} dirty IDs for {qc}")

    # Step 3: Generate Counterbalanced Tasks Using Dirty ID Lists
    print("***REMOVED***n--- Phase 3a: Generating Validated Task Assignments ---")
    task_generator = TaskGenerator(config, dirty_entity_ids)
    assignments = task_generator.generate_all_assignments()
    task_generator.save_assignments(assignments)

    print("***REMOVED***n***REMOVED***n--- Experiment Setup Complete ---")
    print("All datasets and participant assignments have been generated correctly.")
    print("You can now run the human study (Phase 3b) or LLM evaluation (Phase 4).")


if __name__ == "__main__":
    main()