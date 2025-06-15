from src.data_generation.manufacturing_environment import ManufacturingEnvironment
from src.data_generation.data_quality_controller import DataQualityController
from src.data_generation.ground_truth_generator import GroundTruthGenerator
from experiments.validation.error_injection_validation import InjectionValidator
from experiments.validation.ground_truth_validation import GroundTruthValidator
from typing import Dict, List, Any
import os
import json
import yaml

def load_experiment_config():
    """Load the experiment configuration to get the random seed."""
    with open("config/experiment_config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    """
    Executes the full data generation and validation pipeline for Phase 1.
    """
    # Load configuration for seeded generation
    config = load_experiment_config()
    seed = config["experiment"]["random_seed"]
    print(f"🌱 Using random seed {seed} for reproducible data generation")

    # --- Step 1: Setup Q0 Baseline Environment ---
    print("--- Starting Phase 1: Data Corpus Generation ---")
    env = ManufacturingEnvironment()
    env.setup_baseline_environment()

    # Create controller with seeded randomness for reproducible corruption
    controller = DataQualityController(random_seed=seed)
    quality_conditions = ["Q1", "Q2", "Q3"]
    dirty_ids: Dict[str, List[str]] = {}
    # Apply corruption and capture the list of targeted IDs
    for qc in quality_conditions:
        corrupted_data, error_tracker, targeted = controller.apply_corruption(qc)
        controller.save_corrupted_data(corrupted_data, error_tracker, qc)
        dirty_ids[qc] = targeted
        print(f"  - Captured {len(targeted)} dirty IDs for {qc}")

    # Persist dirty‐ID map for Phase 3
    os.makedirs("experiments/human_study", exist_ok=True)
    with open("experiments/human_study/dirty_ids.json", "w") as f:
        json.dump(dirty_ids, f, indent=2)
    print("Saved dirty IDs map → experiments/human_study/dirty_ids.json")

    # --- Step 3: Generate Ground Truth & Traversal Paths ---
    print("***REMOVED***n--- Generating Ground Truth Answers & Paths ---")
    gt_generator = GroundTruthGenerator()
    gt_generator.generate_all_ground_truths()

    # --- Step 4: Validate Error Injection ---
    print("***REMOVED***n--- Validating Corrupted Datasets ---")
    injection_validator = InjectionValidator()
    injection_validator.validate_all_conditions()

    # --- Step 5 (NEW): Validate the Ground Truth Itself ---
    print("***REMOVED***n--- Validating Ground Truth Integrity ---")
    gt_validator = GroundTruthValidator()
    gt_validator.run_all_validations()

    print("***REMOVED***n***REMOVED***n--- Phase 1: Data Generation and Validation Complete ---")


if __name__ == "__main__":
    main()