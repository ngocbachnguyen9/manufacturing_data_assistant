from src.data_generation.manufacturing_environment import ManufacturingEnvironment
from src.data_generation.data_quality_controller import DataQualityController
from src.data_generation.ground_truth_generator import GroundTruthGenerator
from experiments.validation.error_injection_validation import InjectionValidator
from experiments.validation.ground_truth_validation import GroundTruthValidator


def main():
    """
    Executes the full data generation and validation pipeline for Phase 1.
    """
    # --- Step 1: Setup Q0 Baseline Environment ---
    print("--- Starting Phase 1: Data Corpus Generation ---")
    env = ManufacturingEnvironment()
    env.setup_baseline_environment()

    # --- Step 2: Generate Corrupted Datasets ---
    controller = DataQualityController()
    quality_conditions = ["Q1", "Q2", "Q3"]
    for qc in quality_conditions:
        corrupted_data, error_tracker = controller.apply_corruption(qc)
        controller.save_corrupted_data(corrupted_data, error_tracker, qc)

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