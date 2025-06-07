from src.data_generation.manufacturing_environment import ManufacturingEnvironment
from src.data_generation.data_quality_controller import DataQualityController
from src.data_generation.ground_truth_generator import GroundTruthGenerator
from experiments.validation.error_injection_validation import InjectionValidator


def main():
    """
    Executes the full data generation pipeline for Phase 1.
    1. Sets up the Q0 baseline environment and documents.
    2. Generates corrupted datasets (Q1, Q2, Q3) with error logs.
    3. Generates ground truth answers from the Q0 baseline.
    4. Validates the integrity of the generated corrupted datasets.
    """
    # --- Step 1: Setup Q0 Baseline Environment ---
    env = ManufacturingEnvironment()
    env.setup_baseline_environment()

    # --- Step 2: Generate Corrupted Datasets ---
    controller = DataQualityController()
    quality_conditions = ["Q1", "Q2", "Q3"]

    for qc in quality_conditions:
        corrupted_data, error_tracker = controller.apply_corruption(qc)
        controller.save_corrupted_data(corrupted_data, error_tracker, qc)

    # --- Step 3: Generate Ground Truth ---
    print("***REMOVED***n--- Generating Ground Truth Answers ---")
    gt_generator = GroundTruthGenerator()
    gt_generator.generate_all_ground_truths()

    # --- Step 4: Validate Error Injection ---
    validator = InjectionValidator()
    validator.validate_all_conditions()

    print("***REMOVED***n***REMOVED***nPhase 1 Data Generation and Validation Complete.")
    print("Datasets generated in 'data/experimental_datasets/'")
    print("Documents generated in 'data/generated_documents/'")
    print("Ground truth generated in 'data/ground_truth/'")


if __name__ == "__main__":
    main()