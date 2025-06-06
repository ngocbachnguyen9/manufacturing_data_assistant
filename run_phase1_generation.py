from src.data_generation.manufacturing_environment import ManufacturingEnvironment
from src.data_generation.data_quality_controller import DataQualityController


def main():
    """
    Executes the full data generation pipeline for Phase 1.
    1. Sets up the Q0 baseline environment and documents.
    2. Generates corrupted datasets (Q1, Q2, Q3) with error logs.
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

    print("***REMOVED***n***REMOVED***nPhase 1 Data Generation Complete.")
    print("Datasets generated in 'data/experimental_datasets/'")
    print("Documents generated in 'data/generated_documents/'")


if __name__ == "__main__":
    main()