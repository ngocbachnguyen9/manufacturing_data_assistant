# In tests/test_phase1_runner.py

import pytest
import os
import sys

# Ensure the project root is on the path to allow importing the runner script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the main function from your runner script
from run_phase1_generation import main as phase1_main

# Import the classes that the runner uses, so we can patch them
from src.data_generation.manufacturing_environment import ManufacturingEnvironment
from src.data_generation.data_quality_controller import DataQualityController
from src.data_generation.ground_truth_generator import GroundTruthGenerator
from experiments.validation.error_injection_validation import InjectionValidator
from experiments.validation.ground_truth_validation import GroundTruthValidator


@pytest.fixture
def patch_all_phase1_dependencies(monkeypatch):
    """
    A comprehensive fixture to patch all external dependencies and heavy methods
    called by the run_phase1_generation.py script.
    """
    # Create "spy" lists to record method calls
    calls = {
        "setup_baseline": [],
        "apply_corruption": [],
        "save_corrupted": [],
        "generate_gt": [],
        "validate_injection": [],
        "validate_gt": [],
    }

    # --- THIS IS THE FIX ---
    # Define the helper function INSIDE the fixture, before it is used.
    def mock_apply_corruption(self, qc):
        # Action 1: Record the call to the spy list
        calls["apply_corruption"].append(qc)
        # Action 2: Return the required 3-tuple to prevent errors
        return ({}, "dummy_tracker_object", [])
    # --- END FIX ---

    # Patch ManufacturingEnvironment
    monkeypatch.setattr(
        ManufacturingEnvironment,
        "setup_baseline_environment",
        lambda self: calls["setup_baseline"].append(True),
    )

    # Patch DataQualityController methods
    # Now, this call can see the function defined above it.
    monkeypatch.setattr(
        DataQualityController, "apply_corruption", mock_apply_corruption
    )
    monkeypatch.setattr(
        DataQualityController,
        "save_corrupted_data",
        lambda self, d, t, qc: calls["save_corrupted"].append(qc),
    )

    # Patch GroundTruthGenerator
    monkeypatch.setattr(
        GroundTruthGenerator,
        "generate_all_ground_truths",
        lambda self: calls["generate_gt"].append(True),
    )

    # Patch Validators
    monkeypatch.setattr(
        InjectionValidator,
        "validate_all_conditions",
        lambda self: calls["validate_injection"].append(True),
    )
    monkeypatch.setattr(
        GroundTruthValidator,
        "run_all_validations",
        lambda self: calls["validate_gt"].append(True),
    )

    # Return the spy dictionary to the test function
    return calls


def test_phase1_runner_orchestration(patch_all_phase1_dependencies):
    """
    This is a smoke test to ensure the Phase 1 runner script correctly
    orchestrates the calls to all its component classes.
    """
    # The fixture patches all the heavy lifting. Now, just run the main function.
    phase1_main()

    # Retrieve the call log from the fixture
    calls = patch_all_phase1_dependencies

    # Assert that each major step was called the correct number of times
    assert len(calls["setup_baseline"]) == 1
    assert len(calls["apply_corruption"]) == 3  # Called for Q1, Q2, Q3
    assert len(calls["save_corrupted"]) == 3  # Called for Q1, Q2, Q3
    assert len(calls["generate_gt"]) == 1
    assert len(calls["validate_injection"]) == 1
    assert len(calls["validate_gt"]) == 1

    # You can even check the arguments if needed
    assert sorted(calls["apply_corruption"]) == ["Q1", "Q2", "Q3"]