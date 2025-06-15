#!/usr/bin/env python3
"""
Regenerate experimental datasets with seeded random corruption for reproducibility.

This script ensures that the data corruption patterns are deterministic and can be
reproduced exactly for debugging purposes. It uses the random seed from the 
experiment configuration to generate consistent corruption patterns.
"""

import os
import sys
import yaml
import shutil
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation.data_quality_controller import DataQualityController


def load_experiment_config():
    """Load the experiment configuration to get the random seed."""
    config_path = project_root / "config" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def backup_existing_datasets():
    """Create a backup of existing datasets before regeneration."""
    datasets_dir = project_root / "data" / "experimental_datasets"
    backup_dir = project_root / "data" / "experimental_datasets_backup"
    
    if datasets_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(datasets_dir, backup_dir)
        print(f"✅ Backed up existing datasets to: {backup_dir}")
    else:
        print("⚠️  No existing datasets found to backup")


def regenerate_dataset(quality_condition: str, controller: DataQualityController):
    """Regenerate a single dataset with seeded corruption."""
    print(f"***REMOVED***n🔄 Regenerating {quality_condition} dataset...")
    
    try:
        # Apply corruption with seeded randomness
        corrupted_data, error_tracker, targeted_ids = controller.apply_corruption(quality_condition)
        
        # Save the corrupted data and error logs
        controller.save_corrupted_data(corrupted_data, error_tracker, quality_condition)
        
        print(f"✅ Successfully regenerated {quality_condition} dataset")
        print(f"   - Corrupted entities: {len(targeted_ids)}")
        print(f"   - Error log entries: {len(error_tracker.get_log_as_df(quality_condition))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to regenerate {quality_condition} dataset: {str(e)}")
        return False


def verify_seed_consistency():
    """Verify that the same seed produces the same results."""
    print("***REMOVED***n🔍 Verifying seed consistency...")
    
    config = load_experiment_config()
    seed = config["experiment"]["random_seed"]
    
    # Create two controllers with the same seed
    controller1 = DataQualityController(random_seed=seed)
    controller2 = DataQualityController(random_seed=seed)
    
    # Generate Q1 corruption with both controllers
    _, tracker1, ids1 = controller1.apply_corruption("Q1")
    _, tracker2, ids2 = controller2.apply_corruption("Q1")
    
    # Compare results
    log1 = tracker1.get_log_as_df("Q1")
    log2 = tracker2.get_log_as_df("Q1")
    
    if len(log1) == len(log2) and set(ids1) == set(ids2):
        print("✅ Seed consistency verified - same seed produces identical results")
        return True
    else:
        print("❌ Seed consistency failed - same seed produced different results")
        return False


def main():
    """Main function to regenerate all datasets with seeded corruption."""
    print("🌱 Regenerating Experimental Datasets with Seeded Corruption")
    print("=" * 60)
    
    # Load configuration
    config = load_experiment_config()
    seed = config["experiment"]["random_seed"]
    print(f"📋 Using random seed: {seed}")
    
    # Verify seed consistency first
    if not verify_seed_consistency():
        print("❌ Seed consistency check failed. Aborting regeneration.")
        return False
    
    # Backup existing datasets
    backup_existing_datasets()
    
    # Create controller with seeded randomness
    controller = DataQualityController(random_seed=seed)
    
    # Regenerate each quality condition
    quality_conditions = ["Q1", "Q2", "Q3"]
    success_count = 0
    
    for qc in quality_conditions:
        if regenerate_dataset(qc, controller):
            success_count += 1
    
    # Summary
    print(f"***REMOVED***n📊 Regeneration Summary:")
    print(f"   - Successful: {success_count}/{len(quality_conditions)} datasets")
    print(f"   - Random seed: {seed}")
    print(f"   - Datasets are now reproducible for debugging")
    
    if success_count == len(quality_conditions):
        print("✅ All datasets successfully regenerated with seeded corruption!")
        return True
    else:
        print("⚠️  Some datasets failed to regenerate. Check error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
