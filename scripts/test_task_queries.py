#!/usr/bin/env python3
"""
Test Task Queries Script

This script verifies that the evaluation system is loading the correct queries
for each task and that the corrected ground truth is working properly.

Usage:
    python scripts/test_task_queries.py
"""

import json
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_task_query_loading():
    """Test that tasks are loading the correct queries."""
    print("🔍 TESTING TASK QUERY LOADING")
    print("=" * 50)
    
    # Load participant assignments
    with open("experiments/human_study/participant_assignments.json", 'r') as f:
        assignments = json.load(f)
    
    # Test specific tasks that were showing issues
    test_tasks = ["P1_task_1", "P1_task_2", "P1_task_7"]
    
    for participant_id, tasks in assignments.items():
        for task in tasks:
            if task["task_id"] in test_tasks:
                print(f"***REMOVED***n📋 {task['task_id']}:")
                print(f"   Participant: {task['participant_id']}")
                print(f"   Complexity: {task['complexity']}")
                print(f"   Quality: {task['quality_condition']}")
                print(f"   Query: {task['query_string']}")
                print(f"   Dataset: {task['dataset_path']}")


def test_ground_truth_loading():
    """Test that ground truth is loading correctly."""
    print("***REMOVED***n🔍 TESTING GROUND TRUTH LOADING")
    print("=" * 50)
    
    # Check if corrected ground truth exists
    corrected_path = "data/ground_truth/baseline_answers_corrected.json"
    original_path = "data/ground_truth/baseline_answers.json"
    
    if Path(corrected_path).exists():
        print(f"✅ Corrected ground truth exists: {corrected_path}")
        with open(corrected_path, 'r') as f:
            corrected_gt = json.load(f)
        print(f"   Contains {len(corrected_gt)} tasks")
    else:
        print(f"❌ Corrected ground truth missing: {corrected_path}")
    
    if Path(original_path).exists():
        print(f"✅ Original ground truth exists: {original_path}")
        with open(original_path, 'r') as f:
            original_gt = json.load(f)
        print(f"   Contains {len(original_gt)} tasks")
    else:
        print(f"❌ Original ground truth missing: {original_path}")
    
    # Compare specific tasks
    if Path(corrected_path).exists() and Path(original_path).exists():
        print(f"***REMOVED***n📊 Comparing ground truth for key tasks:")
        
        # Create lookup dictionaries
        corrected_lookup = {task["task_id"]: task for task in corrected_gt}
        original_lookup = {task["task_id"]: task for task in original_gt}
        
        test_tasks = ["P1_task_1", "P2_task_6"]  # P1_task_1 had gear count issues, P2_task_6 was fixed
        
        for task_id in test_tasks:
            if task_id in corrected_lookup and task_id in original_lookup:
                corrected_answer = corrected_lookup[task_id]["baseline_answer"]
                original_answer = original_lookup[task_id]["baseline_answer"]
                
                print(f"***REMOVED***n   {task_id}:")
                if corrected_answer != original_answer:
                    print(f"     ⚠️  DIFFERENT:")
                    if "gear_count" in corrected_answer and "gear_count" in original_answer:
                        print(f"       Original: {original_answer['gear_count']} gears")
                        print(f"       Corrected: {corrected_answer['gear_count']} gears")
                    elif "packing_list_id" in corrected_answer and "packing_list_id" in original_answer:
                        print(f"       Original: {original_answer['packing_list_id']}")
                        print(f"       Corrected: {corrected_answer['packing_list_id']}")
                else:
                    print(f"     ✅ IDENTICAL")


def test_evaluation_initialization():
    """Test that the evaluation system initializes correctly."""
    print("***REMOVED***n🔍 TESTING EVALUATION INITIALIZATION")
    print("=" * 50)
    
    try:
        from src.experiment.llm_evaluation import LLMEvaluationRunner
        
        # Test with corrected ground truth
        config = {"test": True}
        runner = LLMEvaluationRunner(config, use_mock=True, use_corrected_ground_truth=True)
        
        print(f"✅ LLMEvaluationRunner initialized successfully")
        print(f"   Assignments loaded: {len(runner.assignments)} participants")
        print(f"   Ground truth loaded: {len(runner.ground_truth)} tasks")
        
        # Test specific task lookup
        test_task_id = "P1_task_1"
        gt_task = next((task for task in runner.ground_truth if task["task_id"] == test_task_id), None)
        
        if gt_task:
            print(f"***REMOVED***n📋 Sample ground truth for {test_task_id}:")
            print(f"   Query: {gt_task['query_instance']}")
            print(f"   Complexity: {gt_task['complexity_level']}")
            if "gear_count" in gt_task["baseline_answer"]:
                print(f"   Expected gears: {gt_task['baseline_answer']['gear_count']}")
        else:
            print(f"❌ Could not find ground truth for {test_task_id}")
            
    except Exception as e:
        print(f"❌ Error initializing evaluation system: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("🧪 TESTING TASK QUERY AND GROUND TRUTH SYSTEM")
    print("=" * 60)
    
    test_task_query_loading()
    test_ground_truth_loading()
    test_evaluation_initialization()
    
    print("***REMOVED***n" + "=" * 60)
    print("🎯 TESTING COMPLETE")
    print("***REMOVED***nIf all tests pass, the evaluation system should work correctly.")
    print("You can now run: python run_phase4_llm_evaluation.py --models deepseek-chat")


if __name__ == "__main__":
    main()
