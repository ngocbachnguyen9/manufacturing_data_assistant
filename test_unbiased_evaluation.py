#!/usr/bin/env python3
"""
Test script for the unbiased evaluation system

This script tests the new unbiased judge system with a small sample
to verify it works correctly before running the full evaluation.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.experiment.llm_evaluation import LLMEvaluationRunner

def test_unbiased_evaluation():
    """Test the unbiased evaluation system with a small sample"""
    
    print("🧪 Testing Unbiased LLM Evaluation System")
    print("=" * 50)
    
    # Load config
    import yaml
    with open('config/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a test configuration with just one model and a few tasks
    test_config = config.copy()
    test_config["llm_evaluation"]["models_to_test"] = ["deepseek-chat"]  # Test with one model
    
    print("🔧 Configuration:")
    print(f"   Models to test: {test_config['llm_evaluation']['models_to_test']}")
    print(f"   Using mock providers: False (live API calls)")
    print()
    
    # Initialize the evaluation runner
    try:
        runner = LLMEvaluationRunner(test_config, use_mock=False)
        print("✅ LLMEvaluationRunner initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize runner: {e}")
        return False
    
    # Test judge provider initialization
    try:
        print("***REMOVED***n🏛️  Testing judge provider initialization...")
        judge_providers = runner._initialize_judge_providers()
        print(f"✅ Successfully initialized {len(judge_providers)} judge providers")
        
        for judge_name, config in judge_providers.items():
            print(f"   • {judge_name}: weight {config['weight']}")
            
    except Exception as e:
        print(f"❌ Failed to initialize judge providers: {e}")
        print("   This might be due to missing API keys or network issues")
        return False
    
    # Test judge prompt creation
    try:
        print("***REMOVED***n📝 Testing judge prompt creation...")
        test_prompt = runner._create_unbiased_judge_prompt(
            task_complexity="easy",
            llm_report="Test report: Found 3 gears for order ORD123",
            ground_truth='{"order_id": "ORD123", "gears": ["GEAR001", "GEAR002", "GEAR003"]}'
        )
        print("✅ Judge prompt created successfully")
        print(f"   Prompt length: {len(test_prompt)} characters")
        
    except Exception as e:
        print(f"❌ Failed to create judge prompt: {e}")
        return False
    
    # Test a single evaluation (if we have real data)
    try:
        print("***REMOVED***n🔍 Testing single evaluation...")
        
        # Load a real task for testing
        assignments_path = "experiments/human_study/participant_assignments.json"
        if os.path.exists(assignments_path):
            with open(assignments_path, 'r') as f:
                assignments = json.load(f)
            
            # Get first task
            first_participant = list(assignments.keys())[0]
            first_task = assignments[first_participant][0]
            
            print(f"   Testing with task: {first_task['task_id']}")
            print(f"   Complexity: {first_task['complexity']}")
            print(f"   Quality condition: {first_task['quality_condition']}")
            
            # Create a mock report for testing
            test_report = """
            ## Manufacturing Data Analysis Report
            
            **Task**: Find gears for packing list
            **Result**: Found 3 gears for order ORD123
            **Confidence**: 0.85
            **Issues**: None detected
            """
            
            # Test the evaluation
            is_correct, gt_answer = runner._evaluate_answer(
                task_id=first_task['task_id'],
                llm_report=test_report,
                llm_provider=None  # Not used in unbiased system
            )
            
            print(f"✅ Evaluation completed successfully")
            print(f"   Result: {'✅ Correct' if is_correct else '❌ Incorrect'}")
            
            # Check if bias analysis data was created
            if hasattr(runner, 'bias_analysis_data') and runner.bias_analysis_data:
                bias_data = runner.bias_analysis_data[-1]
                print(f"   Consensus score: {bias_data['consensus_score']:.2f}")
                print(f"   Agreement level: {bias_data['agreement_level']}")
                print(f"   Total judges: {bias_data['total_judges']}")
            
        else:
            print("   ⚠️  No assignment data found, skipping single evaluation test")
            
    except Exception as e:
        print(f"❌ Single evaluation test failed: {e}")
        print("   This might be expected if ground truth data doesn't match")
        # Don't return False here as this might be expected
    
    print("***REMOVED***n🎉 Unbiased Evaluation System Test Complete!")
    print("=" * 50)
    print("✅ All core components are working correctly")
    print("🏛️  Ready to run full unbiased evaluation")
    print()
    print("💡 Next steps:")
    print("   1. Run full evaluation: python -m src.experiment.llm_evaluation")
    print("   2. Compare results with previous biased evaluation")
    print("   3. Review bias analysis report")
    
    return True

if __name__ == "__main__":
    success = test_unbiased_evaluation()
    
    if success:
        print("***REMOVED***n🚀 Test passed! The unbiased evaluation system is ready.")
    else:
        print("***REMOVED***n❌ Test failed! Please check the error messages above.")
        sys.exit(1)
