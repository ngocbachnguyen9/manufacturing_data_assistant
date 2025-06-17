#!/usr/bin/env python3
"""
Test script to verify benchmarking system compatibility with new unbiased evaluation results
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from benchmarking_suite import ComprehensiveBenchmarkingSuite
from evaluation_framework import LLMEvaluationFramework
from model_comparison import ModelComparison
from prompt_effectiveness_analyzer import PromptEffectivenessAnalyzer

def test_csv_loading(csv_path: str):
    """Test if the evaluation framework can load the new CSV format"""
    print(f"🧪 Testing CSV loading: {csv_path}")
    
    try:
        framework = LLMEvaluationFramework(csv_path)
        print(f"✅ Successfully loaded {len(framework.df)} records")
        
        # Check for new columns
        expected_cols = ['task_id', 'model', 'complexity', 'quality_condition', 
                        'completion_time_sec', 'is_correct', 'total_cost_usd', 
                        'input_tokens', 'output_tokens', 'final_confidence']
        
        new_cols = ['judge_consensus_score', 'judge_details', 'total_judges', 'agreement_level']
        
        missing_cols = [col for col in expected_cols if col not in framework.df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
            
        present_new_cols = [col for col in new_cols if col in framework.df.columns]
        print(f"✅ New judge columns detected: {present_new_cols}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

def test_performance_metrics(csv_path: str):
    """Test performance metrics calculation"""
    print(f"***REMOVED***n🧪 Testing performance metrics calculation...")
    
    try:
        framework = LLMEvaluationFramework(csv_path)
        metrics = framework.calculate_performance_metrics()
        
        for model, metric in metrics.items():
            print(f"✅ {model}:")
            print(f"   - Accuracy: {metric.accuracy:.1%}")
            print(f"   - Avg Time: {metric.avg_completion_time:.1f}s")
            print(f"   - Avg Cost: ${metric.avg_cost:.4f}")
            print(f"   - Avg Confidence: {metric.avg_confidence:.2f}")
            print(f"   - Error Detection Rate: {metric.error_detection_rate:.1%}")
            print(f"   - Token Efficiency: {metric.token_efficiency:.2f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return False

def test_benchmark_creation(csv_path: str):
    """Test benchmark suite creation"""
    print(f"***REMOVED***n🧪 Testing benchmark suite creation...")
    
    try:
        framework = LLMEvaluationFramework(csv_path)
        benchmark_results = framework.create_benchmark_suite()
        
        for model, result in benchmark_results.items():
            print(f"✅ {model}:")
            print(f"   - Overall Score: {result.overall_score:.3f}")
            print(f"   - Rank: {result.rank}")
            print(f"   - Complexity Scores: {result.complexity_scores}")
            print(f"   - Quality Scores: {result.quality_scores}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error creating benchmarks: {e}")
        return False

def test_comprehensive_suite(csv_path: str):
    """Test the comprehensive benchmarking suite"""
    print(f"***REMOVED***n🧪 Testing comprehensive benchmarking suite...")

    try:
        # Create a unique output directory for testing (Option 1 approach)
        from pathlib import Path
        csv_file = Path(csv_path)
        # Extract model and prompt info from filename: model_promptlength_subset_date.csv
        filename_parts = csv_file.stem.split('_')
        if len(filename_parts) >= 3:
            model_name = filename_parts[0]
            prompt_length = filename_parts[1]
            subset_type = filename_parts[2]
            test_output_dir = f"test_benchmark_{model_name}_{prompt_length}_{subset_type}"
        else:
            test_output_dir = "test_benchmark_output"

        suite = ComprehensiveBenchmarkingSuite(
            results_path=csv_path,
            output_dir=test_output_dir
        )
        
        # Run the comprehensive benchmark
        suite_result = suite.run_comprehensive_benchmark()
        
        print(f"✅ Comprehensive benchmark completed!")
        print(f"   - Models evaluated: {len(suite_result.models_evaluated)}")
        print(f"   - Total tasks: {suite_result.total_tasks}")
        print(f"   - Manufacturing metrics calculated: {len(suite_result.manufacturing_metrics)}")
        print(f"   - Recommendations generated: {len(suite_result.recommendations)}")
        
        # Test export functionality
        json_path = suite.export_benchmark_suite(suite_result, "test_results.json")
        print(f"✅ Results exported to: {json_path}")
        
        # Test executive summary
        summary = suite.generate_executive_summary(suite_result)
        print(f"✅ Executive summary generated ({len(summary)} characters)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in comprehensive suite: {e}")
        return False

def test_judge_metrics_integration(csv_path: str):
    """Test integration with new judge metrics"""
    print(f"***REMOVED***n🧪 Testing judge metrics integration...")
    
    try:
        framework = LLMEvaluationFramework(csv_path)
        
        # Check if judge columns are available
        if 'judge_consensus_score' in framework.df.columns:
            print("✅ Judge consensus scores available")
            
            # Calculate some basic stats on judge metrics
            consensus_scores = framework.df['judge_consensus_score']
            print(f"   - Avg consensus score: {consensus_scores.mean():.3f}")
            print(f"   - Min consensus score: {consensus_scores.min():.3f}")
            print(f"   - Max consensus score: {consensus_scores.max():.3f}")
            
            # Check agreement levels
            if 'agreement_level' in framework.df.columns:
                agreement_counts = framework.df['agreement_level'].value_counts()
                print(f"   - Agreement levels: {dict(agreement_counts)}")
                
        else:
            print("⚠️  Judge consensus scores not found (using older format)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing judge metrics: {e}")
        return False

def main():
    """Main test function"""
    print("🏛️  Testing LLM Benchmarking System Compatibility")
    print("=" * 60)
    
    # Use the new CSV file
    csv_path = "performance_logs/deepseek-chat_short_all_20250616.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("Please ensure the file exists before running tests.")
        return False
    
    # Run all tests
    tests = [
        ("CSV Loading", test_csv_loading),
        ("Performance Metrics", test_performance_metrics),
        ("Benchmark Creation", test_benchmark_creation),
        ("Judge Metrics Integration", test_judge_metrics_integration),
        ("Comprehensive Suite", test_comprehensive_suite),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"***REMOVED***n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func(csv_path)
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"***REMOVED***n{'='*60}")
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"***REMOVED***nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Benchmarking system is fully compatible.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
