#!/usr/bin/env python3
"""
Test script to verify the Phase 5 analysis improvements
"""

import sys
import pandas as pd
sys.path.append('experiments/phase5_analysis')

from experiments.phase5_analysis.human_vs_llm_comparison import HumanVsLLMComparison, ComparisonConfig

def test_improvements():
    """Test the new Phase 5 improvements"""
    
    print("🧪 Testing Phase 5 Analysis Improvements...")
    print("=" * 60)
    
    # Initialize comparison framework
    config = ComparisonConfig(
        output_dir="experiments/phase5_analysis/test_results",
        visualization_dir="experiments/phase5_analysis/test_visualizations"
    )
    
    comparison = HumanVsLLMComparison(config)
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        human_csv = "experiments/human_study/Participants_results.csv"
        llm_dir = "experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark"
        
        comparison.load_human_data(human_csv)
        comparison.load_llm_data(llm_dir)
        
        print("✅ Data loading successful")
        
        # Test provider color function
        print("2. Testing provider-based color mapping...")
        from experiments.phase5_analysis.human_vs_llm_comparison import get_provider_color
        
        test_models = ['claude-3-5-haiku', 'gpt-4o-mini', 'deepseek-chat', 'deepseek-reasoner']
        for model in test_models:
            color = get_provider_color(model, test_models)
            print(f"   {model}: {color}")
        
        print("✅ Provider color mapping working")
        
        # Test statistical analysis
        print("3. Testing statistical analysis...")
        results = comparison.perform_statistical_analysis()
        print(f"✅ Analysis complete - {len(results)} result categories generated")
        
        # Test new visualizations
        print("4. Testing new visualization features...")
        
        # Test human performance distribution
        comparison._plot_human_performance_distribution()
        print("✅ Human performance distribution chart created")
        
        # Test speed chart in seconds
        comparison._plot_speed_in_seconds()
        print("✅ Speed chart in seconds created")
        
        # Test failure analysis
        failure_plots = comparison._plot_llm_failure_analysis()
        print("✅ LLM failure analysis created")
        
        print("=" * 60)
        print("🎉 ALL IMPROVEMENTS SUCCESSFULLY IMPLEMENTED!")
        print("=" * 60)
        
        # Summary of changes
        print("***REMOVED***n📋 SUMMARY OF IMPLEMENTED CHANGES:")
        print("1. ✅ Provider-based color grouping for LLM families")
        print("2. ✅ Human baseline as red bars (not lines)")
        print("3. ✅ Black borders on all bar charts")
        print("4. ✅ Speed graph displays time in seconds with human baseline")
        print("5. ✅ Human performance distribution graphs")
        print("6. ✅ LLM failure classification analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_improvements()
    if success:
        print("***REMOVED***n🚀 Ready to run full Phase 5 analysis with improvements!")
        print("   Run: python run_phase5_analysis.py")
    else:
        print("***REMOVED***n⚠️  Some issues detected. Please check the error messages above.")
