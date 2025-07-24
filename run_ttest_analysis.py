#!/usr/bin/env python3

import sys
sys.path.append('.')

from experiments.phase5_analysis.human_vs_llm_comparison import HumanVsLLMComparison, ComparisonConfig

def main():
    print("🧪 Running T-Test Analysis...")
    
    # Configure analysis with T-tests
    config = ComparisonConfig(generate_visualizations=False)
    comparison = HumanVsLLMComparison(config)
    
    # Load data
    comparison.load_human_data('experiments/human_study/Participants_results.csv')
    comparison.load_llm_data('experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark')
    
    # Perform analysis
    results = comparison.perform_statistical_analysis()
    
    # Print ALL statistical test results
    if 'statistical_tests' in results:
        stats = results['statistical_tests']
        
        print("***REMOVED***n📊 ALL STATISTICAL TEST RESULTS")
        print("=" * 50)
        
        # Print all available tests
        for test_name, test_results in stats.items():
            print(f"***REMOVED***n{test_name.upper()}:")
            if isinstance(test_results, dict):
                for key, value in test_results.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {test_results}")
        
        # Specifically look for T-test
        if 'accuracy_ttest' in stats:
            acc_ttest = stats['accuracy_ttest']
            print(f"***REMOVED***n🎯 ACCURACY T-TEST:")
            print(f"  t-statistic: {acc_ttest['statistic']:.4f}")
            print(f"  p-value: {acc_ttest['p_value']:.6f}")
            print(f"  Significant: {acc_ttest['significant']}")
            print(f"  Direction: {acc_ttest['effect_direction']}")
        else:
            print(f"***REMOVED***n❌ No T-test results found in statistical_tests")
            print(f"Available tests: {list(stats.keys())}")

if __name__ == "__main__":
    main()
