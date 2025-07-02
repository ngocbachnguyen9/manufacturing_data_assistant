#!/usr/bin/env python3
"""
Debug script to compare complexity and quality chart generation
"""

import sys
sys.path.insert(0, 'experiments/phase5_analysis')

from human_vs_llm_comparison import HumanVsLLMComparison, ComparisonConfig
from individual_visualizations import plot_complexity_performance_comparison, plot_quality_comparison_bars

def main():
    print("🔍 DEBUGGING CHART GENERATION")
    print("=" * 50)
    
    # Load data
    config = ComparisonConfig(generate_visualizations=False)
    comparison = HumanVsLLMComparison(config)
    comparison.load_human_data('experiments/human_study/Participants_results.csv')
    comparison.load_llm_data('experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark')
    results = comparison.perform_statistical_analysis()
    
    # Get data
    complexity_data = results['model_complexity_analysis']
    quality_data = results['model_quality_analysis']
    models = list(complexity_data.keys())
    complexities = ['easy', 'medium', 'hard']
    qualities = ['Q0', 'Q1', 'Q2', 'Q3']
    
    print(f"📊 Models: {len(models)} models")
    print(f"🧩 Complexities: {complexities}")
    print(f"🛡️ Qualities: {qualities}")
    
    # Generate complexity chart
    print("***REMOVED***n🧩 Generating complexity chart...")
    plot_complexity_performance_comparison(
        complexity_data, models, complexities,
        "debug_complexity_chart.png"
    )
    print("✅ Complexity chart saved as debug_complexity_chart.png")
    
    # Generate quality chart
    print("***REMOVED***n🛡️ Generating quality chart...")
    plot_quality_comparison_bars(
        quality_data, models, qualities,
        "debug_quality_chart.png"
    )
    print("✅ Quality chart saved as debug_quality_chart.png")
    
    print("***REMOVED***n🎯 COMPARISON COMPLETE")
    print("Compare the two files:")
    print("  - debug_complexity_chart.png")
    print("  - debug_quality_chart.png")

if __name__ == "__main__":
    main()
