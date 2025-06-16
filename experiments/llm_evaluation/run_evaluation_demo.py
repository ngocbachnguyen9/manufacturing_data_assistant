#!/usr/bin/env python3
"""
LLM Evaluation System Demo Runner

This script demonstrates the comprehensive LLM evaluation and benchmarking system
by running all components and generating sample outputs.
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from evaluation_framework import LLMEvaluationFramework
from model_comparison import ModelComparison
from prompt_effectiveness_analyzer import PromptEffectivenessAnalyzer
from benchmarking_suite import ComprehensiveBenchmarkingSuite

def run_evaluation_demo():
    """Run a comprehensive demonstration of the evaluation system"""
    
    print("🚀 LLM Manufacturing Data Analysis - Evaluation System Demo")
    print("=" * 70)
    print()
    
    try:
        # 1. Basic Performance Evaluation
        print("📊 1. Running Basic Performance Evaluation...")
        print("-" * 50)
        
        framework = LLMEvaluationFramework()
        
        # Generate performance report
        report = framework.generate_performance_report(
            "demo_performance_report.md"
        )
        print("✅ Performance report generated!")

        # Export benchmark results
        benchmark_data = framework.export_benchmark_results(
            "demo_benchmark_results.json"
        )
        print("✅ Benchmark results exported!")
        print()
        
        # 2. Model Comparison Analysis
        print("🔍 2. Running Model Comparison Analysis...")
        print("-" * 50)
        
        comparison = ModelComparison()
        
        # Statistical comparison
        stats_results = comparison.compare_models_statistical()
        print("✅ Statistical comparison completed!")
        
        # Generate visualizations
        comparison.create_comparison_visualizations(
            "demo_visualizations/"
        )
        print("✅ Comparison visualizations created!")

        # Generate comparison report
        comp_report = comparison.generate_comparison_report(
            "demo_model_comparison_report.md"
        )
        print("✅ Model comparison report generated!")
        print()
        
        # 3. Prompt Effectiveness Analysis
        print("📝 3. Running Prompt Effectiveness Analysis...")
        print("-" * 50)
        
        prompt_analyzer = PromptEffectivenessAnalyzer()
        
        # Analyze prompt effectiveness
        prompt_performance = prompt_analyzer.analyze_prompt_effectiveness()
        print("✅ Prompt effectiveness analysis completed!")
        
        # Create visualizations
        prompt_analyzer.create_prompt_comparison_visualizations(
            "demo_prompt_analysis/"
        )
        print("✅ Prompt analysis visualizations created!")

        # Generate report
        prompt_report = prompt_analyzer.generate_prompt_effectiveness_report(
            "demo_prompt_effectiveness_report.md"
        )
        print("✅ Prompt effectiveness report generated!")
        print()
        
        # 4. Comprehensive Benchmarking Suite
        print("🏆 4. Running Comprehensive Benchmarking Suite...")
        print("-" * 50)
        
        suite = ComprehensiveBenchmarkingSuite(
            output_dir="demo_benchmark_suite/"
        )
        
        # Run complete benchmark suite
        results, json_path, summary_path = suite.run_and_export_complete_suite()
        print("✅ Comprehensive benchmarking suite completed!")
        print()
        
        # 5. Summary of Results
        print("📋 5. Demo Results Summary")
        print("-" * 50)
        
        print(f"Models evaluated: {len(results.models_evaluated)}")
        print(f"Total tasks analyzed: {results.total_tasks}")
        print()
        
        print("Generated Files:")
        print(f"  📄 Performance Report: demo_performance_report.md")
        print(f"  📊 Benchmark Results: demo_benchmark_results.json")
        print(f"  📈 Visualizations: demo_visualizations/")
        print(f"  🔍 Model Comparison: demo_model_comparison_report.md")
        print(f"  📝 Prompt Analysis: demo_prompt_analysis/")
        print(f"  📋 Prompt Report: demo_prompt_effectiveness_report.md")
        print(f"  🏆 Complete Suite: {json_path}")
        print(f"  📋 Executive Summary: {summary_path}")
        print()
        
        # Display key findings
        print("🔍 Key Findings:")
        print("-" * 20)
        
        if results.benchmark_results:
            # Find top performer
            top_performer = max(results.benchmark_results.items(), 
                              key=lambda x: x[1].overall_score)
            print(f"🏆 Top Performer: {top_performer[0]}")
            print(f"   Overall Score: {top_performer[1].overall_score:.3f}")
            print(f"   Accuracy: {top_performer[1].metrics.accuracy:.1%}")
            print(f"   Avg Time: {top_performer[1].metrics.avg_completion_time:.1f}s")
            print(f"   Avg Cost: ${top_performer[1].metrics.avg_cost:.4f}")
            print()
        
        # Manufacturing-specific insights
        if 'manufacturing_domain_accuracy' in results.manufacturing_metrics:
            print("🏭 Manufacturing Task Performance:")
            domain_perf = results.manufacturing_metrics['manufacturing_domain_accuracy']
            for task_type, metrics in domain_perf.items():
                task_name = task_type.replace('_', ' ').title()
                print(f"   {task_name}: {metrics['accuracy']:.1%} accuracy")
            print()
        
        # Data quality handling
        if 'data_quality_handling' in results.manufacturing_metrics:
            print("🔧 Data Quality Handling:")
            quality_perf = results.manufacturing_metrics['data_quality_handling']
            quality_desc = {
                'Q0': 'Perfect Data', 
                'Q1': 'Space Injection Errors', 
                'Q2': 'Character Missing Errors', 
                'Q3': 'Missing Record Errors'
            }
            for quality, metrics in quality_perf.items():
                desc = quality_desc.get(quality, quality)
                print(f"   {desc}: {metrics['accuracy']:.1%} accuracy")
            print()
        
        print("✅ Demo completed successfully!")
        print()
        print("💡 Next Steps:")
        print("   1. Review the generated reports and visualizations")
        print("   2. Use the benchmarking results to select optimal models")
        print("   3. Implement prompt variations based on effectiveness analysis")
        print("   4. Set up continuous evaluation with new models and tasks")
        
    except Exception as e:
        print(f"❌ Error during demo execution: {str(e)}")
        print("Please check that the performance results file exists and is properly formatted.")
        return False
    
    return True

def print_system_overview():
    """Print an overview of the evaluation system capabilities"""
    
    print("🔧 LLM Evaluation System Overview")
    print("=" * 40)
    print()
    
    print("📊 EVALUATION FRAMEWORK")
    print("   • Performance metrics calculation")
    print("   • Benchmark scoring and ranking")
    print("   • Comprehensive reporting")
    print()
    
    print("🔍 MODEL COMPARISON")
    print("   • Statistical significance testing")
    print("   • Performance visualization")
    print("   • Multi-dimensional analysis")
    print()
    
    print("📝 PROMPT EFFECTIVENESS")
    print("   • Short vs Normal vs Long prompts")
    print("   • Token efficiency analysis")
    print("   • Time-accuracy trade-offs")
    print()
    
    print("🏆 BENCHMARKING SUITE")
    print("   • Unified evaluation pipeline")
    print("   • Manufacturing-specific metrics")
    print("   • Executive summaries")
    print("   • Actionable recommendations")
    print()
    
    print("📈 VISUALIZATION CAPABILITIES")
    print("   • Performance comparison charts")
    print("   • Statistical analysis plots")
    print("   • Cost-benefit analysis")
    print("   • Error detection heatmaps")
    print()

if __name__ == "__main__":
    print_system_overview()
    print()
    
    # Ask user if they want to run the demo
    response = input("Would you like to run the evaluation demo? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = run_evaluation_demo()
        if success:
            print("***REMOVED***n🎉 Demo completed successfully! Check the generated files for detailed results.")
        else:
            print("***REMOVED***n❌ Demo encountered errors. Please check the error messages above.")
    else:
        print("***REMOVED***n👋 Demo skipped. You can run this script anytime to see the evaluation system in action!")
