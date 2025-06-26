#!/usr/bin/env python3
"""
Phase 5: Human vs LLM Comparative Analysis Runner

This script orchestrates the complete Phase 5 analysis comparing human study results
with LLM evaluation results, including statistical analysis, cost-effectiveness evaluation,
and comprehensive visualization generation.
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add the experiments directory to the path
sys.path.append('experiments/phase5_analysis')

from human_vs_llm_comparison import HumanVsLLMComparison, ComparisonConfig
from cost_analysis import CostAnalyzer

def create_analysis_report(comparison_results: dict, cost_report: dict, output_path: str):
    """Create a comprehensive markdown report of the analysis"""
    
    report_lines = []
    report_lines.append("# Phase 5: Human vs LLM Comparative Analysis Report")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    
    overall = comparison_results['overall_comparison']
    
    report_lines.append(f"- **Human Accuracy:** {overall['human_accuracy']:.3f} ({overall['human_accuracy']*100:.1f}%)")
    report_lines.append(f"- **LLM Accuracy:** {overall['llm_accuracy']:.3f} ({overall['llm_accuracy']*100:.1f}%)")
    report_lines.append(f"- **Accuracy Difference:** {overall['accuracy_difference']:+.3f} ({overall['accuracy_difference']*100:+.1f}%)")
    report_lines.append("")
    
    report_lines.append(f"- **Human Avg Time:** {overall['human_avg_time_sec']:.1f} seconds")
    report_lines.append(f"- **LLM Avg Time:** {overall['llm_avg_time_sec']:.1f} seconds")
    report_lines.append(f"- **Speed Improvement:** {overall['time_speedup_factor']:.1f}x faster")
    report_lines.append("")
    
    report_lines.append(f"- **Human Avg Cost:** ${overall['human_avg_cost_usd']:.3f} per task")
    report_lines.append(f"- **LLM Avg Cost:** ${overall['llm_avg_cost_usd']:.3f} per task")
    report_lines.append(f"- **Cost Efficiency:** {overall['cost_efficiency_ratio']:.1f}x more cost-effective")
    report_lines.append("")
    
    # Statistical Significance
    report_lines.append("## Statistical Analysis")
    report_lines.append("")
    
    if 'statistical_tests' in comparison_results:
        stats = comparison_results['statistical_tests']
        
        if 'accuracy_chi2' in stats:
            acc_test = stats['accuracy_chi2']
            report_lines.append("### Accuracy Comparison")
            report_lines.append(f"- **Chi-square test:** χ² = {acc_test['statistic']:.4f}, p = {acc_test['p_value']:.4f}")
            report_lines.append(f"- **Result:** {acc_test['interpretation']}")
            report_lines.append("")
        
        if 'time_mannwhitney' in stats:
            time_test = stats['time_mannwhitney']
            report_lines.append("### Completion Time Comparison")
            report_lines.append(f"- **Mann-Whitney U test:** U = {time_test['statistic']:.4f}, p = {time_test['p_value']:.4f}")
            report_lines.append(f"- **Result:** {time_test['interpretation']}")
            report_lines.append("")
    
    # Performance by Complexity
    report_lines.append("## Performance by Task Complexity")
    report_lines.append("")
    
    if 'complexity_comparison' in comparison_results:
        complexity_data = comparison_results['complexity_comparison']
        
        report_lines.append("| Complexity | Human Accuracy | LLM Accuracy | Human Time (s) | LLM Time (s) |")
        report_lines.append("|------------|----------------|--------------|----------------|--------------|")
        
        for complexity, metrics in complexity_data.items():
            report_lines.append(f"| {complexity.capitalize()} | {metrics['human_accuracy']:.3f} | {metrics['llm_accuracy']:.3f} | {metrics['human_avg_time']:.1f} | {metrics['llm_avg_time']:.1f} |")
        
        report_lines.append("")
    
    # Performance by Data Quality
    report_lines.append("## Performance by Data Quality Condition")
    report_lines.append("")

    if 'quality_comparison' in comparison_results:
        quality_data = comparison_results['quality_comparison']

        report_lines.append("| Quality | Human Accuracy | LLM Accuracy | Human Time (s) | LLM Time (s) |")
        report_lines.append("|---------|----------------|--------------|----------------|--------------|")

        for quality, metrics in quality_data.items():
            report_lines.append(f"| {quality} | {metrics['human_accuracy']:.3f} | {metrics['llm_accuracy']:.3f} | {metrics['human_avg_time']:.1f} | {metrics['llm_avg_time']:.1f} |")

        report_lines.append("")

    # Model-specific Performance
    report_lines.append("## Model-Specific Performance vs Human Baseline")
    report_lines.append("")

    if 'model_specific_comparison' in comparison_results:
        model_data = comparison_results['model_specific_comparison']

        report_lines.append("| Model | Accuracy | Accuracy Diff | Speed Factor | Cost Ratio | Acc. Significant | Time Significant |")
        report_lines.append("|-------|----------|---------------|--------------|------------|------------------|------------------|")

        for model, metrics in model_data.items():
            acc_sig = "✓" if metrics['statistical_tests']['accuracy_chi2']['significant'] else "✗"
            time_sig = "✓" if metrics['statistical_tests']['time_mannwhitney'] and metrics['statistical_tests']['time_mannwhitney']['significant'] else "✗"

            report_lines.append(f"| {model} | {metrics['model_accuracy']:.3f} | {metrics['accuracy_difference']:+.3f} | {metrics['time_speedup_factor']:.1f}x | {metrics['cost_efficiency_ratio']:.1f}x | {acc_sig} | {time_sig} |")

        report_lines.append("")
        report_lines.append("*✓ = Statistically significant difference (p < 0.05), ✗ = Not significant*")
        report_lines.append("")

    # Model-specific Complexity Analysis
    if 'model_complexity_analysis' in comparison_results:
        report_lines.append("## Model Performance by Task Complexity")
        report_lines.append("")

        complexity_data = comparison_results['model_complexity_analysis']

        # Create summary table showing best model for each complexity
        report_lines.append("### Best Models by Complexity Level")
        report_lines.append("")

        complexities = ['easy', 'medium', 'hard']
        for complexity in complexities:
            best_acc = 0
            best_model = None
            best_improvement = 0

            for model, model_data in complexity_data.items():
                if complexity in model_data:
                    acc = model_data[complexity]['model_accuracy']
                    improvement = model_data[complexity]['accuracy_difference']
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model
                        best_improvement = improvement

            if best_model:
                report_lines.append(f"- **{complexity.capitalize()}**: {best_model} ({best_acc:.3f} accuracy, {best_improvement:+.3f} vs human)")

        report_lines.append("")

    # Model-specific Quality Analysis
    if 'model_quality_analysis' in comparison_results:
        report_lines.append("## Model Performance by Data Quality Condition")
        report_lines.append("")

        quality_data = comparison_results['model_quality_analysis']

        # Create robustness analysis
        report_lines.append("### Data Quality Robustness Analysis")
        report_lines.append("")

        robustness_scores = []
        for model, model_data in quality_data.items():
            if 'Q0' in model_data:
                q0_acc = model_data['Q0']['model_accuracy']
                corrupted_accs = []
                for q in ['Q1', 'Q2', 'Q3']:
                    if q in model_data:
                        corrupted_accs.append(model_data[q]['model_accuracy'])

                if corrupted_accs:
                    avg_corrupted = sum(corrupted_accs) / len(corrupted_accs)
                    robustness = avg_corrupted / q0_acc if q0_acc > 0 else 0
                    robustness_scores.append((model, robustness, q0_acc, avg_corrupted))

        # Sort by robustness score
        robustness_scores.sort(key=lambda x: x[1], reverse=True)

        report_lines.append("| Model | Robustness Score | Perfect Data (Q0) | Corrupted Avg (Q1-Q3) |")
        report_lines.append("|-------|------------------|-------------------|------------------------|")

        for model, robustness, q0_acc, corrupted_avg in robustness_scores:
            status = "🟢" if robustness > 0.8 else "🟡" if robustness > 0.6 else "🔴"
            report_lines.append(f"| {model} {status} | {robustness:.3f} | {q0_acc:.3f} | {corrupted_avg:.3f} |")

        report_lines.append("")
        report_lines.append("*🟢 = Robust (>0.8), 🟡 = Moderate (0.6-0.8), 🔴 = Fragile (<0.6)*")
        report_lines.append("")
    
    # Cost Analysis
    report_lines.append("## Cost-Effectiveness Analysis")
    report_lines.append("")
    
    if cost_report and 'summary' in cost_report:
        summary = cost_report['summary']
        report_lines.append(f"- **Best ROI Scenario:** {summary['best_roi_scenario']}")
        report_lines.append(f"- **Best ROI Percentage:** {summary['best_roi_percentage']:.1f}%")
        report_lines.append(f"- **Average Cost Savings:** ${summary['average_cost_savings']:.3f} per task")
        report_lines.append("")
        
        if 'recommendations' in cost_report:
            report_lines.append("### Recommendations")
            for rec in cost_report['recommendations']:
                report_lines.append(f"- {rec}")
            report_lines.append("")
    
    # Data Summary
    report_lines.append("## Data Summary")
    report_lines.append("")
    
    if 'summary_statistics' in comparison_results:
        summary_stats = comparison_results['summary_statistics']
        report_lines.append(f"- **Total Common Tasks:** {summary_stats['total_common_tasks']}")
        report_lines.append(f"- **Human Records:** {summary_stats['human_total_records']}")
        report_lines.append(f"- **LLM Records:** {summary_stats['llm_total_records']}")
        report_lines.append(f"- **LLM Models:** {', '.join(summary_stats['llm_models_included'])}")
        report_lines.append("")
    
    # Files Generated
    if 'exported_files' in comparison_results:
        report_lines.append("## Generated Files")
        report_lines.append("")
        report_lines.append("### CSV Exports")
        for name, path in comparison_results['exported_files'].items():
            report_lines.append(f"- **{name.replace('_', ' ').title()}:** `{path}`")
        report_lines.append("")
    
    if 'generated_plots' in comparison_results:
        report_lines.append("### Visualizations")
        for name, path in comparison_results['generated_plots'].items():
            report_lines.append(f"- **{name.replace('_', ' ').title()}:** `{path}`")
        report_lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('***REMOVED***n'.join(report_lines))
    
    return output_path

def main():
    """Main execution function for Phase 5 analysis"""
    
    parser = argparse.ArgumentParser(description='Run Phase 5: Human vs LLM Comparative Analysis')
    parser.add_argument('--human-data', 
                       default='experiments/human_study/Participants_results.csv',
                       help='Path to human study results CSV file')
    parser.add_argument('--llm-data', 
                       default='experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark',
                       help='Path to directory containing LLM evaluation results')
    parser.add_argument('--output-dir', 
                       default='experiments/phase5_analysis/results',
                       help='Output directory for results')
    parser.add_argument('--viz-dir', 
                       default='experiments/phase5_analysis/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--human-hourly-rate', 
                       type=float, default=25.0,
                       help='Human hourly rate for cost analysis (USD)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV export')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-cost-analysis', action='store_true',
                       help='Skip detailed cost analysis')
    
    args = parser.parse_args()
    
    # Validate input paths
    human_path = Path(args.human_data)
    llm_path = Path(args.llm_data)
    
    if not human_path.exists():
        print(f"Error: Human data file not found: {human_path}")
        sys.exit(1)
    
    if not llm_path.exists():
        print(f"Error: LLM data directory not found: {llm_path}")
        sys.exit(1)
    
    # Configure analysis
    config = ComparisonConfig(
        human_hourly_rate=args.human_hourly_rate,
        output_dir=args.output_dir,
        visualization_dir=args.viz_dir,
        export_csv=not args.no_csv,
        generate_visualizations=not args.no_viz
    )
    
    # Run main comparison analysis
    print("Starting Phase 5: Human vs LLM Comparative Analysis")
    print(f"Human data: {human_path}")
    print(f"LLM data: {llm_path}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    comparison = HumanVsLLMComparison(config)
    results = comparison.run_complete_analysis(str(human_path), str(llm_path))
    
    # Run detailed cost analysis if requested
    cost_report = None
    if not args.no_cost_analysis:
        print("***REMOVED***nRunning detailed cost analysis...")
        
        cost_analyzer = CostAnalyzer()
        
        # Extract performance metrics for cost analysis
        overall = results['overall_comparison']
        human_perf = {
            'accuracy': overall['human_accuracy'],
            'avg_time_sec': overall['human_avg_time_sec']
        }
        llm_perf = {
            'accuracy': overall['llm_accuracy'],
            'avg_cost_usd': overall['llm_avg_cost_usd']
        }
        
        cost_report = cost_analyzer.generate_cost_comparison_report(human_perf, llm_perf)
        
        # Generate cost visualization
        cost_viz_path = Path(args.viz_dir) / "detailed_cost_analysis.png"
        cost_analyzer.create_cost_visualization(cost_report, str(cost_viz_path))
        print(f"Cost analysis visualization saved: {cost_viz_path}")
    
    # Generate comprehensive report
    report_path = Path(args.output_dir) / "phase5_analysis_report.md"
    create_analysis_report(results, cost_report, str(report_path))
    print(f"***REMOVED***nComprehensive analysis report saved: {report_path}")
    
    # Save results as JSON for programmatic access
    json_path = Path(args.output_dir) / "phase5_results.json"
    with open(json_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    print(f"Results JSON saved: {json_path}")
    
    print("***REMOVED***n" + "="*60)
    print("PHASE 5 ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Key findings:")

    # Extract overall results for summary
    overall = results['overall_comparison']
    print(f"  - LLM accuracy: {overall['llm_accuracy']:.1%} vs Human: {overall['human_accuracy']:.1%}")
    print(f"  - LLM speed: {overall['time_speedup_factor']:.1f}x faster than humans")
    print(f"  - Cost efficiency: {overall['cost_efficiency_ratio']:.1f}x more cost-effective")

    if cost_report:
        print(f"  - Best ROI scenario: {cost_report['summary']['best_roi_scenario']}")
        print(f"  - ROI percentage: {cost_report['summary']['best_roi_percentage']:.1f}%")

    # Show top 3 models
    if 'model_specific_comparison' in results:
        import pandas as pd
        model_df = pd.DataFrame.from_dict(results['model_specific_comparison'], orient='index')
        top_models = model_df.nlargest(3, 'model_accuracy')

        print(f"***REMOVED***n🏆 Top 3 Models:")
        for i, (model, data) in enumerate(top_models.iterrows(), 1):
            acc_diff = data['accuracy_difference']
            speed = data['time_speedup_factor']
            print(f"  {i}. {model}: {data['model_accuracy']:.1%} ({acc_diff:+.1%}), {speed:.1f}x faster")

    print(f"***REMOVED***n📊 For detailed model rankings, run:")
    print(f"   python experiments/phase5_analysis/model_ranking_summary.py")

if __name__ == "__main__":
    main()
