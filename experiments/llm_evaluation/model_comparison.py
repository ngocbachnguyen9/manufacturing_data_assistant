#!/usr/bin/env python3
"""
LLM Model Comparison and Visualization System

This module provides comprehensive comparison capabilities between different LLM models
on manufacturing data analysis tasks, including statistical analysis and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelComparison:
    """
    Comprehensive model comparison system for LLM performance analysis
    """

    def __init__(self, results_path: str = "performance_logs/llm_performance_results.csv"):
        """Initialize the model comparison system"""
        self.results_path = Path(results_path)
        self.df = None
        self.load_results()

    def load_results(self) -> None:
        """Load performance results from CSV"""
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        self.df = pd.read_csv(self.results_path)
        print(f"Loaded {len(self.df)} performance records for comparison")

    def compare_models_statistical(self) -> Dict[str, Any]:
        """
        Perform statistical comparison between models
        Returns statistical test results and significance levels
        """
        models = self.df['model'].unique()
        if len(models) < 2:
            print("Warning: Need at least 2 models for comparison")
            return {}

        results = {
            'accuracy_comparison': {},
            'time_comparison': {},
            'cost_comparison': {},
            'confidence_comparison': {},
            'statistical_summary': {}
        }

        # Accuracy comparison (Chi-square test for categorical data)
        accuracy_data = []
        for model in models:
            model_data = self.df[self.df['model'] == model]
            correct = model_data['is_correct'].sum()
            total = len(model_data)
            accuracy_data.append([correct, total - correct])

        if len(accuracy_data) >= 2:
            chi2, p_value = chi2_contingency(accuracy_data)[:2]
            results['accuracy_comparison'] = {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': 'Significant difference in accuracy' if p_value < 0.05 else 'No significant difference in accuracy'
            }

        # Time comparison (Mann-Whitney U test for non-parametric data)
        if len(models) == 2:
            model1, model2 = models
            time1 = self.df[self.df['model'] == model1]['completion_time_sec']
            time2 = self.df[self.df['model'] == model2]['completion_time_sec']

            statistic, p_value = mannwhitneyu(time1, time2, alternative='two-sided')
            results['time_comparison'] = {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': f'Significant difference in completion time' if p_value < 0.05 else 'No significant difference in completion time'
            }

        # Summary statistics by model
        for model in models:
            model_data = self.df[self.df['model'] == model]
            results['statistical_summary'][model] = {
                'accuracy_mean': model_data['is_correct'].mean(),
                'accuracy_std': model_data['is_correct'].std(),
                'time_mean': model_data['completion_time_sec'].mean(),
                'time_std': model_data['completion_time_sec'].std(),
                'cost_mean': model_data['total_cost_usd'].mean(),
                'cost_std': model_data['total_cost_usd'].std(),
                'confidence_mean': model_data['final_confidence'].mean(),
                'confidence_std': model_data['final_confidence'].std(),
                'sample_size': len(model_data)
            }

        return results

    def create_comparison_visualizations(self, output_dir: str = "experiments/llm_evaluation/visualizations/"):
        """Create comprehensive comparison visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. Overall Performance Comparison
        self._plot_overall_performance(output_path)

        # 2. Performance by Complexity
        self._plot_performance_by_complexity(output_path)

        # 3. Performance by Quality Condition
        self._plot_performance_by_quality(output_path)

        # 4. Time vs Accuracy Scatter
        self._plot_time_vs_accuracy(output_path)

        # 5. Cost Analysis
        self._plot_cost_analysis(output_path)

        # 6. Error Detection Capabilities
        self._plot_error_detection(output_path)

        print(f"Visualizations saved to: {output_path}")

    def _plot_overall_performance(self, output_path: Path):
        """Plot overall performance metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Overall Model Performance Comparison', fontsize=16, fontweight='bold')

        models = self.df['model'].unique()

        # Accuracy
        accuracy_data = [self.df[self.df['model'] == model]['is_correct'].mean() for model in models]
        axes[0, 0].bar(models, accuracy_data, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracy_data):
            axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')

        # Average Completion Time
        time_data = [self.df[self.df['model'] == model]['completion_time_sec'].mean() for model in models]
        axes[0, 1].bar(models, time_data, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Average Completion Time')
        axes[0, 1].set_ylabel('Time (seconds)')
        for i, v in enumerate(time_data):
            axes[0, 1].text(i, v + 0.5, f'{v:.1f}s', ha='center', va='bottom')

        # Average Cost
        cost_data = [self.df[self.df['model'] == model]['total_cost_usd'].mean() for model in models]
        axes[1, 0].bar(models, cost_data, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Average Cost per Task')
        axes[1, 0].set_ylabel('Cost (USD)')
        for i, v in enumerate(cost_data):
            axes[1, 0].text(i, v + 0.0001, f'${v:.4f}', ha='center', va='bottom')

        # Average Confidence
        confidence_data = [self.df[self.df['model'] == model]['final_confidence'].mean() for model in models]
        axes[1, 1].bar(models, confidence_data, color='gold', alpha=0.7)
        axes[1, 1].set_title('Average Confidence Score')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_ylim(0, 1)
        for i, v in enumerate(confidence_data):
            axes[1, 1].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / 'overall_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_by_complexity(self, output_path: Path):
        """Plot performance breakdown by task complexity"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Performance by Task Complexity', fontsize=16, fontweight='bold')

        complexities = ['easy', 'medium', 'hard']
        models = self.df['model'].unique()

        for i, complexity in enumerate(complexities):
            complexity_data = self.df[self.df['complexity'] == complexity]

            accuracy_by_model = []
            time_by_model = []

            for model in models:
                model_complexity_data = complexity_data[complexity_data['model'] == model]
                if len(model_complexity_data) > 0:
                    accuracy_by_model.append(model_complexity_data['is_correct'].mean())
                    time_by_model.append(model_complexity_data['completion_time_sec'].mean())
                else:
                    accuracy_by_model.append(0)
                    time_by_model.append(0)

            # Create dual-axis plot
            ax1 = axes[i]
            ax2 = ax1.twinx()

            x_pos = np.arange(len(models))
            width = 0.35

            bars1 = ax1.bar(x_pos - width/2, accuracy_by_model, width, label='Accuracy', color='skyblue', alpha=0.7)
            bars2 = ax2.bar(x_pos + width/2, time_by_model, width, label='Time (s)', color='lightcoral', alpha=0.7)

            ax1.set_title(f'{complexity.title()} Tasks')
            ax1.set_ylabel('Accuracy', color='blue')
            ax2.set_ylabel('Time (seconds)', color='red')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(models, rotation=45)
            ax1.set_ylim(0, 1)

            # Add value labels
            for j, (acc, time) in enumerate(zip(accuracy_by_model, time_by_model)):
                ax1.text(j - width/2, acc + 0.01, f'{acc:.1%}', ha='center', va='bottom', fontsize=8)
                ax2.text(j + width/2, time + 0.5, f'{time:.1f}s', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path / 'performance_by_complexity.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_by_quality(self, output_path: Path):
        """Plot performance breakdown by data quality condition"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance by Data Quality Condition', fontsize=16, fontweight='bold')

        quality_conditions = ['Q0', 'Q1', 'Q2', 'Q3']
        quality_labels = ['Q0 (Baseline)', 'Q1 (Spaces)', 'Q2 (Char Missing)', 'Q3 (Missing Records)']
        models = self.df['model'].unique()

        for i, (quality, label) in enumerate(zip(quality_conditions, quality_labels)):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            quality_data = self.df[self.df['quality_condition'] == quality]

            accuracy_by_model = []
            for model in models:
                model_quality_data = quality_data[quality_data['model'] == model]
                if len(model_quality_data) > 0:
                    accuracy_by_model.append(model_quality_data['is_correct'].mean())
                else:
                    accuracy_by_model.append(0)

            bars = ax.bar(models, accuracy_by_model, color='lightgreen', alpha=0.7)
            ax.set_title(label)
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0, 1)

            # Add value labels
            for j, acc in enumerate(accuracy_by_model):
                ax.text(j, acc + 0.01, f'{acc:.1%}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / 'performance_by_quality.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_time_vs_accuracy(self, output_path: Path):
        """Plot time vs accuracy scatter plot"""
        plt.figure(figsize=(12, 8))

        models = self.df['model'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))

        for model, color in zip(models, colors):
            model_data = self.df[self.df['model'] == model]
            plt.scatter(model_data['completion_time_sec'], model_data['is_correct'],
                       label=model, alpha=0.6, s=50, color=color)

        plt.xlabel('Completion Time (seconds)')
        plt.ylabel('Correctness (0=Incorrect, 1=Correct)')
        plt.title('Time vs Accuracy Trade-off Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add trend lines for each model
        for model, color in zip(models, colors):
            model_data = self.df[self.df['model'] == model]
            if len(model_data) > 1:
                z = np.polyfit(model_data['completion_time_sec'], model_data['is_correct'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(model_data['completion_time_sec'].min(),
                                    model_data['completion_time_sec'].max(), 100)
                plt.plot(x_trend, p(x_trend), color=color, linestyle='--', alpha=0.8)

        plt.tight_layout()
        plt.savefig(output_path / 'time_vs_accuracy_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cost_analysis(self, output_path: Path):
        """Plot cost analysis visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Cost Analysis', fontsize=16, fontweight='bold')

        models = self.df['model'].unique()

        # Cost per task
        cost_data = [self.df[self.df['model'] == model]['total_cost_usd'].mean() for model in models]
        axes[0].bar(models, cost_data, color='lightblue', alpha=0.7)
        axes[0].set_title('Average Cost per Task')
        axes[0].set_ylabel('Cost (USD)')
        for i, v in enumerate(cost_data):
            axes[0].text(i, v + 0.0001, f'${v:.4f}', ha='center', va='bottom')

        # Cost efficiency (correct answers per dollar)
        efficiency_data = []
        for model in models:
            model_data = self.df[self.df['model'] == model]
            total_cost = model_data['total_cost_usd'].sum()
            correct_answers = model_data['is_correct'].sum()
            efficiency = correct_answers / total_cost if total_cost > 0 else 0
            efficiency_data.append(efficiency)

        axes[1].bar(models, efficiency_data, color='lightcoral', alpha=0.7)
        axes[1].set_title('Cost Efficiency (Correct Answers per Dollar)')
        axes[1].set_ylabel('Correct Answers / USD')
        for i, v in enumerate(efficiency_data):
            axes[1].text(i, v + 1, f'{v:.0f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / 'cost_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_detection(self, output_path: Path):
        """Plot error detection capabilities"""
        plt.figure(figsize=(12, 8))

        models = self.df['model'].unique()
        quality_conditions = ['Q1', 'Q2', 'Q3']  # Only conditions with errors

        # Calculate error detection rates
        detection_rates = {}
        for model in models:
            detection_rates[model] = []
            for quality in quality_conditions:
                quality_data = self.df[(self.df['model'] == model) & (self.df['quality_condition'] == quality)]
                if len(quality_data) > 0:
                    # Count tasks where reconciliation_issues is not empty
                    detected = quality_data['reconciliation_issues'].apply(
                        lambda x: len(eval(x)) > 0 if isinstance(x, str) and x.strip() else False
                    ).sum()
                    rate = detected / len(quality_data)
                    detection_rates[model].append(rate)
                else:
                    detection_rates[model].append(0)

        # Create grouped bar chart
        x = np.arange(len(quality_conditions))
        width = 0.35

        for i, model in enumerate(models):
            offset = (i - len(models)/2 + 0.5) * width
            plt.bar(x + offset, detection_rates[model], width, label=model, alpha=0.7)

        plt.xlabel('Data Quality Condition')
        plt.ylabel('Error Detection Rate')
        plt.title('Error Detection Capabilities by Model and Quality Condition')
        plt.xticks(x, ['Q1 (Spaces)', 'Q2 (Char Missing)', 'Q3 (Missing Records)'])
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'error_detection_capabilities.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comparison_report(self, output_path: str = "experiments/llm_evaluation/model_comparison_report.md") -> str:
        """Generate comprehensive model comparison report"""
        statistical_results = self.compare_models_statistical()

        report = []
        report.append("# LLM Model Comparison Report")
        report.append("=" * 50)
        report.append("")

        # Statistical Analysis Summary
        report.append("## Statistical Analysis Summary")
        report.append("")

        if 'accuracy_comparison' in statistical_results:
            acc_comp = statistical_results['accuracy_comparison']
            report.append("### Accuracy Comparison")
            report.append(f"- Chi-square statistic: {acc_comp.get('chi2_statistic', 'N/A'):.4f}")
            report.append(f"- P-value: {acc_comp.get('p_value', 'N/A'):.4f}")
            report.append(f"- Result: {acc_comp.get('interpretation', 'N/A')}")
            report.append("")

        if 'time_comparison' in statistical_results:
            time_comp = statistical_results['time_comparison']
            report.append("### Completion Time Comparison")
            report.append(f"- Mann-Whitney U statistic: {time_comp.get('statistic', 'N/A'):.4f}")
            report.append(f"- P-value: {time_comp.get('p_value', 'N/A'):.4f}")
            report.append(f"- Result: {time_comp.get('interpretation', 'N/A')}")
            report.append("")

        # Model Performance Summary
        report.append("## Model Performance Summary")
        report.append("")

        if 'statistical_summary' in statistical_results:
            for model, stats in statistical_results['statistical_summary'].items():
                report.append(f"### {model}")
                report.append(f"- **Accuracy**: {stats['accuracy_mean']:.1%} (±{stats['accuracy_std']:.3f})")
                report.append(f"- **Avg Time**: {stats['time_mean']:.1f}s (±{stats['time_std']:.1f}s)")
                report.append(f"- **Avg Cost**: ${stats['cost_mean']:.4f} (±${stats['cost_std']:.4f})")
                report.append(f"- **Avg Confidence**: {stats['confidence_mean']:.2f} (±{stats['confidence_std']:.2f})")
                report.append(f"- **Sample Size**: {stats['sample_size']} tasks")
                report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        models = self.df['model'].unique()
        if len(models) >= 2:
            # Find best performing model by accuracy
            best_accuracy_model = max(statistical_results['statistical_summary'].items(),
                                    key=lambda x: x[1]['accuracy_mean'])
            report.append(f"- **Highest Accuracy**: {best_accuracy_model[0]} ({best_accuracy_model[1]['accuracy_mean']:.1%})")

            # Find fastest model
            fastest_model = min(statistical_results['statistical_summary'].items(),
                              key=lambda x: x[1]['time_mean'])
            report.append(f"- **Fastest Completion**: {fastest_model[0]} ({fastest_model[1]['time_mean']:.1f}s avg)")

            # Find most cost-effective model
            most_efficient_model = min(statistical_results['statistical_summary'].items(),
                                     key=lambda x: x[1]['cost_mean'])
            report.append(f"- **Most Cost-Effective**: {most_efficient_model[0]} (${most_efficient_model[1]['cost_mean']:.4f} avg)")

        report_text = "***REMOVED***n".join(report)

        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Comparison report saved to: {output_path}")

        return report_text

if __name__ == "__main__":
    # Example usage
    comparison = ModelComparison()

    # Generate statistical comparison
    stats_results = comparison.compare_models_statistical()
    print("Statistical comparison completed!")

    # Create visualizations
    comparison.create_comparison_visualizations()
    print("Visualizations created!")

    # Generate comparison report
    report = comparison.generate_comparison_report()
    print("Comparison report generated!")