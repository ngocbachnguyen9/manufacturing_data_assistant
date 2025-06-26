#!/usr/bin/env python3
"""
Phase 5: Human vs LLM Comparison Framework

This module provides comprehensive comparison capabilities between human participants
and LLM models on manufacturing data analysis tasks, including statistical analysis,
cost-effectiveness evaluation, and visualization generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import warnings
import json
import glob
from dataclasses import dataclass
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class ComparisonConfig:
    """Configuration for human vs LLM comparison analysis"""
    human_hourly_rate: float = 25.0  # USD per hour for human labor cost
    output_dir: str = "experiments/phase5_analysis/results"
    visualization_dir: str = "experiments/phase5_analysis/visualizations"
    export_csv: bool = True
    generate_visualizations: bool = True

class HumanVsLLMComparison:
    """
    Comprehensive comparison framework for human vs LLM performance analysis
    """
    
    def __init__(self, config: ComparisonConfig = None):
        """Initialize the comparison framework"""
        self.config = config or ComparisonConfig()
        self.human_data = None
        self.llm_data = None
        self.aligned_data = None
        self.comparison_results = {}
        
        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.visualization_dir).mkdir(parents=True, exist_ok=True)
    
    def load_human_data(self, human_csv_path: str) -> pd.DataFrame:
        """Load and preprocess human study results"""
        print(f"Loading human data from: {human_csv_path}")
        
        # Handle potential BOM and encoding issues
        try:
            self.human_data = pd.read_csv(human_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.human_data = pd.read_csv(human_csv_path, encoding='utf-8-sig')
        
        # Clean column names (remove BOM if present)
        self.human_data.columns = self.human_data.columns.str.strip()
        
        # Standardize column names
        column_mapping = {
            'data quality': 'quality_condition',
            'completion_time_sec': 'completion_time_sec',
            'accuracy': 'is_correct'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in self.human_data.columns:
                self.human_data = self.human_data.rename(columns={old_col: new_col})
        
        # Add human cost calculation
        self.human_data['total_cost_usd'] = (
            self.human_data['completion_time_sec'] / 3600 * self.config.human_hourly_rate
        )
        
        # Add source identifier
        self.human_data['source'] = 'human'
        
        print(f"Loaded {len(self.human_data)} human task records")
        return self.human_data
    
    def load_llm_data(self, llm_results_dir: str) -> pd.DataFrame:
        """Load and aggregate LLM results from multiple model files"""
        print(f"Loading LLM data from: {llm_results_dir}")
        
        llm_files = glob.glob(f"{llm_results_dir}/*.csv")
        if not llm_files:
            raise FileNotFoundError(f"No CSV files found in {llm_results_dir}")
        
        all_llm_data = []
        
        for file_path in llm_files:
            print(f"  Processing: {Path(file_path).name}")
            try:
                df = pd.read_csv(file_path)
                
                # Convert boolean strings to actual booleans
                if 'is_correct' in df.columns:
                    df['is_correct'] = df['is_correct'].map({
                        'True': 1, 'False': 0, True: 1, False: 0, 1: 1, 0: 0
                    })
                
                all_llm_data.append(df)
            except Exception as e:
                print(f"    Warning: Could not load {file_path}: {e}")
                continue
        
        if not all_llm_data:
            raise ValueError("No valid LLM data files could be loaded")
        
        # Combine all LLM data
        self.llm_data = pd.concat(all_llm_data, ignore_index=True)
        
        # Add source identifier
        self.llm_data['source'] = 'llm'
        
        print(f"Loaded {len(self.llm_data)} LLM task records from {len(all_llm_data)} files")
        print(f"Models included: {sorted(self.llm_data['model'].unique())}")
        
        return self.llm_data
    
    def align_data_by_task_id(self) -> pd.DataFrame:
        """Align human and LLM data by task_id for direct comparison"""
        if self.human_data is None or self.llm_data is None:
            raise ValueError("Both human and LLM data must be loaded first")
        
        print("Aligning data by task_id...")
        
        # Get common task_ids
        human_tasks = set(self.human_data['task_id'].unique())
        llm_tasks = set(self.llm_data['task_id'].unique())
        common_tasks = human_tasks.intersection(llm_tasks)
        
        print(f"Human tasks: {len(human_tasks)}")
        print(f"LLM tasks: {len(llm_tasks)}")
        print(f"Common tasks: {len(common_tasks)}")
        
        if len(common_tasks) == 0:
            raise ValueError("No common task_ids found between human and LLM data")
        
        # Filter to common tasks only
        human_aligned = self.human_data[self.human_data['task_id'].isin(common_tasks)].copy()
        llm_aligned = self.llm_data[self.llm_data['task_id'].isin(common_tasks)].copy()
        
        # Combine aligned data
        self.aligned_data = pd.concat([human_aligned, llm_aligned], ignore_index=True)
        
        print(f"Aligned dataset contains {len(self.aligned_data)} records")
        print(f"  Human records: {len(human_aligned)}")
        print(f"  LLM records: {len(llm_aligned)}")
        
        return self.aligned_data
    
    def calculate_aggregate_llm_performance(self) -> pd.DataFrame:
        """Calculate average LLM performance across all models for each task"""
        if self.llm_data is None:
            raise ValueError("LLM data must be loaded first")
        
        print("Calculating aggregate LLM performance...")
        
        # Group by task_id and calculate averages
        llm_aggregated = self.llm_data.groupby('task_id').agg({
            'completion_time_sec': 'mean',
            'is_correct': 'mean',  # This gives us the accuracy rate
            'total_cost_usd': 'mean',
            'complexity': 'first',  # These should be the same for each task_id
            'quality_condition': 'first'
        }).reset_index()
        
        # Add metadata
        llm_aggregated['source'] = 'llm_aggregate'
        llm_aggregated['model_count'] = self.llm_data.groupby('task_id').size().values
        
        print(f"Aggregated {len(llm_aggregated)} LLM task averages")

        return llm_aggregated

    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive statistical comparison between human and LLM performance"""
        if self.aligned_data is None:
            self.align_data_by_task_id()

        print("Performing statistical analysis...")

        # Get aggregated LLM data for fair comparison
        llm_agg = self.calculate_aggregate_llm_performance()

        # Filter human data to common tasks
        common_tasks = set(llm_agg['task_id'])
        human_filtered = self.human_data[self.human_data['task_id'].isin(common_tasks)].copy()

        results = {
            'overall_comparison': {},
            'complexity_comparison': {},
            'quality_comparison': {},
            'statistical_tests': {},
            'summary_statistics': {}
        }

        # Overall comparison
        human_accuracy = human_filtered['is_correct'].mean()
        llm_accuracy = llm_agg['is_correct'].mean()
        human_avg_time = human_filtered['completion_time_sec'].mean()
        llm_avg_time = llm_agg['completion_time_sec'].mean()
        human_avg_cost = human_filtered['total_cost_usd'].mean()
        llm_avg_cost = llm_agg['total_cost_usd'].mean()

        results['overall_comparison'] = {
            'human_accuracy': human_accuracy,
            'llm_accuracy': llm_accuracy,
            'accuracy_difference': llm_accuracy - human_accuracy,
            'human_avg_time_sec': human_avg_time,
            'llm_avg_time_sec': llm_avg_time,
            'time_speedup_factor': human_avg_time / llm_avg_time if llm_avg_time > 0 else float('inf'),
            'human_avg_cost_usd': human_avg_cost,
            'llm_avg_cost_usd': llm_avg_cost,
            'cost_efficiency_ratio': human_avg_cost / llm_avg_cost if llm_avg_cost > 0 else float('inf')
        }

        # Statistical significance tests
        # Accuracy comparison (Chi-square test)
        human_correct = human_filtered['is_correct'].sum()
        human_total = len(human_filtered)
        llm_correct = (llm_agg['is_correct'] * llm_agg['model_count']).sum()
        llm_total = llm_agg['model_count'].sum()

        contingency_table = [[human_correct, human_total - human_correct],
                           [llm_correct, llm_total - llm_correct]]

        chi2_stat, chi2_p = chi2_contingency(contingency_table)[:2]

        results['statistical_tests']['accuracy_chi2'] = {
            'statistic': chi2_stat,
            'p_value': chi2_p,
            'significant': chi2_p < 0.05,
            'interpretation': 'Significant difference in accuracy' if chi2_p < 0.05 else 'No significant difference in accuracy'
        }

        # Time comparison (Mann-Whitney U test)
        # For this, we need to expand LLM aggregated data back to individual comparisons
        llm_times_expanded = []
        human_times_matched = []

        for _, task_row in llm_agg.iterrows():
            task_id = task_row['task_id']
            human_task_data = human_filtered[human_filtered['task_id'] == task_id]

            for _, human_row in human_task_data.iterrows():
                llm_times_expanded.append(task_row['completion_time_sec'])
                human_times_matched.append(human_row['completion_time_sec'])

        if len(llm_times_expanded) > 0 and len(human_times_matched) > 0:
            mw_stat, mw_p = mannwhitneyu(human_times_matched, llm_times_expanded, alternative='two-sided')

            results['statistical_tests']['time_mannwhitney'] = {
                'statistic': mw_stat,
                'p_value': mw_p,
                'significant': mw_p < 0.05,
                'interpretation': 'Significant difference in completion time' if mw_p < 0.05 else 'No significant difference in completion time'
            }

        # Analysis by complexity
        for complexity in ['easy', 'medium', 'hard']:
            human_comp = human_filtered[human_filtered['complexity'] == complexity]
            llm_comp = llm_agg[llm_agg['complexity'] == complexity]

            if len(human_comp) > 0 and len(llm_comp) > 0:
                results['complexity_comparison'][complexity] = {
                    'human_accuracy': human_comp['is_correct'].mean(),
                    'llm_accuracy': llm_comp['is_correct'].mean(),
                    'human_avg_time': human_comp['completion_time_sec'].mean(),
                    'llm_avg_time': llm_comp['completion_time_sec'].mean(),
                    'human_avg_cost': human_comp['total_cost_usd'].mean(),
                    'llm_avg_cost': llm_comp['total_cost_usd'].mean(),
                    'human_sample_size': len(human_comp),
                    'llm_sample_size': len(llm_comp)
                }

        # Analysis by data quality condition
        for quality in ['Q0', 'Q1', 'Q2', 'Q3']:
            human_qual = human_filtered[human_filtered['quality_condition'] == quality]
            llm_qual = llm_agg[llm_agg['quality_condition'] == quality]

            if len(human_qual) > 0 and len(llm_qual) > 0:
                results['quality_comparison'][quality] = {
                    'human_accuracy': human_qual['is_correct'].mean(),
                    'llm_accuracy': llm_qual['is_correct'].mean(),
                    'human_avg_time': human_qual['completion_time_sec'].mean(),
                    'llm_avg_time': llm_qual['completion_time_sec'].mean(),
                    'human_avg_cost': human_qual['total_cost_usd'].mean(),
                    'llm_avg_cost': llm_qual['total_cost_usd'].mean(),
                    'human_sample_size': len(human_qual),
                    'llm_sample_size': len(llm_qual)
                }

        # Summary statistics
        results['summary_statistics'] = {
            'total_common_tasks': len(common_tasks),
            'human_total_records': len(human_filtered),
            'llm_total_records': len(self.llm_data[self.llm_data['task_id'].isin(common_tasks)]),
            'llm_models_included': sorted(self.llm_data['model'].unique()),
            'analysis_timestamp': datetime.now().isoformat()
        }

        self.comparison_results = results
        print("Statistical analysis completed!")

        return results

    def export_results_to_csv(self) -> Dict[str, str]:
        """Export all comparison results to CSV files"""
        if not self.comparison_results:
            raise ValueError("No comparison results available. Run perform_statistical_analysis() first.")

        print("Exporting results to CSV...")

        exported_files = {}

        # 1. Overall comparison summary
        overall_df = pd.DataFrame([self.comparison_results['overall_comparison']])
        overall_path = Path(self.config.output_dir) / "overall_comparison.csv"
        overall_df.to_csv(overall_path, index=False)
        exported_files['overall_comparison'] = str(overall_path)

        # 2. Statistical tests results
        stats_data = []
        for test_name, test_results in self.comparison_results['statistical_tests'].items():
            row = {'test_name': test_name}
            row.update(test_results)
            stats_data.append(row)

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_path = Path(self.config.output_dir) / "statistical_tests.csv"
            stats_df.to_csv(stats_path, index=False)
            exported_files['statistical_tests'] = str(stats_path)

        # 3. Complexity comparison
        complexity_data = []
        for complexity, metrics in self.comparison_results['complexity_comparison'].items():
            row = {'complexity': complexity}
            row.update(metrics)
            complexity_data.append(row)

        if complexity_data:
            complexity_df = pd.DataFrame(complexity_data)
            complexity_path = Path(self.config.output_dir) / "complexity_comparison.csv"
            complexity_df.to_csv(complexity_path, index=False)
            exported_files['complexity_comparison'] = str(complexity_path)

        # 4. Quality condition comparison
        quality_data = []
        for quality, metrics in self.comparison_results['quality_comparison'].items():
            row = {'quality_condition': quality}
            row.update(metrics)
            quality_data.append(row)

        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            quality_path = Path(self.config.output_dir) / "quality_comparison.csv"
            quality_df.to_csv(quality_path, index=False)
            exported_files['quality_comparison'] = str(quality_path)

        # 5. Detailed task-level comparison
        if self.aligned_data is not None:
            # Create task-level summary
            llm_agg = self.calculate_aggregate_llm_performance()
            common_tasks = set(llm_agg['task_id'])
            human_filtered = self.human_data[self.human_data['task_id'].isin(common_tasks)].copy()

            task_comparison = []
            for task_id in common_tasks:
                human_task = human_filtered[human_filtered['task_id'] == task_id]
                llm_task = llm_agg[llm_agg['task_id'] == task_id].iloc[0]

                for _, human_row in human_task.iterrows():
                    comparison_row = {
                        'task_id': task_id,
                        'complexity': human_row['complexity'],
                        'quality_condition': human_row['quality_condition'],
                        'human_accuracy': human_row['is_correct'],
                        'llm_accuracy': llm_task['is_correct'],
                        'human_time_sec': human_row['completion_time_sec'],
                        'llm_time_sec': llm_task['completion_time_sec'],
                        'human_cost_usd': human_row['total_cost_usd'],
                        'llm_cost_usd': llm_task['total_cost_usd'],
                        'participant_id': human_row.get('participant_id', 'N/A'),
                        'llm_model_count': llm_task['model_count']
                    }
                    task_comparison.append(comparison_row)

            if task_comparison:
                task_df = pd.DataFrame(task_comparison)
                task_path = Path(self.config.output_dir) / "task_level_comparison.csv"
                task_df.to_csv(task_path, index=False)
                exported_files['task_level_comparison'] = str(task_path)

        # 6. Summary metadata
        summary_df = pd.DataFrame([self.comparison_results['summary_statistics']])
        summary_path = Path(self.config.output_dir) / "analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        exported_files['analysis_summary'] = str(summary_path)

        print(f"Exported {len(exported_files)} CSV files:")
        for name, path in exported_files.items():
            print(f"  {name}: {path}")

        return exported_files

    def create_visualization_dashboard(self) -> Dict[str, str]:
        """Create comprehensive visualization dashboard"""
        if not self.comparison_results:
            raise ValueError("No comparison results available. Run perform_statistical_analysis() first.")

        print("Creating visualization dashboard...")

        # Set up plotting parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

        generated_plots = {}

        # 1. Overall Performance Comparison
        self._plot_overall_performance_comparison()
        overall_path = Path(self.config.visualization_dir) / "overall_performance_comparison.png"
        plt.savefig(overall_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['overall_performance'] = str(overall_path)

        # 2. Performance by Complexity
        self._plot_performance_by_complexity()
        complexity_path = Path(self.config.visualization_dir) / "performance_by_complexity.png"
        plt.savefig(complexity_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['performance_by_complexity'] = str(complexity_path)

        # 3. Performance by Data Quality
        self._plot_performance_by_quality()
        quality_path = Path(self.config.visualization_dir) / "performance_by_quality.png"
        plt.savefig(quality_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['performance_by_quality'] = str(quality_path)

        # 4. Time vs Accuracy Scatter Plot
        self._plot_time_vs_accuracy_scatter()
        scatter_path = Path(self.config.visualization_dir) / "time_vs_accuracy_scatter.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['time_vs_accuracy'] = str(scatter_path)

        # 5. Cost Analysis
        self._plot_cost_analysis()
        cost_path = Path(self.config.visualization_dir) / "cost_analysis.png"
        plt.savefig(cost_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['cost_analysis'] = str(cost_path)

        print(f"Generated {len(generated_plots)} visualization files:")
        for name, path in generated_plots.items():
            print(f"  {name}: {path}")

        return generated_plots

    def _plot_overall_performance_comparison(self):
        """Create overall performance comparison chart"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Human vs LLM: Overall Performance Comparison', fontsize=16, fontweight='bold')

        overall = self.comparison_results['overall_comparison']

        # Accuracy comparison
        accuracy_data = [overall['human_accuracy'], overall['llm_accuracy']]
        accuracy_labels = ['Human', 'LLM (Avg)']
        colors = ['#FF6B6B', '#4ECDC4']

        axes[0, 0].bar(accuracy_labels, accuracy_data, color=colors, alpha=0.8)
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy Rate')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracy_data):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # Time comparison
        time_data = [overall['human_avg_time_sec'], overall['llm_avg_time_sec']]
        time_labels = ['Human', 'LLM (Avg)']

        axes[0, 1].bar(time_labels, time_data, color=colors, alpha=0.8)
        axes[0, 1].set_title('Average Completion Time')
        axes[0, 1].set_ylabel('Time (seconds)')
        for i, v in enumerate(time_data):
            axes[0, 1].text(i, v + max(time_data) * 0.02, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold')

        # Cost comparison
        cost_data = [overall['human_avg_cost_usd'], overall['llm_avg_cost_usd']]
        cost_labels = ['Human', 'LLM (Avg)']

        axes[1, 0].bar(cost_labels, cost_data, color=colors, alpha=0.8)
        axes[1, 0].set_title('Average Cost per Task')
        axes[1, 0].set_ylabel('Cost (USD)')
        for i, v in enumerate(cost_data):
            axes[1, 0].text(i, v + max(cost_data) * 0.02, f'${v:.3f}', ha='center', va='bottom', fontweight='bold')

        # Efficiency metrics
        speedup = overall.get('time_speedup_factor', 0)
        cost_ratio = overall.get('cost_efficiency_ratio', 0)

        efficiency_metrics = ['Time Speedup***REMOVED***n(Human/LLM)', 'Cost Ratio***REMOVED***n(Human/LLM)']
        efficiency_values = [speedup if speedup != float('inf') else 0,
                           cost_ratio if cost_ratio != float('inf') else 0]

        bars = axes[1, 1].bar(efficiency_metrics, efficiency_values, color=['#95E1D3', '#F38BA8'], alpha=0.8)
        axes[1, 1].set_title('Efficiency Metrics')
        axes[1, 1].set_ylabel('Ratio')

        for i, (bar, v) in enumerate(zip(bars, efficiency_values)):
            if v > 0:
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, v + max(efficiency_values) * 0.02,
                               f'{v:.1f}x', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

    def _plot_performance_by_complexity(self):
        """Create performance comparison by task complexity"""
        complexity_data = self.comparison_results['complexity_comparison']

        if not complexity_data:
            print("No complexity data available for plotting")
            return

        complexities = list(complexity_data.keys())
        human_accuracy = [complexity_data[c]['human_accuracy'] for c in complexities]
        llm_accuracy = [complexity_data[c]['llm_accuracy'] for c in complexities]
        human_time = [complexity_data[c]['human_avg_time'] for c in complexities]
        llm_time = [complexity_data[c]['llm_avg_time'] for c in complexities]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Performance by Task Complexity', fontsize=16, fontweight='bold')

        x = np.arange(len(complexities))
        width = 0.35

        # Accuracy by complexity
        axes[0].bar(x - width/2, human_accuracy, width, label='Human', color='#FF6B6B', alpha=0.8)
        axes[0].bar(x + width/2, llm_accuracy, width, label='LLM (Avg)', color='#4ECDC4', alpha=0.8)
        axes[0].set_title('Accuracy by Complexity')
        axes[0].set_ylabel('Accuracy Rate')
        axes[0].set_xlabel('Task Complexity')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([c.capitalize() for c in complexities])
        axes[0].legend()
        axes[0].set_ylim(0, 1)

        # Time by complexity
        axes[1].bar(x - width/2, human_time, width, label='Human', color='#FF6B6B', alpha=0.8)
        axes[1].bar(x + width/2, llm_time, width, label='LLM (Avg)', color='#4ECDC4', alpha=0.8)
        axes[1].set_title('Average Time by Complexity')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].set_xlabel('Task Complexity')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([c.capitalize() for c in complexities])
        axes[1].legend()

        plt.tight_layout()

    def _plot_performance_by_quality(self):
        """Create performance comparison by data quality condition"""
        quality_data = self.comparison_results['quality_comparison']

        if not quality_data:
            print("No quality data available for plotting")
            return

        qualities = list(quality_data.keys())
        human_accuracy = [quality_data[q]['human_accuracy'] for q in qualities]
        llm_accuracy = [quality_data[q]['llm_accuracy'] for q in qualities]
        human_time = [quality_data[q]['human_avg_time'] for q in qualities]
        llm_time = [quality_data[q]['llm_avg_time'] for q in qualities]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Performance by Data Quality Condition', fontsize=16, fontweight='bold')

        x = np.arange(len(qualities))
        width = 0.35

        # Accuracy by quality
        axes[0].bar(x - width/2, human_accuracy, width, label='Human', color='#FF6B6B', alpha=0.8)
        axes[0].bar(x + width/2, llm_accuracy, width, label='LLM (Avg)', color='#4ECDC4', alpha=0.8)
        axes[0].set_title('Accuracy by Data Quality')
        axes[0].set_ylabel('Accuracy Rate')
        axes[0].set_xlabel('Data Quality Condition')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(qualities)
        axes[0].legend()
        axes[0].set_ylim(0, 1)

        # Time by quality
        axes[1].bar(x - width/2, human_time, width, label='Human', color='#FF6B6B', alpha=0.8)
        axes[1].bar(x + width/2, llm_time, width, label='LLM (Avg)', color='#4ECDC4', alpha=0.8)
        axes[1].set_title('Average Time by Data Quality')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].set_xlabel('Data Quality Condition')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(qualities)
        axes[1].legend()

        plt.tight_layout()

    def _plot_time_vs_accuracy_scatter(self):
        """Create time vs accuracy scatter plot"""
        if self.aligned_data is None:
            print("No aligned data available for scatter plot")
            return

        # Get aggregated data for plotting
        llm_agg = self.calculate_aggregate_llm_performance()
        common_tasks = set(llm_agg['task_id'])
        human_filtered = self.human_data[self.human_data['task_id'].isin(common_tasks)].copy()

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Human data points
        human_times = human_filtered['completion_time_sec']
        human_accuracy = human_filtered['is_correct']

        # LLM data points (expand for each human task)
        llm_times = []
        llm_accuracy = []

        for _, human_row in human_filtered.iterrows():
            task_id = human_row['task_id']
            llm_task = llm_agg[llm_agg['task_id'] == task_id]
            if len(llm_task) > 0:
                llm_times.append(llm_task.iloc[0]['completion_time_sec'])
                llm_accuracy.append(llm_task.iloc[0]['is_correct'])

        # Create scatter plot
        ax.scatter(human_times, human_accuracy, alpha=0.6, s=60, color='#FF6B6B', label='Human', edgecolors='black', linewidth=0.5)
        ax.scatter(llm_times, llm_accuracy, alpha=0.6, s=60, color='#4ECDC4', label='LLM (Avg)', edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Completion Time (seconds)')
        ax.set_ylabel('Accuracy (0=Incorrect, 1=Correct)')
        ax.set_title('Time vs Accuracy: Human vs LLM Performance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add trend lines if possible
        if len(human_times) > 1:
            z_human = np.polyfit(human_times, human_accuracy, 1)
            p_human = np.poly1d(z_human)
            ax.plot(sorted(human_times), p_human(sorted(human_times)), "--", color='#FF6B6B', alpha=0.8, linewidth=2)

        if len(llm_times) > 1:
            z_llm = np.polyfit(llm_times, llm_accuracy, 1)
            p_llm = np.poly1d(z_llm)
            ax.plot(sorted(llm_times), p_llm(sorted(llm_times)), "--", color='#4ECDC4', alpha=0.8, linewidth=2)

        plt.tight_layout()

    def _plot_cost_analysis(self):
        """Create comprehensive cost analysis visualization"""
        overall = self.comparison_results['overall_comparison']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cost-Effectiveness Analysis: Human vs LLM', fontsize=16, fontweight='bold')

        # 1. Cost per task
        cost_data = [overall['human_avg_cost_usd'], overall['llm_avg_cost_usd']]
        cost_labels = ['Human', 'LLM (Avg)']
        colors = ['#FF6B6B', '#4ECDC4']

        axes[0, 0].bar(cost_labels, cost_data, color=colors, alpha=0.8)
        axes[0, 0].set_title('Average Cost per Task')
        axes[0, 0].set_ylabel('Cost (USD)')
        for i, v in enumerate(cost_data):
            axes[0, 0].text(i, v + max(cost_data) * 0.02, f'${v:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Cost efficiency (cost per correct answer)
        human_cost_per_correct = overall['human_avg_cost_usd'] / overall['human_accuracy'] if overall['human_accuracy'] > 0 else 0
        llm_cost_per_correct = overall['llm_avg_cost_usd'] / overall['llm_accuracy'] if overall['llm_accuracy'] > 0 else 0

        efficiency_data = [human_cost_per_correct, llm_cost_per_correct]
        axes[0, 1].bar(cost_labels, efficiency_data, color=colors, alpha=0.8)
        axes[0, 1].set_title('Cost per Correct Answer')
        axes[0, 1].set_ylabel('Cost (USD)')
        for i, v in enumerate(efficiency_data):
            axes[0, 1].text(i, v + max(efficiency_data) * 0.02, f'${v:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. Time-cost relationship
        time_data = [overall['human_avg_time_sec'], overall['llm_avg_time_sec']]

        axes[1, 0].scatter([overall['human_avg_time_sec']], [overall['human_avg_cost_usd']],
                          s=200, color='#FF6B6B', alpha=0.8, label='Human', edgecolors='black', linewidth=2)
        axes[1, 0].scatter([overall['llm_avg_time_sec']], [overall['llm_avg_cost_usd']],
                          s=200, color='#4ECDC4', alpha=0.8, label='LLM (Avg)', edgecolors='black', linewidth=2)
        axes[1, 0].set_xlabel('Average Time (seconds)')
        axes[1, 0].set_ylabel('Average Cost (USD)')
        axes[1, 0].set_title('Time vs Cost Trade-off')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. ROI comparison (accuracy per dollar)
        human_roi = overall['human_accuracy'] / overall['human_avg_cost_usd'] if overall['human_avg_cost_usd'] > 0 else 0
        llm_roi = overall['llm_accuracy'] / overall['llm_avg_cost_usd'] if overall['llm_avg_cost_usd'] > 0 else 0

        roi_data = [human_roi, llm_roi]
        axes[1, 1].bar(cost_labels, roi_data, color=colors, alpha=0.8)
        axes[1, 1].set_title('Return on Investment (Accuracy/Cost)')
        axes[1, 1].set_ylabel('Accuracy per USD')
        for i, v in enumerate(roi_data):
            axes[1, 1].text(i, v + max(roi_data) * 0.02, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

    def run_complete_analysis(self, human_csv_path: str, llm_results_dir: str) -> Dict[str, Any]:
        """Run the complete Phase 5 analysis pipeline"""
        print("=" * 60)
        print("PHASE 5: HUMAN vs LLM COMPARATIVE ANALYSIS")
        print("=" * 60)

        # Load data
        self.load_human_data(human_csv_path)
        self.load_llm_data(llm_results_dir)

        # Perform analysis
        results = self.perform_statistical_analysis()

        # Export results
        if self.config.export_csv:
            exported_files = self.export_results_to_csv()
            results['exported_files'] = exported_files

        # Generate visualizations
        if self.config.generate_visualizations:
            generated_plots = self.create_visualization_dashboard()
            results['generated_plots'] = generated_plots

        print("***REMOVED***n" + "=" * 60)
        print("PHASE 5 ANALYSIS COMPLETE")
        print("=" * 60)

        return results
