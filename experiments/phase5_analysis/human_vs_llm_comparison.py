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
            'model_specific_comparison': {},
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

        # Model-specific analysis
        models = self.llm_data['model'].unique()
        for model in models:
            model_data = self.llm_data[
                (self.llm_data['model'] == model) &
                (self.llm_data['task_id'].isin(common_tasks))
            ]

            if len(model_data) > 0:
                # Calculate model-specific metrics
                model_accuracy = model_data['is_correct'].mean()
                model_avg_time = model_data['completion_time_sec'].mean()
                model_avg_cost = model_data['total_cost_usd'].mean()

                # Statistical tests for this specific model vs humans
                model_correct = model_data['is_correct'].sum()
                model_total = len(model_data)
                human_correct = human_filtered['is_correct'].sum()
                human_total = len(human_filtered)

                # Chi-square test for accuracy
                contingency_table = [[human_correct, human_total - human_correct],
                                   [model_correct, model_total - model_correct]]
                chi2_stat, chi2_p = chi2_contingency(contingency_table)[:2]

                # Mann-Whitney U test for time (need to match tasks)
                model_times = []
                human_times_matched = []

                for task_id in model_data['task_id'].unique():
                    model_task_time = model_data[model_data['task_id'] == task_id]['completion_time_sec'].iloc[0]
                    human_task_data = human_filtered[human_filtered['task_id'] == task_id]

                    for _, human_row in human_task_data.iterrows():
                        model_times.append(model_task_time)
                        human_times_matched.append(human_row['completion_time_sec'])

                mw_stat, mw_p = None, None
                if len(model_times) > 0 and len(human_times_matched) > 0:
                    mw_stat, mw_p = mannwhitneyu(human_times_matched, model_times, alternative='two-sided')

                results['model_specific_comparison'][model] = {
                    'model_accuracy': model_accuracy,
                    'human_accuracy': human_filtered['is_correct'].mean(),
                    'accuracy_difference': model_accuracy - human_filtered['is_correct'].mean(),
                    'model_avg_time': model_avg_time,
                    'human_avg_time': human_filtered['completion_time_sec'].mean(),
                    'time_speedup_factor': human_filtered['completion_time_sec'].mean() / model_avg_time if model_avg_time > 0 else float('inf'),
                    'model_avg_cost': model_avg_cost,
                    'human_avg_cost': human_filtered['total_cost_usd'].mean(),
                    'cost_efficiency_ratio': human_filtered['total_cost_usd'].mean() / model_avg_cost if model_avg_cost > 0 else float('inf'),
                    'model_sample_size': len(model_data),
                    'human_sample_size': len(human_filtered),
                    'statistical_tests': {
                        'accuracy_chi2': {
                            'statistic': chi2_stat,
                            'p_value': chi2_p,
                            'significant': chi2_p < 0.05 if chi2_p is not None else False
                        },
                        'time_mannwhitney': {
                            'statistic': mw_stat,
                            'p_value': mw_p,
                            'significant': mw_p < 0.05 if mw_p is not None else False
                        } if mw_stat is not None else None
                    }
                }

        # Analysis by complexity (aggregated LLM)
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

        # Model-specific analysis by complexity
        results['model_complexity_analysis'] = {}
        for model in models:
            model_data = self.llm_data[
                (self.llm_data['model'] == model) &
                (self.llm_data['task_id'].isin(common_tasks))
            ]

            results['model_complexity_analysis'][model] = {}

            for complexity in ['easy', 'medium', 'hard']:
                human_comp = human_filtered[human_filtered['complexity'] == complexity]
                model_comp = model_data[model_data['complexity'] == complexity]

                if len(human_comp) > 0 and len(model_comp) > 0:
                    results['model_complexity_analysis'][model][complexity] = {
                        'human_accuracy': human_comp['is_correct'].mean(),
                        'model_accuracy': model_comp['is_correct'].mean(),
                        'accuracy_difference': model_comp['is_correct'].mean() - human_comp['is_correct'].mean(),
                        'human_avg_time': human_comp['completion_time_sec'].mean(),
                        'model_avg_time': model_comp['completion_time_sec'].mean(),
                        'time_speedup_factor': human_comp['completion_time_sec'].mean() / model_comp['completion_time_sec'].mean() if model_comp['completion_time_sec'].mean() > 0 else float('inf'),
                        'human_avg_cost': human_comp['total_cost_usd'].mean(),
                        'model_avg_cost': model_comp['total_cost_usd'].mean(),
                        'human_sample_size': len(human_comp),
                        'model_sample_size': len(model_comp)
                    }

        # Analysis by data quality condition (aggregated LLM)
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

        # Model-specific analysis by data quality condition
        results['model_quality_analysis'] = {}
        for model in models:
            model_data = self.llm_data[
                (self.llm_data['model'] == model) &
                (self.llm_data['task_id'].isin(common_tasks))
            ]

            results['model_quality_analysis'][model] = {}

            for quality in ['Q0', 'Q1', 'Q2', 'Q3']:
                human_qual = human_filtered[human_filtered['quality_condition'] == quality]
                model_qual = model_data[model_data['quality_condition'] == quality]

                if len(human_qual) > 0 and len(model_qual) > 0:
                    results['model_quality_analysis'][model][quality] = {
                        'human_accuracy': human_qual['is_correct'].mean(),
                        'model_accuracy': model_qual['is_correct'].mean(),
                        'accuracy_difference': model_qual['is_correct'].mean() - human_qual['is_correct'].mean(),
                        'human_avg_time': human_qual['completion_time_sec'].mean(),
                        'model_avg_time': model_qual['completion_time_sec'].mean(),
                        'time_speedup_factor': human_qual['completion_time_sec'].mean() / model_qual['completion_time_sec'].mean() if model_qual['completion_time_sec'].mean() > 0 else float('inf'),
                        'human_avg_cost': human_qual['total_cost_usd'].mean(),
                        'model_avg_cost': model_qual['total_cost_usd'].mean(),
                        'human_sample_size': len(human_qual),
                        'model_sample_size': len(model_qual)
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

        # 4. Model-specific comparison
        if 'model_specific_comparison' in self.comparison_results:
            model_data = []
            for model, metrics in self.comparison_results['model_specific_comparison'].items():
                row = {'model': model}
                # Flatten the nested structure
                for key, value in metrics.items():
                    if key == 'statistical_tests':
                        # Add statistical test results with prefixes
                        if value['accuracy_chi2']:
                            row['accuracy_chi2_statistic'] = value['accuracy_chi2']['statistic']
                            row['accuracy_chi2_p_value'] = value['accuracy_chi2']['p_value']
                            row['accuracy_chi2_significant'] = value['accuracy_chi2']['significant']
                        if value['time_mannwhitney']:
                            row['time_mw_statistic'] = value['time_mannwhitney']['statistic']
                            row['time_mw_p_value'] = value['time_mannwhitney']['p_value']
                            row['time_mw_significant'] = value['time_mannwhitney']['significant']
                    else:
                        row[key] = value
                model_data.append(row)

            if model_data:
                model_df = pd.DataFrame(model_data)
                model_path = Path(self.config.output_dir) / "model_specific_comparison.csv"
                model_df.to_csv(model_path, index=False)
                exported_files['model_specific_comparison'] = str(model_path)

        # 5. Quality condition comparison
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

        # 6. Model-specific complexity analysis
        if 'model_complexity_analysis' in self.comparison_results:
            complexity_model_data = []
            for model, complexities in self.comparison_results['model_complexity_analysis'].items():
                for complexity, metrics in complexities.items():
                    row = {'model': model, 'complexity': complexity}
                    row.update(metrics)
                    complexity_model_data.append(row)

            if complexity_model_data:
                complexity_model_df = pd.DataFrame(complexity_model_data)
                complexity_model_path = Path(self.config.output_dir) / "model_complexity_analysis.csv"
                complexity_model_df.to_csv(complexity_model_path, index=False)
                exported_files['model_complexity_analysis'] = str(complexity_model_path)

        # 7. Model-specific quality analysis
        if 'model_quality_analysis' in self.comparison_results:
            quality_model_data = []
            for model, qualities in self.comparison_results['model_quality_analysis'].items():
                for quality, metrics in qualities.items():
                    row = {'model': model, 'quality_condition': quality}
                    row.update(metrics)
                    quality_model_data.append(row)

            if quality_model_data:
                quality_model_df = pd.DataFrame(quality_model_data)
                quality_model_path = Path(self.config.output_dir) / "model_quality_analysis.csv"
                quality_model_df.to_csv(quality_model_path, index=False)
                exported_files['model_quality_analysis'] = str(quality_model_path)

        # 8. Detailed task-level comparison
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

        # 9. Summary metadata
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

        # 6. Model-specific Performance Comparison
        self._plot_model_specific_comparison()
        model_path = Path(self.config.visualization_dir) / "model_specific_comparison.png"
        plt.savefig(model_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['model_specific_comparison'] = str(model_path)

        # 7. Model-specific Complexity Analysis
        self._plot_model_complexity_analysis()
        complexity_path = Path(self.config.visualization_dir) / "model_complexity_analysis.png"
        plt.savefig(complexity_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['model_complexity_analysis'] = str(complexity_path)

        # 8. Model-specific Quality Analysis
        self._plot_model_quality_analysis()
        quality_path = Path(self.config.visualization_dir) / "model_quality_analysis.png"
        plt.savefig(quality_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['model_quality_analysis'] = str(quality_path)

        # 9. Enhanced Model Comparison Dashboard
        self._plot_enhanced_model_comparison_dashboard()
        dashboard_path = Path(self.config.visualization_dir) / "enhanced_model_comparison_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['enhanced_model_comparison_dashboard'] = str(dashboard_path)

        # 10. Complexity Performance Matrix
        self._plot_complexity_performance_matrix()
        complexity_matrix_path = Path(self.config.visualization_dir) / "complexity_performance_matrix.png"
        plt.savefig(complexity_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['complexity_performance_matrix'] = str(complexity_matrix_path)

        # 11. Quality Robustness Analysis
        self._plot_quality_robustness_analysis()
        robustness_path = Path(self.config.visualization_dir) / "quality_robustness_analysis.png"
        plt.savefig(robustness_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots['quality_robustness_analysis'] = str(robustness_path)

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

    def _plot_model_specific_comparison(self):
        """Create model-specific performance comparison"""
        if 'model_specific_comparison' not in self.comparison_results:
            print("No model-specific data available for plotting")
            return

        model_data = self.comparison_results['model_specific_comparison']
        models = list(model_data.keys())

        if not models:
            return

        # Create a comprehensive model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model-Specific Performance vs Human Baseline', fontsize=16, fontweight='bold')

        # 1. Accuracy comparison
        human_accuracy = model_data[models[0]]['human_accuracy']  # Same for all models
        model_accuracies = [model_data[model]['model_accuracy'] for model in models]

        x = np.arange(len(models))
        width = 0.35

        # Create human baseline line
        axes[0, 0].axhline(y=human_accuracy, color='red', linestyle='--', linewidth=2,
                          label=f'Human Baseline ({human_accuracy:.3f})', alpha=0.8)

        bars = axes[0, 0].bar(x, model_accuracies, width, color='lightblue', alpha=0.8,
                             edgecolor='navy', linewidth=1)
        axes[0, 0].set_title('Accuracy: Models vs Human Baseline')
        axes[0, 0].set_ylabel('Accuracy Rate')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=9)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)

        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, model_accuracies)):
            color = 'green' if acc > human_accuracy else 'red'
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', color=color)

        # 2. Time comparison (speedup factors)
        speedup_factors = [model_data[model]['time_speedup_factor'] for model in models]
        # Cap at reasonable maximum for visualization
        speedup_capped = [min(sf, 20) if sf != float('inf') else 20 for sf in speedup_factors]

        bars = axes[0, 1].bar(models, speedup_capped, color='lightgreen', alpha=0.8,
                             edgecolor='darkgreen', linewidth=1)
        axes[0, 1].set_title('Speed Improvement (Human Time / Model Time)')
        axes[0, 1].set_ylabel('Speedup Factor')
        axes[0, 1].set_xticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=9)
        axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No improvement')

        for i, (bar, sf) in enumerate(zip(bars, speedup_capped)):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, sf + max(speedup_capped) * 0.02,
                           f'{sf:.1f}x', ha='center', va='bottom', fontweight='bold')

        # 3. Statistical significance heatmap
        sig_data = []
        test_names = ['Accuracy', 'Time']

        for model in models:
            model_stats = model_data[model]['statistical_tests']
            row = []
            # Accuracy significance
            acc_sig = model_stats['accuracy_chi2']['significant'] if model_stats['accuracy_chi2'] else False
            row.append(1 if acc_sig else 0)
            # Time significance
            time_sig = model_stats['time_mannwhitney']['significant'] if model_stats['time_mannwhitney'] else False
            row.append(1 if time_sig else 0)
            sig_data.append(row)

        sig_array = np.array(sig_data)
        im = axes[1, 0].imshow(sig_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1, 0].set_title('Statistical Significance (p < 0.05)')
        axes[1, 0].set_xticks(range(len(test_names)))
        axes[1, 0].set_xticklabels(test_names)
        axes[1, 0].set_yticks(range(len(models)))
        axes[1, 0].set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=9)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(test_names)):
                text = 'Sig' if sig_array[i, j] == 1 else 'NS'
                axes[1, 0].text(j, i, text, ha='center', va='center',
                               color='white' if sig_array[i, j] == 1 else 'black', fontweight='bold')

        # 4. Accuracy difference from human baseline
        accuracy_diffs = [model_data[model]['accuracy_difference'] for model in models]

        colors = ['green' if diff > 0 else 'red' for diff in accuracy_diffs]
        bars = axes[1, 1].bar(models, accuracy_diffs, color=colors, alpha=0.7,
                             edgecolor='black', linewidth=1)
        axes[1, 1].set_title('Accuracy Difference from Human Baseline')
        axes[1, 1].set_ylabel('Accuracy Difference')
        axes[1, 1].set_xticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=9)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

        for i, (bar, diff) in enumerate(zip(bars, accuracy_diffs)):
            y_pos = diff + (0.01 if diff >= 0 else -0.01)
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, y_pos,
                           f'{diff:+.3f}', ha='center',
                           va='bottom' if diff >= 0 else 'top', fontweight='bold')

        plt.tight_layout()

    def _plot_model_complexity_analysis(self):
        """Create model-specific performance analysis by task complexity"""
        if 'model_complexity_analysis' not in self.comparison_results:
            print("No model complexity data available for plotting")
            return

        complexity_data = self.comparison_results['model_complexity_analysis']
        models = list(complexity_data.keys())
        complexities = ['easy', 'medium', 'hard']

        if not models:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance by Task Complexity', fontsize=16, fontweight='bold')

        # 1. Accuracy by complexity heatmap
        accuracy_matrix = []
        for model in models:
            row = []
            for complexity in complexities:
                if complexity in complexity_data[model]:
                    row.append(complexity_data[model][complexity]['model_accuracy'])
                else:
                    row.append(0)
            accuracy_matrix.append(row)

        accuracy_array = np.array(accuracy_matrix)
        im1 = axes[0, 0].imshow(accuracy_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0, 0].set_title('Model Accuracy by Complexity')
        axes[0, 0].set_xticks(range(len(complexities)))
        axes[0, 0].set_xticklabels(complexities)
        axes[0, 0].set_yticks(range(len(models)))
        axes[0, 0].set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=9)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(complexities)):
                if accuracy_array[i, j] > 0:
                    text = f'{accuracy_array[i, j]:.3f}'
                    axes[0, 0].text(j, i, text, ha='center', va='center',
                                   color='white' if accuracy_array[i, j] < 0.5 else 'black', fontweight='bold')

        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 2. Accuracy improvement over humans by complexity
        improvement_matrix = []
        for model in models:
            row = []
            for complexity in complexities:
                if complexity in complexity_data[model]:
                    row.append(complexity_data[model][complexity]['accuracy_difference'])
                else:
                    row.append(0)
            improvement_matrix.append(row)

        improvement_array = np.array(improvement_matrix)
        im2 = axes[0, 1].imshow(improvement_array, cmap='RdBu_r', aspect='auto',
                               vmin=-0.5, vmax=0.5)
        axes[0, 1].set_title('Accuracy Improvement over Humans')
        axes[0, 1].set_xticks(range(len(complexities)))
        axes[0, 1].set_xticklabels(complexities)
        axes[0, 1].set_yticks(range(len(models)))
        axes[0, 1].set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=9)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(complexities)):
                if improvement_array[i, j] != 0:
                    text = f'{improvement_array[i, j]:+.3f}'
                    color = 'white' if abs(improvement_array[i, j]) > 0.25 else 'black'
                    axes[0, 1].text(j, i, text, ha='center', va='center',
                                   color=color, fontweight='bold')

        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 3. Speed improvement by complexity
        speed_matrix = []
        for model in models:
            row = []
            for complexity in complexities:
                if complexity in complexity_data[model]:
                    speed = complexity_data[model][complexity]['time_speedup_factor']
                    row.append(min(speed, 25) if speed != float('inf') else 25)  # Cap for visualization
                else:
                    row.append(0)
            speed_matrix.append(row)

        speed_array = np.array(speed_matrix)
        im3 = axes[1, 0].imshow(speed_array, cmap='YlOrRd', aspect='auto', vmin=0, vmax=25)
        axes[1, 0].set_title('Speed Improvement (x faster)')
        axes[1, 0].set_xticks(range(len(complexities)))
        axes[1, 0].set_xticklabels(complexities)
        axes[1, 0].set_yticks(range(len(models)))
        axes[1, 0].set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=9)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(complexities)):
                if speed_array[i, j] > 0:
                    text = f'{speed_array[i, j]:.1f}x'
                    axes[1, 0].text(j, i, text, ha='center', va='center',
                                   color='white' if speed_array[i, j] > 12 else 'black', fontweight='bold')

        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 4. Best model by complexity
        best_models_by_complexity = {}
        for complexity in complexities:
            best_acc = 0
            best_model = None
            for model in models:
                if complexity in complexity_data[model]:
                    acc = complexity_data[model][complexity]['model_accuracy']
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model
            best_models_by_complexity[complexity] = (best_model, best_acc)

        # Create bar chart for best models
        best_accs = [best_models_by_complexity[c][1] for c in complexities]
        best_names = [best_models_by_complexity[c][0].replace('-', '***REMOVED***n') if best_models_by_complexity[c][0] else 'None' for c in complexities]

        bars = axes[1, 1].bar(complexities, best_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        axes[1, 1].set_title('Best Model by Complexity')
        axes[1, 1].set_ylabel('Best Accuracy')
        axes[1, 1].set_ylim(0, 1)

        # Add model names on bars
        for i, (bar, name, acc) in enumerate(zip(bars, best_names, best_accs)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                           name, ha='center', va='bottom', fontweight='bold', fontsize=8)
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, acc/2,
                           f'{acc:.3f}', ha='center', va='center', fontweight='bold', color='white')

        plt.tight_layout()

    def _plot_model_quality_analysis(self):
        """Create model-specific performance analysis by data quality condition"""
        if 'model_quality_analysis' not in self.comparison_results:
            print("No model quality data available for plotting")
            return

        quality_data = self.comparison_results['model_quality_analysis']
        models = list(quality_data.keys())
        qualities = ['Q0', 'Q1', 'Q2', 'Q3']

        if not models:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance by Data Quality Condition', fontsize=16, fontweight='bold')

        # 1. Accuracy by quality heatmap
        accuracy_matrix = []
        for model in models:
            row = []
            for quality in qualities:
                if quality in quality_data[model]:
                    row.append(quality_data[model][quality]['model_accuracy'])
                else:
                    row.append(0)
            accuracy_matrix.append(row)

        accuracy_array = np.array(accuracy_matrix)
        im1 = axes[0, 0].imshow(accuracy_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0, 0].set_title('Model Accuracy by Data Quality')
        axes[0, 0].set_xticks(range(len(qualities)))
        axes[0, 0].set_xticklabels(qualities)
        axes[0, 0].set_yticks(range(len(models)))
        axes[0, 0].set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=9)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(qualities)):
                if accuracy_array[i, j] > 0:
                    text = f'{accuracy_array[i, j]:.3f}'
                    axes[0, 0].text(j, i, text, ha='center', va='center',
                                   color='white' if accuracy_array[i, j] < 0.5 else 'black', fontweight='bold')

        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 2. Robustness analysis (Q0 vs corrupted data average)
        robustness_scores = []
        model_names = []

        for model in models:
            if 'Q0' in quality_data[model]:
                q0_acc = quality_data[model]['Q0']['model_accuracy']
                corrupted_accs = []
                for q in ['Q1', 'Q2', 'Q3']:
                    if q in quality_data[model]:
                        corrupted_accs.append(quality_data[model][q]['model_accuracy'])

                if corrupted_accs:
                    avg_corrupted = np.mean(corrupted_accs)
                    robustness = avg_corrupted / q0_acc if q0_acc > 0 else 0
                    robustness_scores.append(robustness)
                    model_names.append(model)

        if robustness_scores:
            colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' for score in robustness_scores]
            bars = axes[0, 1].bar(range(len(model_names)), robustness_scores, color=colors, alpha=0.7)
            axes[0, 1].set_title('Data Quality Robustness***REMOVED***n(Corrupted/Perfect Accuracy Ratio)')
            axes[0, 1].set_ylabel('Robustness Score')
            axes[0, 1].set_xticks(range(len(model_names)))
            axes[0, 1].set_xticklabels([m.replace('-', '***REMOVED***n') for m in model_names], fontsize=9)
            axes[0, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect robustness')
            axes[0, 1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good robustness')
            axes[0, 1].legend()

            for i, (bar, score) in enumerate(zip(bars, robustness_scores)):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, score + 0.02,
                               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. Quality degradation pattern
        quality_labels = ['Perfect***REMOVED***n(Q0)', 'Spaces***REMOVED***n(Q1)', 'Missing Chars***REMOVED***n(Q2)', 'Missing Records***REMOVED***n(Q3)']

        for i, model in enumerate(models[:4]):  # Show top 4 models to avoid clutter
            model_accs = []
            for quality in qualities:
                if quality in quality_data[model]:
                    model_accs.append(quality_data[model][quality]['model_accuracy'])
                else:
                    model_accs.append(0)

            axes[1, 0].plot(range(len(qualities)), model_accs, marker='o', linewidth=2,
                           label=model.replace('-', '***REMOVED***n'), alpha=0.8)

        axes[1, 0].set_title('Accuracy Degradation by Data Quality')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xticks(range(len(qualities)))
        axes[1, 0].set_xticklabels(quality_labels, fontsize=9)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Best model by quality condition
        best_models_by_quality = {}
        for quality in qualities:
            best_acc = 0
            best_model = None
            for model in models:
                if quality in quality_data[model]:
                    acc = quality_data[model][quality]['model_accuracy']
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model
            best_models_by_quality[quality] = (best_model, best_acc)

        # Create bar chart for best models
        best_accs = [best_models_by_quality[q][1] for q in qualities]
        best_names = [best_models_by_quality[q][0].replace('-', '***REMOVED***n') if best_models_by_quality[q][0] else 'None' for q in qualities]

        bars = axes[1, 1].bar(qualities, best_accs, color=['#2E8B57', '#FF6347', '#4169E1', '#FFD700'], alpha=0.8)
        axes[1, 1].set_title('Best Model by Data Quality')
        axes[1, 1].set_ylabel('Best Accuracy')
        axes[1, 1].set_ylim(0, 1)

        # Add model names on bars
        for i, (bar, name, acc) in enumerate(zip(bars, best_names, best_accs)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                           name, ha='center', va='bottom', fontweight='bold', fontsize=8)
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, acc/2,
                           f'{acc:.3f}', ha='center', va='center', fontweight='bold', color='white')

        plt.tight_layout()

    def _plot_enhanced_model_comparison_dashboard(self):
        """Create comprehensive model comparison dashboard"""
        if 'model_specific_comparison' not in self.comparison_results:
            print("No model-specific data available for enhanced dashboard")
            return

        model_data = self.comparison_results['model_specific_comparison']
        models = list(model_data.keys())

        if not models:
            return

        # Create a large comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        fig.suptitle('Enhanced Model Comparison Dashboard', fontsize=20, fontweight='bold', y=0.98)

        # 1. Accuracy vs Speed Scatter Plot (top-left, spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])

        accuracies = [model_data[model]['model_accuracy'] for model in models]
        speeds = [model_data[model]['time_speedup_factor'] for model in models]
        human_acc = model_data[models[0]]['human_accuracy']

        # Create bubble sizes based on statistical significance
        bubble_sizes = []
        colors = []
        for model in models:
            acc_sig = model_data[model]['statistical_tests']['accuracy_chi2']['significant']
            time_sig = model_data[model]['statistical_tests']['time_mannwhitney']['significant'] if model_data[model]['statistical_tests']['time_mannwhitney'] else False

            if acc_sig and time_sig:
                bubble_sizes.append(300)  # Large bubble
                colors.append('#2E8B57')  # Dark green
            elif acc_sig or time_sig:
                bubble_sizes.append(200)  # Medium bubble
                colors.append('#FFD700')  # Gold
            else:
                bubble_sizes.append(100)  # Small bubble
                colors.append('#FF6347')  # Red

        scatter = ax1.scatter(speeds, accuracies, s=bubble_sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=2)

        # Add model labels
        for i, model in enumerate(models):
            ax1.annotate(model.replace('-', '***REMOVED***n'), (speeds[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

        # Add human baseline
        ax1.axhline(y=human_acc, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Human Baseline ({human_acc:.3f})')
        ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='No Speed Improvement')

        ax1.set_xlabel('Speed Improvement Factor (log scale)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy vs Speed Performance***REMOVED***n(Bubble size = Statistical significance)', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Model Ranking Bar Chart (top-right)
        ax2 = fig.add_subplot(gs[0, 2:4])

        # Sort models by accuracy
        model_acc_sorted = sorted([(model, model_data[model]['model_accuracy']) for model in models],
                                 key=lambda x: x[1], reverse=True)

        model_names = [item[0].replace('-', '***REMOVED***n') for item in model_acc_sorted]
        model_accs = [item[1] for item in model_acc_sorted]

        # Color bars based on performance vs human
        bar_colors = ['green' if acc > human_acc else 'red' for acc in model_accs]

        bars = ax2.barh(range(len(model_names)), model_accs, color=bar_colors, alpha=0.7, edgecolor='black')
        ax2.axvline(x=human_acc, color='red', linestyle='--', linewidth=2, label=f'Human ({human_acc:.3f})')

        ax2.set_yticks(range(len(model_names)))
        ax2.set_yticklabels(model_names, fontsize=10)
        ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Model Accuracy Ranking', fontsize=14, fontweight='bold')
        ax2.legend()

        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, model_accs)):
            ax2.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.3f}',
                    va='center', fontweight='bold')

        # 3. Speed Ranking (middle-right)
        ax3 = fig.add_subplot(gs[1, 2:4])

        # Sort models by speed
        model_speed_sorted = sorted([(model, model_data[model]['time_speedup_factor']) for model in models],
                                   key=lambda x: x[1], reverse=True)

        speed_names = [item[0].replace('-', '***REMOVED***n') for item in model_speed_sorted]
        speed_values = [min(item[1], 25) if item[1] != float('inf') else 25 for item in model_speed_sorted]  # Cap for visualization

        bars = ax3.barh(range(len(speed_names)), speed_values, color='lightblue', alpha=0.7, edgecolor='navy')
        ax3.axvline(x=1, color='gray', linestyle='--', linewidth=2, label='No improvement')

        ax3.set_yticks(range(len(speed_names)))
        ax3.set_yticklabels(speed_names, fontsize=10)
        ax3.set_xlabel('Speed Improvement Factor', fontsize=12, fontweight='bold')
        ax3.set_title('Speed Improvement Ranking', fontsize=14, fontweight='bold')
        ax3.legend()

        # Add value labels
        for i, (bar, speed) in enumerate(zip(bars, speed_values)):
            ax3.text(speed + 0.5, bar.get_y() + bar.get_height()/2, f'{speed:.1f}x',
                    va='center', fontweight='bold')

        # 4. Statistical Significance Matrix (bottom-left)
        ax4 = fig.add_subplot(gs[2, 0:2])

        sig_matrix = []
        test_names = ['Accuracy***REMOVED***nSignificant', 'Time***REMOVED***nSignificant']

        for model in models:
            model_stats = model_data[model]['statistical_tests']
            row = []
            acc_sig = model_stats['accuracy_chi2']['significant'] if model_stats['accuracy_chi2'] else False
            time_sig = model_stats['time_mannwhitney']['significant'] if model_stats['time_mannwhitney'] else False
            row.extend([1 if acc_sig else 0, 1 if time_sig else 0])
            sig_matrix.append(row)

        sig_array = np.array(sig_matrix)
        im = ax4.imshow(sig_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax4.set_xticks(range(len(test_names)))
        ax4.set_xticklabels(test_names, fontsize=11, fontweight='bold')
        ax4.set_yticks(range(len(models)))
        ax4.set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=10)
        ax4.set_title('Statistical Significance Matrix', fontsize=14, fontweight='bold')

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(test_names)):
                text = '✓' if sig_array[i, j] == 1 else '✗'
                color = 'white' if sig_array[i, j] == 1 else 'black'
                ax4.text(j, i, text, ha='center', va='center', color=color, fontsize=14, fontweight='bold')

        # 5. Performance Improvement Radar Chart (bottom-right)
        ax5 = fig.add_subplot(gs[2, 2:4], projection='polar')

        # Select top 3 models for radar chart
        top_3_models = sorted(models, key=lambda x: model_data[x]['model_accuracy'], reverse=True)[:3]

        categories = ['Accuracy***REMOVED***nImprovement', 'Speed***REMOVED***nImprovement', 'Cost***REMOVED***nEfficiency']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, model in enumerate(top_3_models):
            values = [
                max(0, model_data[model]['accuracy_difference']),  # Accuracy improvement
                min(model_data[model]['time_speedup_factor'] / 25, 1),  # Speed (normalized)
                min(model_data[model]['cost_efficiency_ratio'] / 100, 1) if model_data[model]['cost_efficiency_ratio'] != float('inf') else 1  # Cost (normalized)
            ]
            values += values[:1]  # Complete the circle

            ax5.plot(angles, values, 'o-', linewidth=2, label=model.replace('-', '***REMOVED***n'), color=colors_radar[i])
            ax5.fill(angles, values, alpha=0.25, color=colors_radar[i])

        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories, fontsize=10)
        ax5.set_ylim(0, 1)
        ax5.set_title('Top 3 Models Performance Radar', fontsize=14, fontweight='bold', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 6. Summary Statistics Table (bottom)
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')

        # Create summary table
        table_data = []
        headers = ['Model', 'Accuracy', 'vs Human', 'Speed', 'Acc Sig', 'Time Sig', 'Recommendation']

        for model in models:
            data = model_data[model]
            acc_sig = '✓' if data['statistical_tests']['accuracy_chi2']['significant'] else '✗'
            time_sig = '✓' if data['statistical_tests']['time_mannwhitney'] and data['statistical_tests']['time_mannwhitney']['significant'] else '✗'

            # Generate recommendation
            if data['model_accuracy'] > human_acc and data['statistical_tests']['accuracy_chi2']['significant']:
                if data['time_speedup_factor'] > 10:
                    rec = "Excellent"
                elif data['time_speedup_factor'] > 2:
                    rec = "Good"
                else:
                    rec = "Accurate"
            elif data['model_accuracy'] > human_acc:
                rec = "Promising"
            else:
                rec = "Avoid"

            table_data.append([
                model.replace('-', '***REMOVED***n'),
                f"{data['model_accuracy']:.3f}",
                f"{data['accuracy_difference']:+.3f}",
                f"{data['time_speedup_factor']:.1f}x",
                acc_sig,
                time_sig,
                rec
            ])

        # Sort by accuracy for table
        table_data.sort(key=lambda x: float(x[1]), reverse=True)

        table = ax6.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color code the recommendation column
        for i in range(len(table_data)):
            rec = table_data[i][6]
            if rec == "Excellent":
                table[(i+1, 6)].set_facecolor('#90EE90')
            elif rec == "Good":
                table[(i+1, 6)].set_facecolor('#FFD700')
            elif rec == "Avoid":
                table[(i+1, 6)].set_facecolor('#FFB6C1')

        ax6.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)

    def _plot_complexity_performance_matrix(self):
        """Create detailed complexity performance matrix visualization"""
        if 'model_complexity_analysis' not in self.comparison_results:
            print("No model complexity data available for matrix plot")
            return

        complexity_data = self.comparison_results['model_complexity_analysis']
        models = list(complexity_data.keys())
        complexities = ['easy', 'medium', 'hard']

        if not models:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Task Complexity Performance Analysis', fontsize=18, fontweight='bold')

        # 1. Accuracy Heatmap
        accuracy_matrix = []
        for model in models:
            row = []
            for complexity in complexities:
                if complexity in complexity_data[model]:
                    row.append(complexity_data[model][complexity]['model_accuracy'])
                else:
                    row.append(0)
            accuracy_matrix.append(row)

        accuracy_array = np.array(accuracy_matrix)
        im1 = axes[0, 0].imshow(accuracy_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0, 0].set_title('Model Accuracy by Complexity', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(range(len(complexities)))
        axes[0, 0].set_xticklabels([c.capitalize() for c in complexities])
        axes[0, 0].set_yticks(range(len(models)))
        axes[0, 0].set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=10)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(complexities)):
                if accuracy_array[i, j] > 0:
                    text = f'{accuracy_array[i, j]:.3f}'
                    color = 'white' if accuracy_array[i, j] < 0.5 else 'black'
                    axes[0, 0].text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 2. Human vs Model Comparison
        human_accs = []
        for complexity in complexities:
            if models and complexity in complexity_data[models[0]]:
                human_accs.append(complexity_data[models[0]][complexity]['human_accuracy'])
            else:
                human_accs.append(0)

        x = np.arange(len(complexities))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            model_accs = []
            for complexity in complexities:
                if complexity in complexity_data[model]:
                    model_accs.append(complexity_data[model][complexity]['model_accuracy'])
                else:
                    model_accs.append(0)

            axes[0, 1].bar(x + i * width, model_accs, width, label=model.replace('-', '***REMOVED***n'), alpha=0.8)

        # Add human baseline
        axes[0, 1].plot(x + width * (len(models)-1)/2, human_accs, 'ro-', linewidth=3, markersize=8, label='Human Baseline')

        axes[0, 1].set_title('Accuracy Comparison by Complexity', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Task Complexity')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xticks(x + width * (len(models)-1)/2)
        axes[0, 1].set_xticklabels([c.capitalize() for c in complexities])
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Improvement over Human
        improvement_matrix = []
        for model in models:
            row = []
            for complexity in complexities:
                if complexity in complexity_data[model]:
                    row.append(complexity_data[model][complexity]['accuracy_difference'])
                else:
                    row.append(0)
            improvement_matrix.append(row)

        improvement_array = np.array(improvement_matrix)
        im2 = axes[0, 2].imshow(improvement_array, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
        axes[0, 2].set_title('Accuracy Improvement over Humans', fontsize=14, fontweight='bold')
        axes[0, 2].set_xticks(range(len(complexities)))
        axes[0, 2].set_xticklabels([c.capitalize() for c in complexities])
        axes[0, 2].set_yticks(range(len(models)))
        axes[0, 2].set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=10)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(complexities)):
                if improvement_array[i, j] != 0:
                    text = f'{improvement_array[i, j]:+.3f}'
                    color = 'white' if abs(improvement_array[i, j]) > 0.25 else 'black'
                    axes[0, 2].text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # 4. Speed Performance by Complexity
        speed_matrix = []
        for model in models:
            row = []
            for complexity in complexities:
                if complexity in complexity_data[model]:
                    speed = complexity_data[model][complexity]['time_speedup_factor']
                    row.append(min(speed, 25) if speed != float('inf') else 25)
                else:
                    row.append(0)
            speed_matrix.append(row)

        speed_array = np.array(speed_matrix)
        im3 = axes[1, 0].imshow(speed_array, cmap='YlOrRd', aspect='auto', vmin=0, vmax=25)
        axes[1, 0].set_title('Speed Improvement by Complexity', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(len(complexities)))
        axes[1, 0].set_xticklabels([c.capitalize() for c in complexities])
        axes[1, 0].set_yticks(range(len(models)))
        axes[1, 0].set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=10)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(complexities)):
                if speed_array[i, j] > 0:
                    text = f'{speed_array[i, j]:.1f}x'
                    color = 'white' if speed_array[i, j] > 12 else 'black'
                    axes[1, 0].text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 5. Best Model by Complexity
        best_models_by_complexity = {}
        for complexity in complexities:
            best_acc = 0
            best_model = None
            for model in models:
                if complexity in complexity_data[model]:
                    acc = complexity_data[model][complexity]['model_accuracy']
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model
            best_models_by_complexity[complexity] = (best_model, best_acc)

        # Create grouped bar chart
        best_accs = [best_models_by_complexity[c][1] for c in complexities]
        best_names = [best_models_by_complexity[c][0].replace('-', '***REMOVED***n') if best_models_by_complexity[c][0] else 'None' for c in complexities]

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = axes[1, 1].bar(complexities, best_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[1, 1].set_title('Best Model by Complexity Level', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Best Accuracy')
        axes[1, 1].set_ylim(0, 1)

        # Add model names and accuracy values
        for i, (bar, name, acc) in enumerate(zip(bars, best_names, best_accs)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                           name, ha='center', va='bottom', fontweight='bold', fontsize=10)
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, acc/2,
                           f'{acc:.3f}', ha='center', va='center', fontweight='bold', color='white', fontsize=12)

        # 6. Complexity Challenge Analysis
        # Show which models struggle with which complexity levels
        struggle_data = []
        for model in models:
            for complexity in complexities:
                if complexity in complexity_data[model]:
                    human_acc = complexity_data[model][complexity]['human_accuracy']
                    model_acc = complexity_data[model][complexity]['model_accuracy']
                    if model_acc < human_acc:
                        struggle_data.append((model, complexity, model_acc - human_acc))

        if struggle_data:
            struggle_models = [item[0].replace('-', '***REMOVED***n') for item in struggle_data]
            struggle_complexities = [item[1].capitalize() for item in struggle_data]
            struggle_diffs = [item[2] for item in struggle_data]

            scatter = axes[1, 2].scatter(range(len(struggle_data)), struggle_diffs,
                                       s=200, c='red', alpha=0.7, edgecolors='black', linewidth=2)

            axes[1, 2].set_title('Models Struggling vs Humans', fontsize=14, fontweight='bold')
            axes[1, 2].set_ylabel('Accuracy Difference (Model - Human)')
            axes[1, 2].set_xlabel('Model-Complexity Combinations')
            axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 2].grid(True, alpha=0.3)

            # Add labels
            for i, (model, complexity, diff) in enumerate(struggle_data):
                model_name = model.replace("-", "***REMOVED***n")
                axes[1, 2].annotate(f'{model_name}***REMOVED***n({complexity})',
                                   (i, diff), xytext=(0, -20), textcoords='offset points',
                                   ha='center', fontsize=8, fontweight='bold')
        else:
            axes[1, 2].text(0.5, 0.5, 'No models perform***REMOVED***nworse than humans***REMOVED***non any complexity level',
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=14, fontweight='bold', color='green')
            axes[1, 2].set_title('Models Struggling vs Humans', fontsize=14, fontweight='bold')

        plt.tight_layout()

    def _plot_quality_robustness_analysis(self):
        """Create comprehensive data quality robustness analysis"""
        if 'model_quality_analysis' not in self.comparison_results:
            print("No model quality data available for robustness analysis")
            return

        quality_data = self.comparison_results['model_quality_analysis']
        models = list(quality_data.keys())
        qualities = ['Q0', 'Q1', 'Q2', 'Q3']

        if not models:
            return

        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Data Quality Robustness Analysis', fontsize=18, fontweight='bold')

        # 1. Quality Performance Heatmap
        accuracy_matrix = []
        for model in models:
            row = []
            for quality in qualities:
                if quality in quality_data[model]:
                    row.append(quality_data[model][quality]['model_accuracy'])
                else:
                    row.append(0)
            accuracy_matrix.append(row)

        accuracy_array = np.array(accuracy_matrix)
        im1 = axes[0, 0].imshow(accuracy_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0, 0].set_title('Accuracy by Data Quality', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(range(len(qualities)))
        axes[0, 0].set_xticklabels(['Perfect', 'Spaces', 'Missing***REMOVED***nChars', 'Missing***REMOVED***nRecords'])
        axes[0, 0].set_yticks(range(len(models)))
        axes[0, 0].set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=10)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(qualities)):
                if accuracy_array[i, j] > 0:
                    text = f'{accuracy_array[i, j]:.3f}'
                    color = 'white' if accuracy_array[i, j] < 0.5 else 'black'
                    axes[0, 0].text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 2. Robustness Score Ranking
        robustness_scores = []
        for model in models:
            if 'Q0' in quality_data[model]:
                q0_acc = quality_data[model]['Q0']['model_accuracy']
                corrupted_accs = []
                for q in ['Q1', 'Q2', 'Q3']:
                    if q in quality_data[model]:
                        corrupted_accs.append(quality_data[model][q]['model_accuracy'])

                if corrupted_accs:
                    avg_corrupted = np.mean(corrupted_accs)
                    robustness = avg_corrupted / q0_acc if q0_acc > 0 else 0
                    robustness_scores.append((model, robustness, q0_acc, avg_corrupted))

        # Sort by robustness score
        robustness_scores.sort(key=lambda x: x[1], reverse=True)

        if robustness_scores:
            rob_models = [item[0].replace('-', '***REMOVED***n') for item in robustness_scores]
            rob_scores = [item[1] for item in robustness_scores]

            colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' for score in rob_scores]
            bars = axes[0, 1].barh(range(len(rob_models)), rob_scores, color=colors, alpha=0.7, edgecolor='black')

            axes[0, 1].set_yticks(range(len(rob_models)))
            axes[0, 1].set_yticklabels(rob_models, fontsize=10)
            axes[0, 1].set_xlabel('Robustness Score (Corrupted/Perfect)')
            axes[0, 1].set_title('Data Quality Robustness Ranking', fontsize=14, fontweight='bold')
            axes[0, 1].axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Good (>0.8)')
            axes[0, 1].axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate (>0.6)')
            axes[0, 1].legend()

            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, rob_scores)):
                axes[0, 1].text(score + 0.02, bar.get_y() + bar.get_height()/2,
                               f'{score:.3f}', va='center', fontweight='bold')

        # 3. Quality Degradation Patterns
        quality_labels = ['Perfect***REMOVED***n(Q0)', 'Spaces***REMOVED***n(Q1)', 'Missing Chars***REMOVED***n(Q2)', 'Missing Records***REMOVED***n(Q3)']

        for i, model in enumerate(models[:4]):  # Show top 4 models
            model_accs = []
            for quality in qualities:
                if quality in quality_data[model]:
                    model_accs.append(quality_data[model][quality]['model_accuracy'])
                else:
                    model_accs.append(0)

            axes[0, 2].plot(range(len(qualities)), model_accs, marker='o', linewidth=2,
                           label=model.replace('-', '***REMOVED***n'), alpha=0.8, markersize=8)

        axes[0, 2].set_title('Accuracy Degradation Patterns', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_xticks(range(len(qualities)))
        axes[0, 2].set_xticklabels(quality_labels, fontsize=10)
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)

        # 4. Perfect vs Corrupted Data Scatter
        perfect_accs = []
        corrupted_avgs = []
        model_names_scatter = []

        for model in models:
            if 'Q0' in quality_data[model]:
                q0_acc = quality_data[model]['Q0']['model_accuracy']
                corrupted_accs = []
                for q in ['Q1', 'Q2', 'Q3']:
                    if q in quality_data[model]:
                        corrupted_accs.append(quality_data[model][q]['model_accuracy'])

                if corrupted_accs:
                    avg_corrupted = np.mean(corrupted_accs)
                    perfect_accs.append(q0_acc)
                    corrupted_avgs.append(avg_corrupted)
                    model_names_scatter.append(model)

        if perfect_accs:
            scatter = axes[1, 0].scatter(perfect_accs, corrupted_avgs, s=200, alpha=0.7,
                                       c=range(len(perfect_accs)), cmap='viridis', edgecolors='black', linewidth=2)

            # Add diagonal line (perfect robustness)
            max_val = max(max(perfect_accs), max(corrupted_avgs))
            axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.7, linewidth=2, label='Perfect Robustness')

            # Add model labels
            for i, model in enumerate(model_names_scatter):
                axes[1, 0].annotate(model.replace('-', '***REMOVED***n'), (perfect_accs[i], corrupted_avgs[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

            axes[1, 0].set_xlabel('Perfect Data Accuracy (Q0)')
            axes[1, 0].set_ylabel('Average Corrupted Data Accuracy (Q1-Q3)')
            axes[1, 0].set_title('Perfect vs Corrupted Performance', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Quality-specific Best Models
        best_by_quality = {}
        for quality in qualities:
            best_acc = 0
            best_model = None
            for model in models:
                if quality in quality_data[model]:
                    acc = quality_data[model][quality]['model_accuracy']
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model
            best_by_quality[quality] = (best_model, best_acc)

        quality_names = ['Perfect', 'Spaces', 'Missing***REMOVED***nChars', 'Missing***REMOVED***nRecords']
        best_accs = [best_by_quality[q][1] for q in qualities]
        best_names = [best_by_quality[q][0].replace('-', '***REMOVED***n') if best_by_quality[q][0] else 'None' for q in qualities]

        colors = ['#2E8B57', '#FF6347', '#4169E1', '#FFD700']
        bars = axes[1, 1].bar(quality_names, best_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[1, 1].set_title('Best Model by Quality Condition', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Best Accuracy')
        axes[1, 1].set_ylim(0, 1)

        # Add model names and values
        for i, (bar, name, acc) in enumerate(zip(bars, best_names, best_accs)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                           name, ha='center', va='bottom', fontweight='bold', fontsize=10)
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, acc/2,
                           f'{acc:.3f}', ha='center', va='center', fontweight='bold', color='white', fontsize=12)

        # 6. Human vs Model by Quality
        human_accs_quality = []
        for quality in qualities:
            if models and quality in quality_data[models[0]]:
                human_accs_quality.append(quality_data[models[0]][quality]['human_accuracy'])
            else:
                human_accs_quality.append(0)

        x = np.arange(len(qualities))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            model_accs_quality = []
            for quality in qualities:
                if quality in quality_data[model]:
                    model_accs_quality.append(quality_data[model][quality]['model_accuracy'])
                else:
                    model_accs_quality.append(0)

            axes[1, 2].bar(x + i * width, model_accs_quality, width,
                          label=model.replace('-', '***REMOVED***n'), alpha=0.8)

        # Add human baseline
        axes[1, 2].plot(x + width * (len(models)-1)/2, human_accs_quality, 'ro-',
                       linewidth=3, markersize=8, label='Human Baseline')

        axes[1, 2].set_title('Human vs Models by Quality', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Data Quality Condition')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].set_xticks(x + width * (len(models)-1)/2)
        axes[1, 2].set_xticklabels(quality_names)
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 2].grid(True, alpha=0.3)

        # 7. Quality Impact Analysis (bottom row)
        # Show the impact of each quality condition on model performance
        quality_impacts = {}
        for model in models:
            if 'Q0' in quality_data[model]:
                q0_acc = quality_data[model]['Q0']['model_accuracy']
                impacts = []
                for q in ['Q1', 'Q2', 'Q3']:
                    if q in quality_data[model]:
                        impact = quality_data[model][q]['model_accuracy'] - q0_acc
                        impacts.append(impact)
                    else:
                        impacts.append(0)
                quality_impacts[model] = impacts

        if quality_impacts:
            impact_qualities = ['Spaces***REMOVED***nImpact', 'Missing Chars***REMOVED***nImpact', 'Missing Records***REMOVED***nImpact']

            # Create grouped bar chart for impacts
            x_impact = np.arange(len(impact_qualities))
            width_impact = 0.8 / len(models)

            for i, model in enumerate(models):
                if model in quality_impacts:
                    impacts = quality_impacts[model]
                    colors_impact = ['red' if imp < 0 else 'green' for imp in impacts]
                    axes[2, 0].bar(x_impact + i * width_impact, impacts, width_impact,
                                  label=model.replace('-', '***REMOVED***n'), alpha=0.8)

            axes[2, 0].set_title('Quality Condition Impact on Performance', fontsize=14, fontweight='bold')
            axes[2, 0].set_xlabel('Quality Condition')
            axes[2, 0].set_ylabel('Accuracy Change from Perfect Data')
            axes[2, 0].set_xticks(x_impact + width_impact * (len(models)-1)/2)
            axes[2, 0].set_xticklabels(impact_qualities)
            axes[2, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[2, 0].grid(True, alpha=0.3)

        # 8. Robustness vs Accuracy Trade-off
        if robustness_scores:
            rob_scores_plot = [item[1] for item in robustness_scores]
            perfect_accs_plot = [item[2] for item in robustness_scores]
            model_names_plot = [item[0] for item in robustness_scores]

            scatter = axes[2, 1].scatter(perfect_accs_plot, rob_scores_plot, s=200, alpha=0.7,
                                       c=range(len(perfect_accs_plot)), cmap='plasma', edgecolors='black', linewidth=2)

            # Add model labels
            for i, model in enumerate(model_names_plot):
                axes[2, 1].annotate(model.replace('-', '***REMOVED***n'), (perfect_accs_plot[i], rob_scores_plot[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

            axes[2, 1].set_xlabel('Perfect Data Accuracy')
            axes[2, 1].set_ylabel('Robustness Score')
            axes[2, 1].set_title('Accuracy vs Robustness Trade-off', fontsize=14, fontweight='bold')
            axes[2, 1].grid(True, alpha=0.3)

            # Add quadrant lines
            axes[2, 1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High Robustness')
            axes[2, 1].axvline(x=0.9, color='blue', linestyle='--', alpha=0.5, label='High Accuracy')
            axes[2, 1].legend()

        # 9. Quality Condition Summary
        axes[2, 2].axis('off')

        # Create summary text
        summary_text = "DATA QUALITY INSIGHTS:***REMOVED***n***REMOVED***n"

        if robustness_scores:
            most_robust = max(robustness_scores, key=lambda x: x[1])
            least_robust = min(robustness_scores, key=lambda x: x[1])

            summary_text += f"🛡️ MOST ROBUST:***REMOVED***n{most_robust[0]}***REMOVED***nScore: {most_robust[1]:.3f}***REMOVED***n***REMOVED***n"
            summary_text += f"⚠️ LEAST ROBUST:***REMOVED***n{least_robust[0]}***REMOVED***nScore: {least_robust[1]:.3f}***REMOVED***n***REMOVED***n"

        # Best performers by condition
        summary_text += "🏆 BEST BY CONDITION:***REMOVED***n"
        condition_names = {'Q0': 'Perfect', 'Q1': 'Spaces', 'Q2': 'Missing Chars', 'Q3': 'Missing Records'}
        for quality in qualities:
            if quality in best_by_quality:
                model, acc = best_by_quality[quality]
                summary_text += f"{condition_names[quality]}: {model}***REMOVED***n({acc:.3f})***REMOVED***n"

        summary_text += f"***REMOVED***n💡 RECOMMENDATION:***REMOVED***nFor robust performance across***REMOVED***nall data quality conditions,***REMOVED***nuse: {most_robust[0] if robustness_scores else 'N/A'}"

        axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes,
                       fontsize=12, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

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
