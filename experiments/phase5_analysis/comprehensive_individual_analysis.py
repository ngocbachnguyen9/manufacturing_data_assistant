#!/usr/bin/env python3
"""
Comprehensive Individual Model Analysis
=====================================

This script generates a detailed statistical analysis of each LLM model's performance
against humans and other models, including confidence intervals, effect sizes,
and practical recommendations.

Author: Manufacturing Data Assistant
Date: 2025-06-27
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveModelAnalyzer:
    """Comprehensive statistical analysis of individual model performance"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.analysis_results = {}
        self.confidence_level = 0.95
        self.alpha = 1 - self.confidence_level
        
    def load_data(self):
        """Load all analysis data files"""
        print("Loading analysis data...")
        
        # Load CSV files
        csv_files = {
            'model_specific': 'model_specific_comparison.csv',
            'model_complexity': 'model_complexity_analysis.csv',
            'model_quality': 'model_quality_analysis.csv',
            'task_level': 'task_level_comparison.csv',
            'overall': 'overall_comparison.csv',
            'statistical_tests': 'statistical_tests.csv'
        }
        
        for key, filename in csv_files.items():
            file_path = self.results_dir / filename
            if file_path.exists():
                self.data[key] = pd.read_csv(file_path)
                print(f"✓ Loaded {filename}")
            else:
                print(f"✗ Missing {filename}")
        
        # Load JSON results
        json_file = self.results_dir / 'phase5_results.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                self.data['json_results'] = json.load(f)
            print("✓ Loaded phase5_results.json")
        
        print(f"Data loading complete. Loaded {len(self.data)} datasets.")
    
    def calculate_confidence_intervals(self, data: np.array, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence intervals using bootstrap method"""
        if len(data) == 0:
            return (np.nan, np.nan)
        
        # Bootstrap confidence interval
        def bootstrap_mean(x):
            return np.mean(x)
        
        try:
            # Use scipy bootstrap for robust CI calculation
            rng = np.random.default_rng(42)  # For reproducibility
            res = bootstrap((data,), bootstrap_mean, n_resamples=1000, 
                          confidence_level=confidence_level, random_state=rng)
            return (res.confidence_interval.low, res.confidence_interval.high)
        except:
            # Fallback to t-distribution CI
            mean = np.mean(data)
            sem = stats.sem(data)
            h = sem * stats.t.ppf((1 + confidence_level) / 2., len(data)-1)
            return (mean - h, mean + h)
    
    def calculate_effect_size(self, group1: np.array, group2: np.array) -> Dict[str, float]:
        """Calculate Cohen's d effect size and other effect size measures"""
        if len(group1) == 0 or len(group2) == 0:
            return {'cohens_d': np.nan, 'glass_delta': np.nan, 'hedges_g': np.nan}
        
        # Cohen's d
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else np.nan
        
        # Glass's Delta (using control group std)
        glass_delta = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1) if np.std(group2, ddof=1) > 0 else np.nan
        
        # Hedges' g (bias-corrected Cohen's d)
        j = 1 - (3 / (4 * (len(group1) + len(group2)) - 9))
        hedges_g = cohens_d * j
        
        return {
            'cohens_d': cohens_d,
            'glass_delta': glass_delta,
            'hedges_g': hedges_g
        }
    
    def interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude"""
        abs_effect = abs(effect_size)
        if np.isnan(abs_effect):
            return "Cannot determine"
        elif abs_effect < 0.2:
            return "Negligible"
        elif abs_effect < 0.5:
            return "Small"
        elif abs_effect < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def analyze_individual_model(self, model_name: str) -> Dict[str, Any]:
        """Comprehensive analysis of a single model"""
        print(f"***REMOVED***nAnalyzing {model_name}...")
        
        # Get model-specific data
        model_specific = self.data['model_specific'][self.data['model_specific']['model'] == model_name].iloc[0]
        model_complexity = self.data['model_complexity'][self.data['model_complexity']['model'] == model_name]
        model_quality = self.data['model_quality'][self.data['model_quality']['model'] == model_name]
        
        analysis = {
            'model_name': model_name,
            'overall_performance': {},
            'complexity_analysis': {},
            'quality_analysis': {},
            'statistical_significance': {},
            'effect_sizes': {},
            'confidence_intervals': {},
            'strengths_weaknesses': {},
            'recommendations': {}
        }
        
        # Overall Performance Analysis
        analysis['overall_performance'] = {
            'accuracy': {
                'model': float(model_specific['model_accuracy']),
                'human': float(model_specific['human_accuracy']),
                'difference': float(model_specific['accuracy_difference']),
                'improvement_percentage': float(model_specific['accuracy_difference']) / float(model_specific['human_accuracy']) * 100
            },
            'speed': {
                'model_time': float(model_specific['model_avg_time']),
                'human_time': float(model_specific['human_avg_time']),
                'speedup_factor': float(model_specific['time_speedup_factor']),
                'time_saved_percentage': (1 - 1/float(model_specific['time_speedup_factor'])) * 100
            },
            'cost': {
                'model_cost': float(model_specific['model_avg_cost']),
                'human_cost': float(model_specific['human_avg_cost']),
                'cost_efficiency': float(model_specific['cost_efficiency_ratio'])
            }
        }
        
        # Statistical Significance Analysis
        json_model_data = self.data['json_results']['model_specific_comparison'].get(model_name, {})
        if 'statistical_tests' in json_model_data:
            stat_tests = json_model_data['statistical_tests']
            analysis['statistical_significance'] = {
                'accuracy_test': {
                    'chi2_statistic': stat_tests.get('accuracy_chi2', {}).get('statistic', np.nan),
                    'p_value': stat_tests.get('accuracy_chi2', {}).get('p_value', np.nan),
                    'significant': stat_tests.get('accuracy_chi2', {}).get('significant', False),
                    'interpretation': 'Significantly different from human performance' if stat_tests.get('accuracy_chi2', {}).get('significant', False) else 'Not significantly different from human performance'
                },
                'time_test': {
                    'mw_statistic': stat_tests.get('time_mannwhitney', {}).get('statistic', np.nan) if stat_tests.get('time_mannwhitney') else np.nan,
                    'p_value': stat_tests.get('time_mannwhitney', {}).get('p_value', np.nan) if stat_tests.get('time_mannwhitney') else np.nan,
                    'significant': stat_tests.get('time_mannwhitney', {}).get('significant', False) if stat_tests.get('time_mannwhitney') else False,
                    'interpretation': 'Significantly different completion time' if (stat_tests.get('time_mannwhitney') and stat_tests.get('time_mannwhitney', {}).get('significant', False)) else 'No significant difference in completion time'
                }
            }

        # Complexity Analysis
        complexity_results = {}
        for _, row in model_complexity.iterrows():
            complexity = row['complexity']
            complexity_results[complexity] = {
                'model_accuracy': float(row['model_accuracy']),
                'human_accuracy': float(row['human_accuracy']),
                'accuracy_difference': float(row['accuracy_difference']),
                'model_time': float(row['model_avg_time']),
                'human_time': float(row['human_avg_time']),
                'speedup_factor': float(row['time_speedup_factor']),
                'relative_performance': 'Better' if float(row['accuracy_difference']) > 0 else 'Worse' if float(row['accuracy_difference']) < 0 else 'Equal'
            }
        analysis['complexity_analysis'] = complexity_results

        # Quality Analysis
        quality_results = {}
        for _, row in model_quality.iterrows():
            quality = row['quality_condition']
            quality_results[quality] = {
                'model_accuracy': float(row['model_accuracy']),
                'human_accuracy': float(row['human_accuracy']),
                'accuracy_difference': float(row['accuracy_difference']),
                'model_time': float(row['model_avg_time']),
                'human_time': float(row['human_avg_time']),
                'speedup_factor': float(row['time_speedup_factor']),
                'robustness_score': float(row['model_accuracy'])  # Model's ability to maintain performance
            }
        analysis['quality_analysis'] = quality_results

        # Calculate robustness metrics
        if quality_results:
            q0_accuracy = quality_results.get('Q0', {}).get('model_accuracy', 0)
            corrupted_accuracies = [quality_results[q]['model_accuracy'] for q in ['Q1', 'Q2', 'Q3'] if q in quality_results]
            if corrupted_accuracies:
                avg_corrupted_accuracy = np.mean(corrupted_accuracies)
                robustness_score = avg_corrupted_accuracy / q0_accuracy if q0_accuracy > 0 else 0
                analysis['quality_analysis']['robustness_metrics'] = {
                    'baseline_accuracy': q0_accuracy,
                    'avg_corrupted_accuracy': avg_corrupted_accuracy,
                    'robustness_score': robustness_score,
                    'robustness_category': 'Robust' if robustness_score > 0.8 else 'Moderate' if robustness_score > 0.6 else 'Fragile'
                }

        return analysis

    def analyze_model_vs_model_comparisons(self) -> Dict[str, Any]:
        """Analyze head-to-head comparisons between all models"""
        models = self.get_models_list()
        comparisons = {}

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:  # Avoid duplicate comparisons
                    comparison_key = f"{model1}_vs_{model2}"

                    # Get model data
                    model1_data = self.data['model_specific'][self.data['model_specific']['model'] == model1].iloc[0]
                    model2_data = self.data['model_specific'][self.data['model_specific']['model'] == model2].iloc[0]

                    # Calculate differences
                    accuracy_diff = float(model1_data['model_accuracy']) - float(model2_data['model_accuracy'])
                    speed_ratio = float(model2_data['model_avg_time']) / float(model1_data['model_avg_time'])

                    comparisons[comparison_key] = {
                        'model1': model1,
                        'model2': model2,
                        'accuracy_difference': accuracy_diff,
                        'accuracy_winner': model1 if accuracy_diff > 0 else model2 if accuracy_diff < 0 else 'Tie',
                        'speed_ratio': speed_ratio,
                        'speed_winner': model1 if speed_ratio > 1 else model2 if speed_ratio < 1 else 'Tie',
                        'overall_winner': self._determine_overall_winner(model1_data, model2_data)
                    }

        return comparisons

    def _determine_overall_winner(self, model1_data: pd.Series, model2_data: pd.Series) -> str:
        """Determine overall winner based on weighted criteria"""
        # Weights: accuracy (0.6), speed (0.3), statistical significance (0.1)
        model1_score = 0
        model2_score = 0

        # Accuracy comparison
        if float(model1_data['model_accuracy']) > float(model2_data['model_accuracy']):
            model1_score += 0.6
        elif float(model1_data['model_accuracy']) < float(model2_data['model_accuracy']):
            model2_score += 0.6

        # Speed comparison (higher speedup factor is better)
        if float(model1_data['time_speedup_factor']) > float(model2_data['time_speedup_factor']):
            model1_score += 0.3
        elif float(model1_data['time_speedup_factor']) < float(model2_data['time_speedup_factor']):
            model2_score += 0.3

        # Statistical significance bonus
        if model1_data['accuracy_chi2_significant']:
            model1_score += 0.1
        if model2_data['accuracy_chi2_significant']:
            model2_score += 0.1

        if model1_score > model2_score:
            return model1_data['model']
        elif model2_score > model1_score:
            return model2_data['model']
        else:
            return 'Tie'

    def get_models_list(self) -> List[str]:
        """Get list of all models in the analysis"""
        if 'model_specific' in self.data:
            return self.data['model_specific']['model'].tolist()
        return []

    def generate_model_recommendations(self, model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific recommendations for a model"""
        model_name = model_analysis['model_name']
        overall_perf = model_analysis['overall_performance']
        complexity_analysis = model_analysis['complexity_analysis']
        quality_analysis = model_analysis['quality_analysis']
        stat_sig = model_analysis['statistical_significance']

        recommendations = {
            'deployment_readiness': 'Not Ready',
            'best_use_cases': [],
            'avoid_use_cases': [],
            'strengths': [],
            'weaknesses': [],
            'risk_assessment': 'High',
            'confidence_level': 'Low'
        }

        # Assess deployment readiness
        accuracy_better = overall_perf['accuracy']['difference'] > 0
        statistically_significant = stat_sig.get('accuracy_test', {}).get('significant', False)
        speed_improvement = overall_perf['speed']['speedup_factor'] > 1

        if accuracy_better and statistically_significant and speed_improvement:
            recommendations['deployment_readiness'] = 'Ready'
            recommendations['confidence_level'] = 'High'
            recommendations['risk_assessment'] = 'Low'
        elif accuracy_better and statistically_significant:
            recommendations['deployment_readiness'] = 'Ready with Caution'
            recommendations['confidence_level'] = 'Medium'
            recommendations['risk_assessment'] = 'Medium'
        elif accuracy_better:
            recommendations['deployment_readiness'] = 'Pilot Testing Recommended'
            recommendations['confidence_level'] = 'Low'
            recommendations['risk_assessment'] = 'Medium'

        # Identify strengths
        if overall_perf['accuracy']['difference'] > 0.1:
            recommendations['strengths'].append('Significantly higher accuracy than humans')
        if overall_perf['speed']['speedup_factor'] > 5:
            recommendations['strengths'].append('Substantial speed improvement')
        if overall_perf['speed']['speedup_factor'] > 10:
            recommendations['strengths'].append('Exceptional processing speed')

        # Analyze complexity performance
        complexity_strengths = []
        complexity_weaknesses = []
        for complexity, data in complexity_analysis.items():
            if data['accuracy_difference'] > 0.1:
                complexity_strengths.append(f"Excellent performance on {complexity} tasks")
            elif data['accuracy_difference'] < -0.1:
                complexity_weaknesses.append(f"Poor performance on {complexity} tasks")

        recommendations['strengths'].extend(complexity_strengths)
        recommendations['weaknesses'].extend(complexity_weaknesses)

        # Analyze quality robustness
        if 'robustness_metrics' in quality_analysis:
            robustness = quality_analysis['robustness_metrics']
            if robustness['robustness_score'] > 0.8:
                recommendations['strengths'].append('Robust to data quality issues')
            elif robustness['robustness_score'] < 0.6:
                recommendations['weaknesses'].append('Sensitive to data quality degradation')

        # Use case recommendations
        if accuracy_better and speed_improvement:
            recommendations['best_use_cases'].extend([
                'High-volume production environments',
                'Real-time quality control',
                'Automated decision making'
            ])

        if overall_perf['accuracy']['difference'] < 0:
            recommendations['avoid_use_cases'].extend([
                'Critical safety applications',
                'High-stakes decision making',
                'Regulatory compliance tasks'
            ])

        return recommendations

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive markdown report"""
        models = self.get_models_list()

        # Analyze all models
        all_analyses = {}
        for model in models:
            all_analyses[model] = self.analyze_individual_model(model)
            all_analyses[model]['recommendations'] = self.generate_model_recommendations(all_analyses[model])

        # Generate model-vs-model comparisons
        model_comparisons = self.analyze_model_vs_model_comparisons()

        # Start building the report
        report_lines = []
        report_lines.append("# Comprehensive Individual Model Analysis Report")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Confidence Level:** {self.confidence_level*100:.0f}%")
        report_lines.append("")

        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")

        # Overall rankings
        accuracy_ranking = sorted(models, key=lambda m: all_analyses[m]['overall_performance']['accuracy']['model'], reverse=True)
        speed_ranking = sorted(models, key=lambda m: all_analyses[m]['overall_performance']['speed']['speedup_factor'], reverse=True)

        report_lines.append("### Model Rankings")
        report_lines.append("")
        report_lines.append("**By Accuracy:**")
        for i, model in enumerate(accuracy_ranking, 1):
            acc = all_analyses[model]['overall_performance']['accuracy']['model']
            diff = all_analyses[model]['overall_performance']['accuracy']['difference']
            report_lines.append(f"{i}. **{model}**: {acc:.3f} ({diff:+.3f} vs human)")

        report_lines.append("")
        report_lines.append("**By Speed:**")
        for i, model in enumerate(speed_ranking, 1):
            speedup = all_analyses[model]['overall_performance']['speed']['speedup_factor']
            report_lines.append(f"{i}. **{model}**: {speedup:.1f}x faster")

        report_lines.append("")

        # Deployment Readiness Summary
        ready_models = [m for m in models if all_analyses[m]['recommendations']['deployment_readiness'] == 'Ready']
        caution_models = [m for m in models if all_analyses[m]['recommendations']['deployment_readiness'] == 'Ready with Caution']
        pilot_models = [m for m in models if all_analyses[m]['recommendations']['deployment_readiness'] == 'Pilot Testing Recommended']
        not_ready_models = [m for m in models if all_analyses[m]['recommendations']['deployment_readiness'] == 'Not Ready']

        report_lines.append("### Deployment Readiness Summary")
        report_lines.append("")
        report_lines.append(f"🟢 **Ready for Production ({len(ready_models)}):** {', '.join(ready_models) if ready_models else 'None'}")
        report_lines.append(f"🟡 **Ready with Caution ({len(caution_models)}):** {', '.join(caution_models) if caution_models else 'None'}")
        report_lines.append(f"🔵 **Pilot Testing Recommended ({len(pilot_models)}):** {', '.join(pilot_models) if pilot_models else 'None'}")
        report_lines.append(f"🔴 **Not Ready ({len(not_ready_models)}):** {', '.join(not_ready_models) if not_ready_models else 'None'}")
        report_lines.append("")

        # Individual Model Analysis
        report_lines.append("## Individual Model Analysis")
        report_lines.append("")

        for model in models:
            analysis = all_analyses[model]
            recommendations = analysis['recommendations']

            report_lines.append(f"### {model}")
            report_lines.append("")

            # Deployment status badge
            status = recommendations['deployment_readiness']
            if status == 'Ready':
                badge = "🟢 READY FOR PRODUCTION"
            elif status == 'Ready with Caution':
                badge = "🟡 READY WITH CAUTION"
            elif status == 'Pilot Testing Recommended':
                badge = "🔵 PILOT TESTING RECOMMENDED"
            else:
                badge = "🔴 NOT READY"

            report_lines.append(f"**Status:** {badge}")
            report_lines.append(f"**Risk Level:** {recommendations['risk_assessment']}")
            report_lines.append(f"**Confidence:** {recommendations['confidence_level']}")
            report_lines.append("")

            # Performance Metrics
            overall_perf = analysis['overall_performance']
            report_lines.append("#### Performance Metrics")
            report_lines.append("")
            report_lines.append("| Metric | Model | Human | Difference | Improvement |")
            report_lines.append("|--------|-------|-------|------------|-------------|")

            acc_model = overall_perf['accuracy']['model']
            acc_human = overall_perf['accuracy']['human']
            acc_diff = overall_perf['accuracy']['difference']
            acc_imp = overall_perf['accuracy']['improvement_percentage']

            speed_model = overall_perf['speed']['model_time']
            speed_human = overall_perf['speed']['human_time']
            speed_factor = overall_perf['speed']['speedup_factor']
            time_saved = overall_perf['speed']['time_saved_percentage']

            report_lines.append(f"| Accuracy | {acc_model:.3f} | {acc_human:.3f} | {acc_diff:+.3f} | {acc_imp:+.1f}% |")
            report_lines.append(f"| Avg Time (sec) | {speed_model:.1f} | {speed_human:.1f} | {speed_factor:.1f}x faster | {time_saved:.1f}% saved |")
            report_lines.append("")

            # Statistical Significance
            stat_sig = analysis['statistical_significance']
            report_lines.append("#### Statistical Significance")
            report_lines.append("")

            acc_test = stat_sig.get('accuracy_test', {})
            time_test = stat_sig.get('time_test', {})

            acc_sig = "✅ Significant" if acc_test.get('significant', False) else "❌ Not Significant"
            time_sig = "✅ Significant" if time_test.get('significant', False) else "❌ Not Significant"

            report_lines.append(f"- **Accuracy vs Human:** {acc_sig} (p = {acc_test.get('p_value', 'N/A'):.2e})")
            report_lines.append(f"- **Completion Time vs Human:** {time_sig}")
            if time_test.get('p_value') is not None and not np.isnan(time_test.get('p_value')):
                report_lines.append(f"  (p = {time_test.get('p_value'):.2e})")
            report_lines.append("")

            # Performance by Complexity
            complexity_analysis = analysis['complexity_analysis']
            if complexity_analysis:
                report_lines.append("#### Performance by Task Complexity")
                report_lines.append("")
                report_lines.append("| Complexity | Model Acc | Human Acc | Difference | Speed Factor | Performance |")
                report_lines.append("|------------|-----------|-----------|------------|--------------|-------------|")

                for complexity in ['easy', 'medium', 'hard']:
                    if complexity in complexity_analysis:
                        data = complexity_analysis[complexity]
                        model_acc = data['model_accuracy']
                        human_acc = data['human_accuracy']
                        diff = data['accuracy_difference']
                        speed = data['speedup_factor']
                        perf = data['relative_performance']

                        perf_emoji = "🟢" if perf == "Better" else "🔴" if perf == "Worse" else "🟡"
                        report_lines.append(f"| {complexity.title()} | {model_acc:.3f} | {human_acc:.3f} | {diff:+.3f} | {speed:.1f}x | {perf_emoji} {perf} |")

                report_lines.append("")

            # Performance by Data Quality
            quality_analysis = analysis['quality_analysis']
            if quality_analysis and 'robustness_metrics' in quality_analysis:
                robustness = quality_analysis['robustness_metrics']
                report_lines.append("#### Data Quality Robustness")
                report_lines.append("")
                report_lines.append(f"- **Baseline Performance (Q0):** {robustness['baseline_accuracy']:.3f}")
                report_lines.append(f"- **Average on Corrupted Data (Q1-Q3):** {robustness['avg_corrupted_accuracy']:.3f}")
                report_lines.append(f"- **Robustness Score:** {robustness['robustness_score']:.3f}")
                report_lines.append(f"- **Category:** {robustness['robustness_category']}")
                report_lines.append("")

                # Quality condition breakdown
                report_lines.append("| Quality Condition | Model Acc | Human Acc | Difference | Speed Factor |")
                report_lines.append("|-------------------|-----------|-----------|------------|--------------|")

                for quality in ['Q0', 'Q1', 'Q2', 'Q3']:
                    if quality in quality_analysis and quality != 'robustness_metrics':
                        data = quality_analysis[quality]
                        model_acc = data['model_accuracy']
                        human_acc = data['human_accuracy']
                        diff = data['accuracy_difference']
                        speed = data['speedup_factor']

                        quality_label = 'Normal Baseline' if quality == 'Q0' else f'Corrupted {quality}'
                        report_lines.append(f"| {quality_label} | {model_acc:.3f} | {human_acc:.3f} | {diff:+.3f} | {speed:.1f}x |")

                report_lines.append("")

            # Strengths and Weaknesses
            if recommendations['strengths'] or recommendations['weaknesses']:
                report_lines.append("#### Strengths and Weaknesses")
                report_lines.append("")

                if recommendations['strengths']:
                    report_lines.append("**Strengths:**")
                    for strength in recommendations['strengths']:
                        report_lines.append(f"- ✅ {strength}")
                    report_lines.append("")

                if recommendations['weaknesses']:
                    report_lines.append("**Weaknesses:**")
                    for weakness in recommendations['weaknesses']:
                        report_lines.append(f"- ❌ {weakness}")
                    report_lines.append("")

            # Use Case Recommendations
            if recommendations['best_use_cases'] or recommendations['avoid_use_cases']:
                report_lines.append("#### Use Case Recommendations")
                report_lines.append("")

                if recommendations['best_use_cases']:
                    report_lines.append("**Recommended Use Cases:**")
                    for use_case in recommendations['best_use_cases']:
                        report_lines.append(f"- 🎯 {use_case}")
                    report_lines.append("")

                if recommendations['avoid_use_cases']:
                    report_lines.append("**Avoid These Use Cases:**")
                    for use_case in recommendations['avoid_use_cases']:
                        report_lines.append(f"- ⚠️ {use_case}")
                    report_lines.append("")

            report_lines.append("---")
            report_lines.append("")

        # Head-to-Head Model Comparisons
        report_lines.append("## Head-to-Head Model Comparisons")
        report_lines.append("")
        report_lines.append("This section provides detailed pairwise comparisons between all models.")
        report_lines.append("")

        # Create comparison matrix
        report_lines.append("### Model Comparison Matrix")
        report_lines.append("")
        report_lines.append("| Model 1 | Model 2 | Accuracy Winner | Speed Winner | Overall Winner |")
        report_lines.append("|---------|---------|-----------------|--------------|----------------|")

        for comparison_key, comparison_data in model_comparisons.items():
            model1 = comparison_data['model1']
            model2 = comparison_data['model2']
            acc_winner = comparison_data['accuracy_winner']
            speed_winner = comparison_data['speed_winner']
            overall_winner = comparison_data['overall_winner']

            # Shorten model names for table
            model1_short = model1.split('-')[0] if '-' in model1 else model1
            model2_short = model2.split('-')[0] if '-' in model2 else model2
            acc_winner_short = acc_winner.split('-')[0] if '-' in acc_winner and acc_winner != 'Tie' else acc_winner
            speed_winner_short = speed_winner.split('-')[0] if '-' in speed_winner and speed_winner != 'Tie' else speed_winner
            overall_winner_short = overall_winner.split('-')[0] if '-' in overall_winner and overall_winner != 'Tie' else overall_winner

            report_lines.append(f"| {model1_short} | {model2_short} | {acc_winner_short} | {speed_winner_short} | **{overall_winner_short}** |")

        report_lines.append("")

        # Model Selection Framework
        report_lines.append("## Model Selection Framework")
        report_lines.append("")
        report_lines.append("### Decision Tree for Model Selection")
        report_lines.append("")

        # Find best models for different scenarios
        best_accuracy = max(models, key=lambda m: all_analyses[m]['overall_performance']['accuracy']['model'])
        best_speed = max(models, key=lambda m: all_analyses[m]['overall_performance']['speed']['speedup_factor'])
        best_overall = max(models, key=lambda m: all_analyses[m]['overall_performance']['accuracy']['model'] *
                          min(all_analyses[m]['overall_performance']['speed']['speedup_factor'], 10))  # Cap speed factor for balance

        # Find most robust model
        robust_models = [(m, all_analyses[m]['quality_analysis'].get('robustness_metrics', {}).get('robustness_score', 0))
                        for m in models if 'robustness_metrics' in all_analyses[m]['quality_analysis']]
        best_robust = max(robust_models, key=lambda x: x[1])[0] if robust_models else best_accuracy

        report_lines.append("```")
        report_lines.append("1. Is accuracy the top priority?")
        report_lines.append(f"   YES → Use {best_accuracy}")
        report_lines.append(f"        (Accuracy: {all_analyses[best_accuracy]['overall_performance']['accuracy']['model']:.3f})")
        report_lines.append("")
        report_lines.append("2. Is speed the top priority?")
        report_lines.append(f"   YES → Use {best_speed}")
        report_lines.append(f"        (Speed: {all_analyses[best_speed]['overall_performance']['speed']['speedup_factor']:.1f}x faster)")
        report_lines.append("")
        report_lines.append("3. Do you have poor quality data?")
        report_lines.append(f"   YES → Use {best_robust}")
        if robust_models:
            robustness_score = all_analyses[best_robust]['quality_analysis']['robustness_metrics']['robustness_score']
            report_lines.append(f"        (Robustness: {robustness_score:.3f})")
        report_lines.append("")
        report_lines.append("4. Need balanced performance?")
        report_lines.append(f"   YES → Use {best_overall}")
        report_lines.append(f"        (Balanced accuracy and speed)")
        report_lines.append("```")
        report_lines.append("")

        # Statistical Summary
        report_lines.append("## Statistical Summary")
        report_lines.append("")

        # Count statistically significant models
        sig_accuracy_models = [m for m in models if all_analyses[m]['statistical_significance'].get('accuracy_test', {}).get('significant', False)]
        sig_time_models = [m for m in models if all_analyses[m]['statistical_significance'].get('time_test', {}).get('significant', False)]

        report_lines.append(f"- **Models with statistically significant accuracy improvement:** {len(sig_accuracy_models)}/{len(models)}")
        for model in sig_accuracy_models:
            p_val = all_analyses[model]['statistical_significance']['accuracy_test']['p_value']
            report_lines.append(f"  - {model} (p = {p_val:.2e})")

        report_lines.append("")
        report_lines.append(f"- **Models with statistically significant time improvement:** {len(sig_time_models)}/{len(models)}")
        for model in sig_time_models:
            time_test = all_analyses[model]['statistical_significance']['time_test']
            if time_test.get('p_value') is not None and not np.isnan(time_test.get('p_value')):
                p_val = time_test['p_value']
                report_lines.append(f"  - {model} (p = {p_val:.2e})")

        report_lines.append("")

        # Final Recommendations
        report_lines.append("## Final Recommendations")
        report_lines.append("")

        # Tier the models
        tier1_models = [m for m in models if all_analyses[m]['recommendations']['deployment_readiness'] == 'Ready']
        tier2_models = [m for m in models if all_analyses[m]['recommendations']['deployment_readiness'] == 'Ready with Caution']
        tier3_models = [m for m in models if all_analyses[m]['recommendations']['deployment_readiness'] == 'Pilot Testing Recommended']

        if tier1_models:
            report_lines.append("### Tier 1: Production Ready")
            for model in tier1_models:
                acc = all_analyses[model]['overall_performance']['accuracy']['model']
                speed = all_analyses[model]['overall_performance']['speed']['speedup_factor']
                report_lines.append(f"- **{model}**: {acc:.3f} accuracy, {speed:.1f}x speed")

        if tier2_models:
            report_lines.append("")
            report_lines.append("### Tier 2: Production Ready with Monitoring")
            for model in tier2_models:
                acc = all_analyses[model]['overall_performance']['accuracy']['model']
                speed = all_analyses[model]['overall_performance']['speed']['speedup_factor']
                report_lines.append(f"- **{model}**: {acc:.3f} accuracy, {speed:.1f}x speed")

        if tier3_models:
            report_lines.append("")
            report_lines.append("### Tier 3: Pilot Testing Phase")
            for model in tier3_models:
                acc = all_analyses[model]['overall_performance']['accuracy']['model']
                speed = all_analyses[model]['overall_performance']['speed']['speedup_factor']
                report_lines.append(f"- **{model}**: {acc:.3f} accuracy, {speed:.1f}x speed")

        report_lines.append("")
        report_lines.append("### Implementation Strategy")
        report_lines.append("")
        report_lines.append("1. **Start with Tier 1 models** for immediate deployment")
        report_lines.append("2. **Monitor performance** closely in production")
        report_lines.append("3. **A/B test** between top performers to find optimal choice")
        report_lines.append("4. **Consider ensemble approaches** combining multiple models")
        report_lines.append("5. **Regular re-evaluation** as new models become available")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {self.confidence_level*100:.0f}% confidence level*")

        return "***REMOVED***n".join(report_lines)

    def run_full_analysis(self) -> str:
        """Run complete analysis and generate report"""
        print("Starting comprehensive individual model analysis...")
        self.load_data()

        if not self.data:
            raise ValueError("No data loaded. Please check data files exist.")

        print("Generating comprehensive report...")
        report = self.generate_comprehensive_report()

        # Save report
        output_file = self.results_dir / "comprehensive_individual_model_analysis.md"
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"✅ Comprehensive analysis complete!")
        print(f"📄 Report saved to: {output_file}")

        return report

if __name__ == "__main__":
    analyzer = ComprehensiveModelAnalyzer()
    report = analyzer.run_full_analysis()

    # Print summary
    models = analyzer.get_models_list()
    print(f"***REMOVED***n📊 ANALYSIS SUMMARY")
    print(f"   Models analyzed: {len(models)}")
    print(f"   Confidence level: {analyzer.confidence_level*100:.0f}%")
    print(f"   Report length: {len(report.split())} words")
