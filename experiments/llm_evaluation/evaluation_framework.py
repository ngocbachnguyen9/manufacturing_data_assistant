#!/usr/bin/env python3
"""
LLM Performance Evaluation and Benchmarking Framework

This module provides comprehensive evaluation and benchmarking capabilities for
LLM performance on manufacturing data analysis tasks.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    accuracy: float
    avg_completion_time: float
    avg_cost: float
    avg_confidence: float
    error_detection_rate: float
    token_efficiency: float

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    model_name: str
    overall_score: float
    complexity_scores: Dict[str, float]
    quality_scores: Dict[str, float]
    metrics: PerformanceMetrics
    rank: int

class LLMEvaluationFramework:
    """
    Comprehensive evaluation and benchmarking framework for LLM performance
    on manufacturing data analysis tasks.
    """
    
    def __init__(self, results_path: str = "performance_logs/llm_performance_results.csv"):
        """Initialize the evaluation framework"""
        self.results_path = Path(results_path)
        self.df = None
        self.load_results()
        
    def load_results(self) -> None:
        """Load performance results from CSV"""
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
            
        self.df = pd.read_csv(self.results_path)
        print(f"Loaded {len(self.df)} performance records")
        
    def calculate_performance_metrics(self, model_name: str = None) -> Dict[str, PerformanceMetrics]:
        """Calculate comprehensive performance metrics for each model"""
        if model_name:
            models = [model_name]
        else:
            models = self.df['model'].unique()
            
        metrics_by_model = {}
        
        for model in models:
            model_data = self.df[self.df['model'] == model]
            
            # Basic metrics
            accuracy = model_data['is_correct'].mean()
            avg_completion_time = model_data['completion_time_sec'].mean()
            avg_cost = model_data['total_cost_usd'].mean()
            avg_confidence = model_data['final_confidence'].mean()
            
            # Error detection rate (tasks with data quality issues that were identified)
            quality_issues = model_data[model_data['quality_condition'] != 'Q0']
            error_detection_rate = 0.0
            if len(quality_issues) > 0:
                # Count tasks where reconciliation_issues is not empty
                detected_issues = quality_issues['reconciliation_issues'].apply(
                    lambda x: len(eval(x)) > 0 if isinstance(x, str) and x.strip() else False
                )
                error_detection_rate = detected_issues.mean()
            
            # Token efficiency (correct answers per token)
            total_tokens = model_data['input_tokens'] + model_data['output_tokens']
            correct_tasks = model_data[model_data['is_correct'] == True]
            token_efficiency = len(correct_tasks) / total_tokens.sum() if total_tokens.sum() > 0 else 0
            
            metrics_by_model[model] = PerformanceMetrics(
                accuracy=accuracy,
                avg_completion_time=avg_completion_time,
                avg_cost=avg_cost,
                avg_confidence=avg_confidence,
                error_detection_rate=error_detection_rate,
                token_efficiency=token_efficiency * 1000  # Per 1000 tokens
            )
            
        return metrics_by_model
    
    def generate_performance_report(self, output_path: str = None) -> str:
        """Generate a comprehensive performance report"""
        metrics = self.calculate_performance_metrics()
        
        report = []
        report.append("# LLM Performance Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Overall summary
        report.append("## Overall Performance Summary")
        report.append("")
        
        for model, metric in metrics.items():
            report.append(f"### {model}")
            report.append(f"- **Accuracy**: {metric.accuracy:.1%}")
            report.append(f"- **Avg Completion Time**: {metric.avg_completion_time:.1f}s")
            report.append(f"- **Avg Cost**: ${metric.avg_cost:.4f}")
            report.append(f"- **Avg Confidence**: {metric.avg_confidence:.2f}")
            report.append(f"- **Error Detection Rate**: {metric.error_detection_rate:.1%}")
            report.append(f"- **Token Efficiency**: {metric.token_efficiency:.2f} correct/1000 tokens")
            report.append("")
        
        # Performance by complexity
        report.append("## Performance by Task Complexity")
        report.append("")
        
        complexity_analysis = self.analyze_by_dimension('complexity')
        for complexity, stats in complexity_analysis.items():
            report.append(f"### {complexity.title()} Tasks")
            report.append(f"- Accuracy: {stats['accuracy']:.1%}")
            report.append(f"- Avg Time: {stats['avg_time']:.1f}s")
            report.append(f"- Task Count: {stats['count']}")
            report.append("")
        
        # Performance by quality condition
        report.append("## Performance by Data Quality Condition")
        report.append("")
        
        quality_analysis = self.analyze_by_dimension('quality_condition')
        for quality, stats in quality_analysis.items():
            report.append(f"### {quality} - {self._get_quality_description(quality)}")
            report.append(f"- Accuracy: {stats['accuracy']:.1%}")
            report.append(f"- Avg Time: {stats['avg_time']:.1f}s")
            report.append(f"- Task Count: {stats['count']}")
            report.append("")
        
        report_text = "***REMOVED***n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")
            
        return report_text
    
    def analyze_by_dimension(self, dimension: str) -> Dict[str, Dict]:
        """Analyze performance by a specific dimension (complexity, quality_condition, etc.)"""
        analysis = {}
        
        for value in self.df[dimension].unique():
            subset = self.df[self.df[dimension] == value]
            
            analysis[value] = {
                'accuracy': subset['is_correct'].mean(),
                'avg_time': subset['completion_time_sec'].mean(),
                'avg_cost': subset['total_cost_usd'].mean(),
                'avg_confidence': subset['final_confidence'].mean(),
                'count': len(subset)
            }
            
        return analysis
    
    def _get_quality_description(self, quality_code: str) -> str:
        """Get human-readable description for quality condition codes"""
        descriptions = {
            'Q0': 'Baseline (Perfect Data)',
            'Q1': 'Space Injection Errors',
            'Q2': 'Character Removal Errors', 
            'Q3': 'Missing Record Errors'
        }
        return descriptions.get(quality_code, 'Unknown')
    
    def create_benchmark_suite(self) -> Dict[str, BenchmarkResult]:
        """Create standardized benchmark results for model comparison"""
        metrics = self.calculate_performance_metrics()
        benchmark_results = {}
        
        # Calculate weighted benchmark scores
        for model, metric in metrics.items():
            # Weighted scoring (accuracy=40%, speed=20%, cost=15%, confidence=15%, error_detection=10%)
            speed_score = max(0, 1 - (metric.avg_completion_time / 60))  # Normalize to 60s max
            cost_score = max(0, 1 - (metric.avg_cost / 0.01))  # Normalize to $0.01 max
            
            overall_score = (
                metric.accuracy * 0.40 +
                speed_score * 0.20 +
                cost_score * 0.15 +
                metric.avg_confidence * 0.15 +
                metric.error_detection_rate * 0.10
            )
            
            # Calculate scores by complexity and quality
            complexity_scores = {}
            for complexity in ['easy', 'medium', 'hard']:
                subset = self.df[(self.df['model'] == model) & (self.df['complexity'] == complexity)]
                if len(subset) > 0:
                    complexity_scores[complexity] = subset['is_correct'].mean()
            
            quality_scores = {}
            for quality in ['Q0', 'Q1', 'Q2', 'Q3']:
                subset = self.df[(self.df['model'] == model) & (self.df['quality_condition'] == quality)]
                if len(subset) > 0:
                    quality_scores[quality] = subset['is_correct'].mean()
            
            benchmark_results[model] = BenchmarkResult(
                model_name=model,
                overall_score=overall_score,
                complexity_scores=complexity_scores,
                quality_scores=quality_scores,
                metrics=metric,
                rank=0  # Will be set after sorting
            )
        
        # Assign ranks
        sorted_results = sorted(benchmark_results.values(), key=lambda x: x.overall_score, reverse=True)
        for i, result in enumerate(sorted_results):
            result.rank = i + 1
            benchmark_results[result.model_name] = result
            
        return benchmark_results
    
    def export_benchmark_results(self, output_path: str = "benchmark_results.json"):
        """Export benchmark results to JSON"""
        benchmark_results = self.create_benchmark_suite()
        
        # Convert to serializable format
        export_data = {}
        for model, result in benchmark_results.items():
            export_data[model] = {
                'model_name': result.model_name,
                'overall_score': result.overall_score,
                'rank': result.rank,
                'complexity_scores': result.complexity_scores,
                'quality_scores': result.quality_scores,
                'metrics': {
                    'accuracy': result.metrics.accuracy,
                    'avg_completion_time': result.metrics.avg_completion_time,
                    'avg_cost': result.metrics.avg_cost,
                    'avg_confidence': result.metrics.avg_confidence,
                    'error_detection_rate': result.metrics.error_detection_rate,
                    'token_efficiency': result.metrics.token_efficiency
                }
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Benchmark results exported to: {output_path}")
        return export_data

if __name__ == "__main__":
    # Example usage
    framework = LLMEvaluationFramework()

    # Generate performance report (use relative path from current directory)
    report = framework.generate_performance_report("performance_report.md")
    print("Performance report generated!")

    # Export benchmark results (use relative path from current directory)
    benchmark_data = framework.export_benchmark_results("benchmark_results.json")
    print("Benchmark results exported!")
