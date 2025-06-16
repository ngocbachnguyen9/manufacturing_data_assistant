#!/usr/bin/env python3
"""
Comprehensive LLM Benchmarking Suite

This module provides a unified benchmarking system that combines all evaluation
components to provide comprehensive LLM performance assessment.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from evaluation_framework import LLMEvaluationFramework, BenchmarkResult
from model_comparison import ModelComparison
from prompt_effectiveness_analyzer import PromptEffectivenessAnalyzer

@dataclass
class BenchmarkSuite:
    """Container for complete benchmark suite results"""
    timestamp: str
    models_evaluated: List[str]
    total_tasks: int
    benchmark_results: Dict[str, BenchmarkResult]
    statistical_comparison: Dict[str, Any]
    prompt_effectiveness: Dict[str, Any]
    manufacturing_metrics: Dict[str, Any]
    recommendations: Dict[str, str]

class ComprehensiveBenchmarkingSuite:
    """
    Unified benchmarking system for comprehensive LLM evaluation
    """
    
    def __init__(self,
                 results_path: str = "performance_logs/llm_performance_results.csv",
                 output_dir: str = "benchmark_suite_results/"):
        """Initialize the comprehensive benchmarking suite"""
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component analyzers
        self.evaluation_framework = LLMEvaluationFramework(str(results_path))
        self.model_comparison = ModelComparison(str(results_path))
        self.prompt_analyzer = PromptEffectivenessAnalyzer(str(results_path))
        
        print(f"Benchmarking suite initialized with output directory: {self.output_dir}")
    
    def run_comprehensive_benchmark(self) -> BenchmarkSuite:
        """Run the complete benchmarking suite"""
        print("Starting comprehensive LLM benchmarking suite...")
        start_time = time.time()
        
        # 1. Basic Performance Evaluation
        print("1. Running performance evaluation...")
        benchmark_results = self.evaluation_framework.create_benchmark_suite()
        
        # 2. Statistical Model Comparison
        print("2. Running statistical model comparison...")
        statistical_comparison = self.model_comparison.compare_models_statistical()
        
        # 3. Prompt Effectiveness Analysis
        print("3. Running prompt effectiveness analysis...")
        prompt_effectiveness = self.prompt_analyzer.analyze_prompt_effectiveness()
        
        # 4. Manufacturing-Specific Metrics
        print("4. Calculating manufacturing-specific metrics...")
        manufacturing_metrics = self._calculate_manufacturing_metrics()
        
        # 5. Generate Recommendations
        print("5. Generating recommendations...")
        recommendations = self._generate_recommendations(
            benchmark_results, statistical_comparison, manufacturing_metrics
        )
        
        # Create comprehensive benchmark suite result
        models_evaluated = list(benchmark_results.keys())
        total_tasks = len(self.evaluation_framework.df)
        
        suite_result = BenchmarkSuite(
            timestamp=datetime.now().isoformat(),
            models_evaluated=models_evaluated,
            total_tasks=total_tasks,
            benchmark_results=benchmark_results,
            statistical_comparison=statistical_comparison,
            prompt_effectiveness={k: asdict(v) for k, v in prompt_effectiveness.items()},
            manufacturing_metrics=manufacturing_metrics,
            recommendations=recommendations
        )
        
        end_time = time.time()
        print(f"Comprehensive benchmarking completed in {end_time - start_time:.1f} seconds")
        
        return suite_result
    
    def _calculate_manufacturing_metrics(self) -> Dict[str, Any]:
        """Calculate manufacturing-specific performance metrics"""
        df = self.evaluation_framework.df
        
        metrics = {
            'data_quality_handling': {},
            'task_complexity_performance': {},
            'error_detection_capabilities': {},
            'cross_system_validation': {},
            'manufacturing_domain_accuracy': {}
        }
        
        # Data Quality Handling Performance
        for quality in ['Q0', 'Q1', 'Q2', 'Q3']:
            quality_data = df[df['quality_condition'] == quality]
            if len(quality_data) > 0:
                metrics['data_quality_handling'][quality] = {
                    'accuracy': quality_data['is_correct'].mean(),
                    'avg_confidence': quality_data['final_confidence'].mean(),
                    'task_count': len(quality_data),
                    'error_detection_rate': self._calculate_error_detection_rate(quality_data)
                }
        
        # Task Complexity Performance
        for complexity in ['easy', 'medium', 'hard']:
            complexity_data = df[df['complexity'] == complexity]
            if len(complexity_data) > 0:
                metrics['task_complexity_performance'][complexity] = {
                    'accuracy': complexity_data['is_correct'].mean(),
                    'avg_time': complexity_data['completion_time_sec'].mean(),
                    'avg_cost': complexity_data['total_cost_usd'].mean(),
                    'task_count': len(complexity_data)
                }
        
        # Manufacturing Domain Accuracy by Task Type
        task_types = {
            'gear_identification': df[df['complexity'] == 'easy'],
            'printer_analysis': df[df['complexity'] == 'medium'],
            'compliance_verification': df[df['complexity'] == 'hard']
        }
        
        for task_type, task_data in task_types.items():
            if len(task_data) > 0:
                metrics['manufacturing_domain_accuracy'][task_type] = {
                    'accuracy': task_data['is_correct'].mean(),
                    'avg_confidence': task_data['final_confidence'].mean(),
                    'task_count': len(task_data)
                }
        
        # Cross-System Validation Performance
        # Analyze how well models handle multi-system data integration
        metrics['cross_system_validation'] = {
            'overall_integration_success': df['is_correct'].mean(),
            'data_reconciliation_quality': df['final_confidence'].mean(),
            'system_complexity_handling': {
                'single_system_tasks': df[df['complexity'] == 'easy']['is_correct'].mean(),
                'multi_system_tasks': df[df['complexity'].isin(['medium', 'hard'])]['is_correct'].mean()
            }
        }
        
        return metrics
    
    def _calculate_error_detection_rate(self, data: pd.DataFrame) -> float:
        """Calculate error detection rate for a dataset"""
        if len(data) == 0:
            return 0.0
        
        # Count tasks where reconciliation_issues is not empty
        detected_issues = data['reconciliation_issues'].apply(
            lambda x: len(eval(x)) > 0 if isinstance(x, str) and x.strip() else False
        )
        return detected_issues.mean()
    
    def _generate_recommendations(self, 
                                benchmark_results: Dict[str, BenchmarkResult],
                                statistical_comparison: Dict[str, Any],
                                manufacturing_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate actionable recommendations based on benchmark results"""
        recommendations = {}
        
        # Model Selection Recommendations
        if len(benchmark_results) > 1:
            best_overall = max(benchmark_results.items(), key=lambda x: x[1].overall_score)
            best_accuracy = max(benchmark_results.items(), key=lambda x: x[1].metrics.accuracy)
            best_speed = min(benchmark_results.items(), key=lambda x: x[1].metrics.avg_completion_time)
            best_cost = min(benchmark_results.items(), key=lambda x: x[1].metrics.avg_cost)
            
            recommendations['model_selection'] = f"""
            Best Overall Performance: {best_overall[0]} (Score: {best_overall[1].overall_score:.3f})
            Highest Accuracy: {best_accuracy[0]} ({best_accuracy[1].metrics.accuracy:.1%})
            Fastest Processing: {best_speed[0]} ({best_speed[1].metrics.avg_completion_time:.1f}s avg)
            Most Cost-Effective: {best_cost[0]} (${best_cost[1].metrics.avg_cost:.4f} avg)
            """
        
        # Data Quality Handling Recommendations
        quality_performance = manufacturing_metrics.get('data_quality_handling', {})
        if quality_performance:
            worst_quality = min(quality_performance.items(), key=lambda x: x[1]['accuracy'])
            recommendations['data_quality'] = f"""
            Weakest Performance on: {worst_quality[0]} conditions ({worst_quality[1]['accuracy']:.1%} accuracy)
            Recommendation: Implement additional data validation and error handling for {worst_quality[0]} scenarios
            Consider prompt engineering improvements for data quality issue detection
            """
        
        # Task Complexity Recommendations
        complexity_performance = manufacturing_metrics.get('task_complexity_performance', {})
        if complexity_performance:
            hardest_tasks = min(complexity_performance.items(), key=lambda x: x[1]['accuracy'])
            recommendations['task_complexity'] = f"""
            Most Challenging Tasks: {hardest_tasks[0]} ({hardest_tasks[1]['accuracy']:.1%} accuracy)
            Recommendation: Consider task decomposition or specialized prompting for {hardest_tasks[0]} tasks
            Average time for {hardest_tasks[0]} tasks: {hardest_tasks[1]['avg_time']:.1f}s
            """
        
        # Manufacturing Domain Recommendations
        domain_accuracy = manufacturing_metrics.get('manufacturing_domain_accuracy', {})
        if domain_accuracy:
            weakest_domain = min(domain_accuracy.items(), key=lambda x: x[1]['accuracy'])
            recommendations['manufacturing_domain'] = f"""
            Weakest Manufacturing Domain: {weakest_domain[0]} ({weakest_domain[1]['accuracy']:.1%} accuracy)
            Recommendation: Develop domain-specific training or fine-tuning for {weakest_domain[0]}
            Consider adding more context or examples for this task type
            """
        
        # Cost Optimization Recommendations
        if len(benchmark_results) > 1:
            cost_analysis = [(model, result.metrics.avg_cost, result.metrics.accuracy) 
                           for model, result in benchmark_results.items()]
            cost_analysis.sort(key=lambda x: x[1])  # Sort by cost
            
            recommendations['cost_optimization'] = f"""
            Most Cost-Effective: {cost_analysis[0][0]} (${cost_analysis[0][1]:.4f}, {cost_analysis[0][2]:.1%} accuracy)
            Consider cost vs accuracy trade-offs based on use case requirements
            For high-volume tasks, consider the most cost-effective model
            For critical tasks, prioritize accuracy over cost
            """
        
        return recommendations
    
    def export_benchmark_suite(self, suite_result: BenchmarkSuite, 
                              filename: str = None) -> str:
        """Export complete benchmark suite results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_suite_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert BenchmarkResult objects to dictionaries for JSON serialization
        exportable_results = {}
        for model, result in suite_result.benchmark_results.items():
            exportable_results[model] = {
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
        
        export_data = {
            'timestamp': suite_result.timestamp,
            'models_evaluated': suite_result.models_evaluated,
            'total_tasks': suite_result.total_tasks,
            'benchmark_results': exportable_results,
            'statistical_comparison': suite_result.statistical_comparison,
            'prompt_effectiveness': suite_result.prompt_effectiveness,
            'manufacturing_metrics': suite_result.manufacturing_metrics,
            'recommendations': suite_result.recommendations
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Complete benchmark suite exported to: {output_path}")
        return str(output_path)
    
    def generate_executive_summary(self, suite_result: BenchmarkSuite) -> str:
        """Generate executive summary of benchmark results"""
        summary = []
        summary.append("# LLM Manufacturing Data Analysis - Executive Benchmark Summary")
        summary.append("=" * 70)
        summary.append("")
        summary.append(f"**Evaluation Date:** {suite_result.timestamp}")
        summary.append(f"**Models Evaluated:** {', '.join(suite_result.models_evaluated)}")
        summary.append(f"**Total Tasks Analyzed:** {suite_result.total_tasks}")
        summary.append("")
        
        # Top Performer
        if suite_result.benchmark_results:
            top_performer = max(suite_result.benchmark_results.items(), 
                              key=lambda x: x[1].overall_score)
            summary.append("## 🏆 Top Performer")
            summary.append(f"**{top_performer[0]}** - Overall Score: {top_performer[1].overall_score:.3f}")
            summary.append(f"- Accuracy: {top_performer[1].metrics.accuracy:.1%}")
            summary.append(f"- Avg Time: {top_performer[1].metrics.avg_completion_time:.1f}s")
            summary.append(f"- Avg Cost: ${top_performer[1].metrics.avg_cost:.4f}")
            summary.append("")
        
        # Key Findings
        summary.append("## 🔍 Key Findings")
        
        # Manufacturing domain performance
        if 'manufacturing_domain_accuracy' in suite_result.manufacturing_metrics:
            domain_perf = suite_result.manufacturing_metrics['manufacturing_domain_accuracy']
            summary.append("### Manufacturing Task Performance:")
            for task_type, metrics in domain_perf.items():
                summary.append(f"- **{task_type.replace('_', ' ').title()}**: {metrics['accuracy']:.1%} accuracy")
        
        # Data quality handling
        if 'data_quality_handling' in suite_result.manufacturing_metrics:
            quality_perf = suite_result.manufacturing_metrics['data_quality_handling']
            summary.append("### Data Quality Handling:")
            for quality, metrics in quality_perf.items():
                quality_desc = {'Q0': 'Perfect Data', 'Q1': 'Space Errors', 
                              'Q2': 'Missing Chars', 'Q3': 'Missing Records'}
                summary.append(f"- **{quality_desc.get(quality, quality)}**: {metrics['accuracy']:.1%} accuracy")
        
        summary.append("")
        
        # Recommendations
        summary.append("## 💡 Key Recommendations")
        for rec_type, rec_text in suite_result.recommendations.items():
            summary.append(f"### {rec_type.replace('_', ' ').title()}")
            summary.append(rec_text.strip())
            summary.append("")
        
        return "***REMOVED***n".join(summary)
    
    def run_and_export_complete_suite(self) -> Tuple[BenchmarkSuite, str, str]:
        """Run complete benchmark suite and export all results"""
        # Run comprehensive benchmark
        suite_result = self.run_comprehensive_benchmark()
        
        # Export detailed results
        json_path = self.export_benchmark_suite(suite_result)
        
        # Generate and save executive summary
        summary = self.generate_executive_summary(suite_result)
        summary_path = self.output_dir / "executive_summary.md"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        # Generate all visualizations
        print("Generating comprehensive visualizations...")
        self.model_comparison.create_comparison_visualizations(
            str(self.output_dir / "visualizations")
        )
        self.prompt_analyzer.create_prompt_comparison_visualizations(
            str(self.output_dir / "prompt_analysis")
        )
        
        print(f"Complete benchmark suite completed!")
        print(f"Results exported to: {json_path}")
        print(f"Executive summary: {summary_path}")
        
        return suite_result, json_path, str(summary_path)

if __name__ == "__main__":
    # Example usage
    suite = ComprehensiveBenchmarkingSuite()
    
    # Run complete benchmark suite
    results, json_path, summary_path = suite.run_and_export_complete_suite()
    
    print("***REMOVED***n" + "="*50)
    print("COMPREHENSIVE BENCHMARKING COMPLETED")
    print("="*50)
    print(f"Models evaluated: {len(results.models_evaluated)}")
    print(f"Total tasks analyzed: {results.total_tasks}")
    print(f"Detailed results: {json_path}")
    print(f"Executive summary: {summary_path}")
