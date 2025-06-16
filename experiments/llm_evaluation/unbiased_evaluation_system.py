#!/usr/bin/env python3
"""
Unbiased LLM Evaluation System

This module implements an unbiased evaluation system that uses different models
for judging performance to eliminate self-evaluation bias.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import LLM providers
import sys
sys.path.append('../../src')
from src.utils.llm_provider import OpenAIProvider, AnthropicProvider, DeepSeekProvider, MockLLMProvider

class UnbiasedEvaluationSystem:
    """
    Evaluation system that uses different models for judging to eliminate bias
    """
    
    def __init__(self, 
                 results_path: str = "performance_logs/llm_performance_results.csv",
                 judge_models: List[str] = None,
                 use_mock: bool = False):
        """Initialize the unbiased evaluation system"""
        self.results_path = Path(results_path)
        self.df = None
        self.use_mock = use_mock
        
        # Default judge models (different from typical test models)
        if judge_models is None:
            self.judge_models = [
                "gpt-4o-mini",  # Different from test models
                "claude-3-haiku-20240307",  # Fast, different provider
                "gpt-3.5-turbo"  # Baseline model
            ]
        else:
            self.judge_models = judge_models
            
        self.judge_providers = self._initialize_judge_providers()
        self.load_results()
        
    def _initialize_judge_providers(self) -> Dict[str, Any]:
        """Initialize judge model providers"""
        providers = {}
        
        for model in self.judge_models:
            try:
                if self.use_mock:
                    providers[model] = MockLLMProvider(model)
                elif "gpt" in model or "o4" in model:
                    providers[model] = OpenAIProvider(model)
                elif "claude" in model:
                    providers[model] = AnthropicProvider(model)
                elif "deepseek" in model:
                    providers[model] = DeepSeekProvider(model)
                else:
                    print(f"Warning: Unknown model {model}, using mock provider")
                    providers[model] = MockLLMProvider(model)
                    
                print(f"Initialized judge: {model}")
            except Exception as e:
                print(f"Failed to initialize {model}: {e}")
                providers[model] = MockLLMProvider(model)
                
        return providers
    
    def load_results(self) -> None:
        """Load performance results from CSV"""
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
            
        self.df = pd.read_csv(self.results_path)
        print(f"Loaded {len(self.df)} performance records for unbiased evaluation")
    
    def create_judge_prompt(self, task_complexity: str, llm_report: str, ground_truth: str) -> str:
        """Create an improved judge prompt with better instructions"""
        
        complexity_context = {
            'easy': 'gear identification from packing lists',
            'medium': 'printer assignment and part counting',
            'hard': 'compliance verification and date matching'
        }
        
        context = complexity_context.get(task_complexity, 'manufacturing data analysis')
        
        return f"""You are an expert evaluator for manufacturing data analysis systems. Your task is to objectively determine if the AI-generated report correctly answers the query.

**Task Type**: {context}

**Ground Truth Answer (Expected Result):**
{ground_truth}

**AI-Generated Report (To Evaluate):**
{llm_report}

**Evaluation Criteria:**
1. **Accuracy**: Does the report contain the correct core information?
2. **Completeness**: Are all required elements present?
3. **Data Quality Awareness**: For corrupted data conditions, does the report appropriately identify issues?

**Special Evaluation Rules:**
- For Q0 (perfect data): Report must match ground truth exactly
- For Q1 (space errors): Report is CORRECT if it finds the right answer despite extra spaces
- For Q2 (missing characters): Report is CORRECT if it finds the right answer despite missing characters  
- For Q3 (missing records): Report is CORRECT if it appropriately reports missing data with reasonable confidence

**Important**: Focus on whether the core factual content is correct, not on formatting or presentation style.

**Your Response**: Respond with exactly one word: "Correct" or "Incorrect"

Do not provide explanations, reasoning, or additional text. Just the single word judgment."""

    def evaluate_with_multiple_judges(self, task_id: str, llm_report: str, ground_truth: str, 
                                    task_complexity: str) -> Dict[str, Any]:
        """Evaluate a single response using multiple judge models"""
        
        judge_prompt = self.create_judge_prompt(task_complexity, llm_report, ground_truth)
        
        judgments = {}
        judge_details = {}
        
        for judge_model, provider in self.judge_providers.items():
            try:
                response = provider.generate(judge_prompt)
                judgment_text = response["content"].strip().lower()
                
                # Parse judgment
                if "correct" in judgment_text and "incorrect" not in judgment_text:
                    judgment = True
                elif "incorrect" in judgment_text:
                    judgment = False
                else:
                    print(f"Warning: Unclear judgment from {judge_model}: {judgment_text}")
                    judgment = False  # Default to incorrect for unclear responses
                
                judgments[judge_model] = judgment
                judge_details[judge_model] = {
                    'judgment': judgment,
                    'raw_response': judgment_text,
                    'tokens_used': response.get('input_tokens', 0) + response.get('output_tokens', 0)
                }
                
            except Exception as e:
                print(f"Error with judge {judge_model}: {e}")
                judgments[judge_model] = False
                judge_details[judge_model] = {
                    'judgment': False,
                    'raw_response': f"Error: {str(e)}",
                    'tokens_used': 0
                }
        
        # Calculate consensus
        correct_votes = sum(judgments.values())
        total_votes = len(judgments)
        consensus_score = correct_votes / total_votes if total_votes > 0 else 0
        
        # Majority vote decision
        final_judgment = consensus_score >= 0.5
        
        return {
            'task_id': task_id,
            'final_judgment': final_judgment,
            'consensus_score': consensus_score,
            'individual_judgments': judgments,
            'judge_details': judge_details,
            'total_judges': total_votes,
            'agreement_level': 'unanimous' if consensus_score in [0, 1] else 'majority' if consensus_score >= 0.5 else 'minority'
        }
    
    def re_evaluate_all_results(self) -> pd.DataFrame:
        """Re-evaluate all results using unbiased multi-judge system"""
        
        print("Starting unbiased re-evaluation of all results...")
        
        evaluation_results = []
        
        for idx, row in self.df.iterrows():
            print(f"Re-evaluating {idx+1}/{len(self.df)}: {row['task_id']}")
            
            # Parse ground truth
            try:
                ground_truth = json.loads(row['ground_truth_answer']) if isinstance(row['ground_truth_answer'], str) else row['ground_truth_answer']
                ground_truth_str = json.dumps(ground_truth, indent=2)
            except:
                ground_truth_str = str(row['ground_truth_answer'])
            
            # Evaluate with multiple judges
            evaluation = self.evaluate_with_multiple_judges(
                task_id=row['task_id'],
                llm_report=row['llm_final_report'],
                ground_truth=ground_truth_str,
                task_complexity=row['complexity']
            )
            
            # Add original data
            evaluation.update({
                'original_model': row['model'],
                'original_judgment': row['is_correct'],
                'complexity': row['complexity'],
                'quality_condition': row['quality_condition'],
                'completion_time_sec': row['completion_time_sec'],
                'total_cost_usd': row['total_cost_usd'],
                'final_confidence': row['final_confidence']
            })
            
            evaluation_results.append(evaluation)
        
        # Create results DataFrame
        results_df = pd.DataFrame(evaluation_results)
        
        # Save results
        output_path = "unbiased_evaluation_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Unbiased evaluation results saved to: {output_path}")
        
        return results_df
    
    def analyze_bias_impact(self, unbiased_results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the impact of removing evaluation bias"""
        
        analysis = {
            'overall_comparison': {},
            'by_model': {},
            'by_complexity': {},
            'by_quality': {},
            'judge_agreement': {},
            'bias_metrics': {}
        }
        
        # Overall comparison
        original_accuracy = self.df['is_correct'].mean()
        unbiased_accuracy = unbiased_results['final_judgment'].mean()
        
        analysis['overall_comparison'] = {
            'original_accuracy': original_accuracy,
            'unbiased_accuracy': unbiased_accuracy,
            'accuracy_change': unbiased_accuracy - original_accuracy,
            'bias_direction': 'positive' if original_accuracy > unbiased_accuracy else 'negative'
        }
        
        # Judge agreement analysis
        consensus_scores = unbiased_results['consensus_score'].values
        analysis['judge_agreement'] = {
            'mean_consensus': np.mean(consensus_scores),
            'unanimous_decisions': np.sum(consensus_scores == 1.0) / len(consensus_scores),
            'majority_decisions': np.sum(consensus_scores >= 0.5) / len(consensus_scores),
            'controversial_decisions': np.sum((consensus_scores > 0) & (consensus_scores < 1)) / len(consensus_scores)
        }
        
        # Bias metrics
        agreement_rate = np.mean(self.df['is_correct'] == unbiased_results['final_judgment'])
        analysis['bias_metrics'] = {
            'self_evaluation_agreement': agreement_rate,
            'bias_magnitude': abs(original_accuracy - unbiased_accuracy),
            'overconfidence_rate': np.mean((self.df['is_correct'] == True) & (unbiased_results['final_judgment'] == False)),
            'underconfidence_rate': np.mean((self.df['is_correct'] == False) & (unbiased_results['final_judgment'] == True))
        }
        
        return analysis
    
    def generate_bias_analysis_report(self, analysis: Dict[str, Any], 
                                    output_path: str = "bias_analysis_report.md") -> str:
        """Generate comprehensive bias analysis report"""
        
        report = []
        report.append("# LLM Evaluation Bias Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        overall = analysis['overall_comparison']
        report.append(f"**Original (Self-Evaluation) Accuracy**: {overall['original_accuracy']:.1%}")
        report.append(f"**Unbiased (Multi-Judge) Accuracy**: {overall['unbiased_accuracy']:.1%}")
        report.append(f"**Accuracy Change**: {overall['accuracy_change']:+.1%}")
        report.append(f"**Bias Direction**: {overall['bias_direction'].title()} bias detected")
        report.append("")
        
        # Judge Agreement Analysis
        agreement = analysis['judge_agreement']
        report.append("## Judge Agreement Analysis")
        report.append("")
        report.append(f"**Mean Consensus Score**: {agreement['mean_consensus']:.2f}")
        report.append(f"**Unanimous Decisions**: {agreement['unanimous_decisions']:.1%}")
        report.append(f"**Majority Decisions**: {agreement['majority_decisions']:.1%}")
        report.append(f"**Controversial Decisions**: {agreement['controversial_decisions']:.1%}")
        report.append("")
        
        # Bias Metrics
        bias = analysis['bias_metrics']
        report.append("## Bias Impact Metrics")
        report.append("")
        report.append(f"**Self-Evaluation Agreement Rate**: {bias['self_evaluation_agreement']:.1%}")
        report.append(f"**Bias Magnitude**: {bias['bias_magnitude']:.1%}")
        report.append(f"**Overconfidence Rate**: {bias['overconfidence_rate']:.1%}")
        report.append(f"**Underconfidence Rate**: {bias['underconfidence_rate']:.1%}")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if overall['accuracy_change'] > 0.05:
            report.append("- **High positive bias detected**: The model was significantly underestimating its performance")
            report.append("- Consider using external judges for all future evaluations")
        elif overall['accuracy_change'] < -0.05:
            report.append("- **High negative bias detected**: The model was significantly overestimating its performance")
            report.append("- Implement stricter evaluation criteria and external validation")
        else:
            report.append("- **Low bias detected**: Self-evaluation was relatively accurate")
            report.append("- Consider periodic external validation to maintain accuracy")
        
        if agreement['controversial_decisions'] > 0.2:
            report.append("- **High disagreement among judges**: Consider adding more judges or refining evaluation criteria")
        
        report_text = "***REMOVED***n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Bias analysis report saved to: {output_path}")
        
        return report_text

if __name__ == "__main__":
    # Example usage
    print("🔍 Starting Unbiased LLM Evaluation System")
    
    # Initialize with mock providers for demo
    evaluator = UnbiasedEvaluationSystem(use_mock=True)
    
    # Re-evaluate all results
    unbiased_results = evaluator.re_evaluate_all_results()
    
    # Analyze bias impact
    bias_analysis = evaluator.analyze_bias_impact(unbiased_results)
    
    # Generate report
    report = evaluator.generate_bias_analysis_report(bias_analysis)
    
    print("✅ Unbiased evaluation completed!")
    print(f"📊 Results: {len(unbiased_results)} tasks re-evaluated")
    print(f"📋 Bias analysis report generated")
