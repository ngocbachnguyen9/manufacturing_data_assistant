#!/usr/bin/env python3
"""
Unbiased LLM Evaluation Runner

This script runs the complete unbiased evaluation system that eliminates
self-evaluation bias by using multiple judge models.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

# Now import the modules
import yaml
from experiment.llm_evaluation import LLMEvaluationRunner

def load_config():
    """Load the experiment configuration"""
    config_files = [
        "config/experiment_config.yaml",
        "config/llm_config.yaml",
        "config/agent_config.yaml",
        "config/master_agent_planning_prompt.yaml"
    ]

    config = {}
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:  # Only update if file has content
                    config.update(file_config)
            print(f"   ✅ Loaded: {config_file}")
        else:
            print(f"   ⚠️  Missing: {config_file}")

    # Ensure we have the required llm_evaluation section
    if "llm_evaluation" not in config:
        config["llm_evaluation"] = {
            "models_to_test": ["deepseek-chat"]
        }

    # Verify master agent planning prompt is loaded
    if "master_agent_planning_prompt" not in config:
        print("   ⚠️  Master agent planning prompt not found in config")
        print("   Loading from master_agent_planning_prompt.yaml...")

        prompt_file = "config/master_agent_planning_prompt.yaml"
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                prompt_config = yaml.safe_load(f)
                if prompt_config:
                    config.update(prompt_config)
                    print("   ✅ Master agent planning prompt loaded")

    return config

def run_unbiased_evaluation():
    """Run the complete unbiased evaluation system"""
    
    print("🏛️  Starting Unbiased LLM Evaluation System")
    print("=" * 60)
    print()
    
    # Load configuration
    try:
        config = load_config()
        models_to_test = config["llm_evaluation"]["models_to_test"]
        print(f"📋 Configuration loaded successfully")
        print(f"   Models to evaluate: {models_to_test}")
        print(f"   Using unbiased judge system: ✅")
        print()
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Initialize the evaluation runner
    try:
        print("🔧 Initializing evaluation runner...")
        runner = LLMEvaluationRunner(config, use_mock=False)
        print("✅ Evaluation runner initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize evaluation runner: {e}")
        return False
    
    # Run the evaluation
    try:
        print("🚀 Starting unbiased evaluation...")
        print("   This will re-evaluate all tasks with multiple judge models")
        print("   to eliminate self-evaluation bias.")
        print()
        
        # Run the evaluation
        runner.run_evaluation()
        
        print("***REMOVED***n🎉 Unbiased Evaluation Complete!")
        print("=" * 40)
        
        # Print summary
        if hasattr(runner, 'results') and runner.results:
            total_tasks = len(runner.results)
            correct_tasks = sum(1 for r in runner.results if r['is_correct'])
            accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0
            
            print(f"📊 Evaluation Summary:")
            print(f"   • Total tasks evaluated: {total_tasks}")
            print(f"   • Unbiased accuracy: {accuracy:.1%}")
            print(f"   • Results saved to: experiments/llm_evaluation/performance_logs/")
            
            # Bias analysis summary
            if hasattr(runner, 'bias_analysis_data') and runner.bias_analysis_data:
                consensus_scores = [data['consensus_score'] for data in runner.bias_analysis_data]
                avg_consensus = sum(consensus_scores) / len(consensus_scores)
                unanimous = sum(1 for score in consensus_scores if score in [0, 1])
                
                print(f"***REMOVED***n🏛️  Judge Consensus Analysis:")
                print(f"   • Average consensus score: {avg_consensus:.2f}")
                print(f"   • Unanimous decisions: {unanimous}/{len(consensus_scores)} ({unanimous/len(consensus_scores)*100:.1f}%)")
                print(f"   • Bias analysis report generated")
        
        print(f"***REMOVED***n📁 Generated Files:")
        print(f"   • Performance results: experiments/llm_evaluation/performance_logs/llm_performance_results.csv")
        print(f"   • Bias analysis: experiments/llm_evaluation/performance_logs/bias_analysis_report.md")
        
        print(f"***REMOVED***n💡 Next Steps:")
        print(f"   1. Compare with previous biased results (94.4% accuracy)")
        print(f"   2. Review bias analysis report for detailed insights")
        print(f"   3. Use unbiased results for model selection decisions")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_previous_results():
    """Compare new unbiased results with previous biased results"""
    
    print("***REMOVED***n📊 Comparing with Previous Biased Results")
    print("-" * 45)
    
    # Load previous results if they exist
    previous_results_path = "experiments/llm_evaluation/performance_logs/llm_performance_results.csv"
    
    if os.path.exists(previous_results_path):
        import pandas as pd
        
        try:
            df = pd.read_csv(previous_results_path)
            
            # Check if we have bias analysis columns (indicating new results)
            if 'judge_consensus_score' in df.columns:
                print("✅ New unbiased results detected")
                
                # Calculate metrics
                total_tasks = len(df)
                unbiased_accuracy = df['is_correct'].mean()
                avg_consensus = df['judge_consensus_score'].mean()
                
                print(f"📈 Unbiased Evaluation Results:")
                print(f"   • Total tasks: {total_tasks}")
                print(f"   • Unbiased accuracy: {unbiased_accuracy:.1%}")
                print(f"   • Average judge consensus: {avg_consensus:.2f}")
                
                # Compare with expected biased results (94.4%)
                biased_accuracy = 0.944  # Previous inflated result
                bias_magnitude = biased_accuracy - unbiased_accuracy
                
                print(f"***REMOVED***n🔍 Bias Impact Analysis:")
                print(f"   • Previous (biased) accuracy: {biased_accuracy:.1%}")
                print(f"   • Current (unbiased) accuracy: {unbiased_accuracy:.1%}")
                print(f"   • Bias magnitude: {bias_magnitude:+.1%}")
                
                if bias_magnitude > 0.05:
                    print(f"   ⚠️  Significant positive bias detected in previous results!")
                elif bias_magnitude < -0.05:
                    print(f"   ⚠️  Significant negative bias detected in previous results!")
                else:
                    print(f"   ✅ Low bias detected - previous results were relatively accurate")
                
            else:
                print("⚠️  Results file exists but doesn't contain bias analysis")
                print("   Run the unbiased evaluation to generate new results")
                
        except Exception as e:
            print(f"❌ Error reading previous results: {e}")
    else:
        print("ℹ️  No previous results found - this is the first evaluation")

def main():
    """Main function"""
    
    # Check if we're in the right directory
    if not os.path.exists("src/experiment/llm_evaluation.py"):
        print("❌ Error: Please run this script from the manufacturing_data_assistant directory")
        print("   Current directory should contain src/experiment/llm_evaluation.py")
        return
    
    # Run the unbiased evaluation
    success = run_unbiased_evaluation()
    
    if success:
        # Compare with previous results
        compare_with_previous_results()
        
        print("***REMOVED***n🎯 Unbiased Evaluation System Successfully Deployed!")
        print("   Your LLM evaluation is now free from self-evaluation bias.")
        
    else:
        print("***REMOVED***n❌ Evaluation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
