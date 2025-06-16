#!/usr/bin/env python3
"""
Separated LLM Evaluation Runner

This script runs controlled, separated evaluations to test:
1. Different models (one at a time)
2. Different prompt lengths (short, normal, long)
3. Manageable data chunks for analysis

Each run generates separate output files for clean comparison.
"""

import sys
import os
from pathlib import Path
import yaml
import json
from datetime import datetime

# Add src to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

from experiment.llm_evaluation import LLMEvaluationRunner

class SeparatedEvaluationRunner:
    """
    Manages separated evaluation runs for clean model and prompt comparison
    """
    
    def __init__(self):
        self.base_config = self.load_base_config()
        self.output_base_dir = Path("experiments/llm_evaluation/separated_runs")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Available models and prompt lengths
        self.available_models = ["deepseek-reasoner", "o4-mini-2025-04-16", "claude-sonnet-4-20250514"]
        self.prompt_lengths = ["short", "normal", "long"]
        
    def load_base_config(self):
        """Load the base configuration"""
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
                    if file_config:
                        config.update(file_config)
        
        return config
    
    def create_run_config(self, model: str, prompt_length: str, task_subset: str = "all"):
        """Create configuration for a specific run"""
        run_config = self.base_config.copy()
        
        # Set single model
        run_config["llm_evaluation"]["models_to_test"] = [model]
        
        # Add prompt length specification
        run_config["llm_evaluation"]["prompt_length"] = prompt_length
        run_config["llm_evaluation"]["task_subset"] = task_subset
        
        return run_config
    
    def run_single_evaluation(self, model: str, prompt_length: str, task_subset: str = "all"):
        """Run evaluation for a single model with specific prompt length"""
        
        # Create run identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{model}_{prompt_length}_{task_subset}_{timestamp}"
        
        print(f"***REMOVED***n🚀 Starting Evaluation Run: {run_id}")
        print("=" * 60)
        print(f"   Model: {model}")
        print(f"   Prompt Length: {prompt_length}")
        print(f"   Task Subset: {task_subset}")
        print()
        
        # Create run-specific output directory
        run_output_dir = self.output_base_dir / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run configuration
        run_config = self.create_run_config(model, prompt_length, task_subset)
        
        # Save run configuration
        config_path = run_output_dir / "run_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(run_config, f, default_flow_style=False)
        
        try:
            # Initialize evaluation runner with custom log directory
            runner = LLMEvaluationRunner(run_config, use_mock=False)
            runner.log_dir = str(run_output_dir)  # Override log directory
            
            # Run the evaluation
            runner.run_evaluation()
            
            # Generate run summary
            self.generate_run_summary(run_output_dir, run_id, model, prompt_length, task_subset)
            
            print(f"***REMOVED***n✅ Run Complete: {run_id}")
            print(f"📁 Results saved to: {run_output_dir}")
            
            return True, run_output_dir
            
        except Exception as e:
            print(f"***REMOVED***n❌ Run Failed: {run_id}")
            print(f"Error: {e}")
            return False, None
    
    def generate_run_summary(self, run_dir: Path, run_id: str, model: str, prompt_length: str, task_subset: str):
        """Generate summary for a completed run"""
        
        # Load results if available
        results_file = run_dir / "llm_performance_results.csv"
        
        summary = {
            "run_id": run_id,
            "model": model,
            "prompt_length": prompt_length,
            "task_subset": task_subset,
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if results_file.exists() else "failed"
        }
        
        if results_file.exists():
            import pandas as pd
            df = pd.read_csv(results_file)
            
            summary.update({
                "total_tasks": len(df),
                "accuracy": df['is_correct'].mean(),
                "avg_completion_time": df['completion_time_sec'].mean(),
                "avg_cost": df['total_cost_usd'].mean(),
                "avg_consensus": df['judge_consensus_score'].mean() if 'judge_consensus_score' in df.columns else None
            })
        
        # Save summary
        summary_file = run_dir / "run_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def list_available_runs(self):
        """List all available evaluation run options"""
        print("📋 Available Evaluation Runs")
        print("=" * 40)
        print()
        
        print("🤖 Models:")
        for i, model in enumerate(self.available_models, 1):
            print(f"   {i}. {model}")
        print()
        
        print("📝 Prompt Lengths:")
        for i, length in enumerate(self.prompt_lengths, 1):
            description = {
                "short": "Minimal, concise instructions",
                "normal": "Balanced detail and clarity", 
                "long": "Comprehensive, detailed instructions (current baseline)"
            }
            print(f"   {i}. {length} - {description[length]}")
        print()
        
        print("📊 Task Subsets:")
        print("   1. all - All 90 tasks (30 easy + 30 medium + 30 hard)")
        print("   2. sample - 9 tasks (3 easy + 3 medium + 3 hard)")
        print("   3. easy - 30 easy tasks only")
        print("   4. medium - 30 medium tasks only") 
        print("   5. hard - 30 hard tasks only")
        print()
    
    def interactive_run_selection(self):
        """Interactive selection of evaluation run parameters"""
        self.list_available_runs()
        
        print("🎯 Select Evaluation Parameters:")
        print("-" * 35)
        
        # Model selection
        print("***REMOVED***nSelect model:")
        for i, model in enumerate(self.available_models, 1):
            print(f"  {i}. {model}")
        
        while True:
            try:
                model_choice = int(input("Enter model number (1-3): ")) - 1
                if 0 <= model_choice < len(self.available_models):
                    selected_model = self.available_models[model_choice]
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Prompt length selection
        print("***REMOVED***nSelect prompt length:")
        for i, length in enumerate(self.prompt_lengths, 1):
            print(f"  {i}. {length}")
        
        while True:
            try:
                prompt_choice = int(input("Enter prompt length number (1-3): ")) - 1
                if 0 <= prompt_choice < len(self.prompt_lengths):
                    selected_prompt = self.prompt_lengths[prompt_choice]
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Task subset selection
        task_options = ["all", "sample", "easy", "medium", "hard"]
        print("***REMOVED***nSelect task subset:")
        for i, subset in enumerate(task_options, 1):
            print(f"  {i}. {subset}")
        
        while True:
            try:
                task_choice = int(input("Enter task subset number (1-5): ")) - 1
                if 0 <= task_choice < len(task_options):
                    selected_tasks = task_options[task_choice]
                    break
                else:
                    print("Invalid choice. Please enter 1-5.")
            except ValueError:
                print("Please enter a valid number.")
        
        return selected_model, selected_prompt, selected_tasks
    
    def run_batch_evaluation(self, models: list = None, prompt_lengths: list = None, task_subset: str = "sample"):
        """Run batch evaluation across multiple models and prompt lengths"""
        
        models = models or self.available_models
        prompt_lengths = prompt_lengths or self.prompt_lengths
        
        print(f"***REMOVED***n🔄 Starting Batch Evaluation")
        print("=" * 40)
        print(f"Models: {models}")
        print(f"Prompt Lengths: {prompt_lengths}")
        print(f"Task Subset: {task_subset}")
        print()
        
        results = []
        total_runs = len(models) * len(prompt_lengths)
        current_run = 0
        
        for model in models:
            for prompt_length in prompt_lengths:
                current_run += 1
                print(f"***REMOVED***n📊 Batch Progress: {current_run}/{total_runs}")
                
                success, run_dir = self.run_single_evaluation(model, prompt_length, task_subset)
                
                results.append({
                    "model": model,
                    "prompt_length": prompt_length,
                    "task_subset": task_subset,
                    "success": success,
                    "output_dir": str(run_dir) if run_dir else None
                })
        
        # Generate batch summary
        self.generate_batch_summary(results)
        
        return results
    
    def generate_batch_summary(self, results: list):
        """Generate summary for batch evaluation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_base_dir / f"batch_summary_{timestamp}.json"
        
        batch_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(results),
            "successful_runs": sum(1 for r in results if r["success"]),
            "failed_runs": sum(1 for r in results if not r["success"]),
            "results": results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"***REMOVED***n📋 Batch Summary:")
        print(f"   Total Runs: {batch_summary['total_runs']}")
        print(f"   Successful: {batch_summary['successful_runs']}")
        print(f"   Failed: {batch_summary['failed_runs']}")
        print(f"   Summary saved to: {summary_file}")

def main():
    """Main function with user interaction"""
    
    print("🏛️  Separated LLM Evaluation System")
    print("=" * 50)
    print()
    print("This system runs controlled, separated evaluations to test:")
    print("• Different models (one at a time)")
    print("• Different prompt lengths (short, normal, long)")
    print("• Manageable data chunks for clean analysis")
    print()
    
    runner = SeparatedEvaluationRunner()
    
    print("Choose evaluation mode:")
    print("1. Interactive single run")
    print("2. Batch evaluation (sample tasks)")
    print("3. Quick test (deepseek-reasoner, short prompts, sample tasks)")
    
    while True:
        try:
            choice = int(input("***REMOVED***nEnter choice (1-3): "))
            if choice in [1, 2, 3]:
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")
    
    if choice == 1:
        # Interactive single run
        model, prompt_length, task_subset = runner.interactive_run_selection()
        runner.run_single_evaluation(model, prompt_length, task_subset)
        
    elif choice == 2:
        # Batch evaluation with sample tasks
        print("***REMOVED***n⚠️  This will run 9 evaluations (3 models × 3 prompt lengths)")
        confirm = input("Continue? (y/n): ").lower().strip()
        if confirm in ['y', 'yes']:
            runner.run_batch_evaluation(task_subset="sample")
        else:
            print("Batch evaluation cancelled.")
            
    elif choice == 3:
        # Quick test
        print("***REMOVED***n🚀 Running quick test...")
        runner.run_single_evaluation("deepseek-reasoner", "short", "sample")

if __name__ == "__main__":
    main()
