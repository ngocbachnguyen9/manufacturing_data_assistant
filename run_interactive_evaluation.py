#!/usr/bin/env python3
"""
Interactive LLM Evaluation Runner

This script provides an interactive interface to run LLM evaluations with:
- Model selection (single model per run)
- Prompt length selection (short, normal, long)
- Task subset selection (all, sample, by complexity)
- Unbiased judge system for accurate results
"""

import sys
import os
from pathlib import Path
import yaml

# Add src to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

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
                if file_config:
                    config.update(file_config)
            print(f"   ✅ Loaded: {config_file}")
        else:
            print(f"   ⚠️  Missing: {config_file}")
    
    return config

def get_available_models(config):
    """Extract available models from configuration"""
    models = []
    
    # Get models from llm_providers config
    if "llm_providers" in config:
        for provider, provider_config in config["llm_providers"].items():
            if "models" in provider_config:
                for model_name in provider_config["models"].keys():
                    models.append(model_name)
    
    # Also check primary_models from model_selection
    if "model_selection" in config and "primary_models" in config["model_selection"]:
        primary_models = config["model_selection"]["primary_models"]
        for model in primary_models:
            if model not in models:
                models.append(model)
    
    return models

def interactive_model_selection(available_models):
    """Interactive model selection"""
    print("***REMOVED***n🤖 Available Models:")
    print("-" * 30)
    
    for i, model in enumerate(available_models, 1):
        # Add descriptions for known models
        descriptions = {
            "deepseek-reasoner": "DeepSeek's reasoning model (cost-effective)",
            "deepseek-chat": "DeepSeek's chat model (very cost-effective)",
            "o4-mini-2025-04-16": "OpenAI's O4-mini (balanced performance)",
            "gpt-4o-mini-2024-07-18": "OpenAI's GPT-4o-mini (cost-effective, high performance)",
            "claude-sonnet-4-20250514": "Anthropic's Claude Sonnet 4 (high performance)",
            "claude-3-5-haiku-latest": "Anthropic's Claude Haiku (fast)"
        }
        
        description = descriptions.get(model, "")
        print(f"  {i}. {model}")
        if description:
            print(f"     {description}")
    
    while True:
        try:
            choice = int(input(f"***REMOVED***nSelect model (1-{len(available_models)}): ")) - 1
            if 0 <= choice < len(available_models):
                return available_models[choice]
            else:
                print(f"Invalid choice. Please enter 1-{len(available_models)}.")
        except ValueError:
            print("Please enter a valid number.")

def interactive_prompt_selection():
    """Interactive prompt length selection"""
    prompt_options = {
        "short": "Minimal, concise instructions (fastest)",
        "normal": "Balanced detail and clarity (recommended)", 
        "long": "Comprehensive, detailed instructions (current baseline)"
    }
    
    print("***REMOVED***n📝 Prompt Length Options:")
    print("-" * 35)
    
    for i, (length, description) in enumerate(prompt_options.items(), 1):
        print(f"  {i}. {length} - {description}")
    
    while True:
        try:
            choice = int(input(f"***REMOVED***nSelect prompt length (1-3): ")) - 1
            if 0 <= choice < 3:
                return list(prompt_options.keys())[choice]
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")

def interactive_task_selection():
    """Interactive task subset selection"""
    task_options = {
        "sample": "9 tasks (3 easy + 3 medium + 3 hard) - Quick test",
        "all": "90 tasks (complete evaluation) - Full analysis",
        "easy": "30 easy tasks only - Focus on simple tasks",
        "medium": "30 medium tasks only - Focus on complex tasks", 
        "hard": "30 hard tasks only - Focus on difficult tasks"
    }
    
    print("***REMOVED***n📊 Task Subset Options:")
    print("-" * 30)
    
    for i, (subset, description) in enumerate(task_options.items(), 1):
        print(f"  {i}. {subset} - {description}")
    
    while True:
        try:
            choice = int(input(f"***REMOVED***nSelect task subset (1-5): ")) - 1
            if 0 <= choice < 5:
                return list(task_options.keys())[choice]
            else:
                print("Invalid choice. Please enter 1-5.")
        except ValueError:
            print("Please enter a valid number.")

def estimate_runtime(model, prompt_length, task_subset):
    """Provide runtime estimates"""
    task_counts = {
        "sample": 9,
        "all": 90,
        "easy": 30,
        "medium": 30,
        "hard": 30
    }
    
    # Base time per task (seconds)
    base_times = {
        "short": 20,
        "normal": 35,
        "long": 50
    }
    
    # Model speed multipliers
    model_multipliers = {
        "deepseek-reasoner": 1.2,  # Slower due to reasoning
        "deepseek-chat": 0.8,      # Faster
        "o4-mini-2025-04-16": 1.0, # Baseline
        "gpt-4o-mini-2024-07-18": 0.9,  # Fast and efficient
        "claude-sonnet-4-20250514": 1.1,  # Slightly slower
        "claude-3-5-haiku-latest": 0.7    # Faster
    }
    
    num_tasks = task_counts.get(task_subset, 9)
    base_time = base_times.get(prompt_length, 35)
    multiplier = model_multipliers.get(model, 1.0)
    
    # Add time for judge evaluations (3 judges per task)
    judge_time = 10  # seconds per task for all judges
    
    total_seconds = (base_time * multiplier + judge_time) * num_tasks
    total_minutes = total_seconds / 60
    
    return total_minutes, num_tasks

def run_evaluation(config, model, prompt_length, task_subset):
    """Run the evaluation with selected parameters"""
    
    print(f"***REMOVED***n🚀 Starting Evaluation")
    print("=" * 50)
    print(f"   Model: {model}")
    print(f"   Prompt Length: {prompt_length}")
    print(f"   Task Subset: {task_subset}")
    
    # Estimate runtime
    estimated_minutes, num_tasks = estimate_runtime(model, prompt_length, task_subset)
    print(f"   Tasks: {num_tasks}")
    print(f"   Estimated Time: {estimated_minutes:.1f} minutes")
    print()
    
    # Confirm before starting
    confirm = input("Continue with evaluation? (y/n): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("Evaluation cancelled.")
        return False
    
    # Update config for single model
    config["llm_evaluation"]["models_to_test"] = [model]
    
    try:
        # Initialize evaluation runner with custom parameters
        runner = LLMEvaluationRunner(
            config, 
            use_mock=False, 
            prompt_length=prompt_length,
            task_subset=task_subset
        )
        
        # Run the evaluation
        runner.run_evaluation()
        
        print(f"***REMOVED***n🎉 Evaluation Complete!")
        print("=" * 30)
        
        # Print summary
        if hasattr(runner, 'results') and runner.results:
            total_tasks = len(runner.results)
            correct_tasks = sum(1 for r in runner.results if r['is_correct'])
            accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0
            
            print(f"📊 Results Summary:")
            print(f"   • Model: {model}")
            print(f"   • Prompt Length: {prompt_length}")
            print(f"   • Tasks Completed: {total_tasks}")
            print(f"   • Accuracy: {accuracy:.1%}")
            
            # Show output file location
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{model}_{prompt_length}_{task_subset}_{timestamp}.csv"
            print(f"   • Results File: experiments/llm_evaluation/performance_logs/{filename}")
        
        return True
        
    except Exception as e:
        print(f"***REMOVED***n❌ Evaluation Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main interactive function"""
    
    print("🏛️  Interactive LLM Evaluation System")
    print("=" * 50)
    print()
    print("This system provides unbiased evaluation using multiple judge models")
    print("to eliminate self-evaluation bias and provide accurate performance metrics.")
    print()
    
    # Load configuration
    print("📋 Loading Configuration...")
    config = load_config()
    
    # Get available models
    available_models = get_available_models(config)
    if not available_models:
        print("❌ No models found in configuration. Please check your config files.")
        return
    
    print(f"✅ Configuration loaded successfully")
    print()
    
    # Interactive selections
    selected_model = interactive_model_selection(available_models)
    selected_prompt = interactive_prompt_selection()
    selected_tasks = interactive_task_selection()
    
    # Run evaluation
    success = run_evaluation(config, selected_model, selected_prompt, selected_tasks)
    
    if success:
        print("***REMOVED***n💡 Next Steps:")
        print("   1. Review the results CSV file")
        print("   2. Check the bias analysis report")
        print("   3. Compare with other model/prompt combinations")
        print("   4. Run additional evaluations for comparison")
    else:
        print("***REMOVED***n🔧 Troubleshooting:")
        print("   1. Check API keys are set correctly")
        print("   2. Verify model names in configuration")
        print("   3. Ensure all config files are present")

if __name__ == "__main__":
    main()
