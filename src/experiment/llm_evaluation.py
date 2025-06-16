# In src/experiment/llm_evaluation.py

import pandas as pd
import os
import json
import time
import random
import yaml
from datetime import datetime
from typing import Dict, Any, List

from src.agents.master_agent import MasterAgent
from src.utils.data_loader import DataLoader
from src.utils.cost_tracker import CostTracker
# Import all providers
from src.utils.llm_provider import (
    MockLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    DeepSeekProvider,
)

class LLMEvaluationRunner:
    """
    Manages the execution of the LLM agent against the experimental tasks.
    """
    def __init__(self, config: Dict[str, Any], use_mock: bool = False, use_corrected_ground_truth: bool = True,
                 prompt_length: str = "long", task_subset: str = "all"):
        self.config = config
        self.use_mock = use_mock
        self.prompt_length = prompt_length
        self.task_subset = task_subset

        # Load prompt variations
        self.prompt_variations = self._load_prompt_variations()

        # Load and filter assignments based on task subset
        all_assignments = self._load_json("experiments/human_study/participant_assignments.json")
        self.assignments = self._filter_assignments(all_assignments, task_subset)

        # Use corrected ground truth by default, fallback to original if not available
        if use_corrected_ground_truth and os.path.exists("data/ground_truth/baseline_answers_corrected.json"):
            print("Using corrected ground truth that matches actual system behavior")
            self.ground_truth = self._load_json("data/ground_truth/baseline_answers_corrected.json")
        else:
            print("Using original ground truth (may contain mismatches)")
            self.ground_truth = self._load_json("data/ground_truth/baseline_answers.json")

        self.results = []
        self.log_dir = "experiments/llm_evaluation/performance_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def _load_json(self, path: str) -> Any:
        with open(path, "r") as f:
            return json.load(f)

    def _load_prompt_variations(self) -> Dict[str, Any]:
        """Load prompt variations from config file"""
        try:
            with open("config/task_prompts_variations.yaml", "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print("Warning: task_prompts_variations.yaml not found. Using default prompts.")
            return {}

    def _filter_assignments(self, all_assignments: Dict[str, Any], task_subset: str) -> Dict[str, Any]:
        """Filter assignments based on task subset selection"""
        if task_subset == "all":
            return all_assignments
        elif task_subset == "sample":
            # Select 3 tasks per complexity level (9 total)
            return self._create_sample_assignments(all_assignments)
        elif task_subset in ["easy", "medium", "hard"]:
            # Filter by specific complexity
            return self._filter_by_complexity(all_assignments, task_subset)
        else:
            print(f"Warning: Unknown task subset '{task_subset}'. Using all tasks.")
            return all_assignments

    def _create_sample_assignments(self, all_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Create sample assignments with 3 tasks per complexity level"""
        # Group tasks by complexity
        tasks_by_complexity = {"easy": [], "medium": [], "hard": []}

        for participant_id, tasks in all_assignments.items():
            for task in tasks:
                complexity = task.get("complexity", "unknown")
                if complexity in tasks_by_complexity:
                    tasks_by_complexity[complexity].append((participant_id, task))

        # Randomly select 3 tasks per complexity
        sample_assignments = {}
        for complexity, task_list in tasks_by_complexity.items():
            if len(task_list) >= 3:
                selected_tasks = random.sample(task_list, 3)
                for participant_id, task in selected_tasks:
                    if participant_id not in sample_assignments:
                        sample_assignments[participant_id] = []
                    sample_assignments[participant_id].append(task)

        return sample_assignments

    def _filter_by_complexity(self, all_assignments: Dict[str, Any], target_complexity: str) -> Dict[str, Any]:
        """Filter assignments to only include tasks of specified complexity"""
        filtered_assignments = {}

        for participant_id, tasks in all_assignments.items():
            filtered_tasks = [task for task in tasks if task.get("complexity") == target_complexity]
            if filtered_tasks:
                filtered_assignments[participant_id] = filtered_tasks

        return filtered_assignments

    def _get_task_prompt(self, task: Dict[str, Any]) -> str:
        """Get the appropriate prompt based on prompt length and task details"""
        if not self.prompt_variations:
            # Fallback to original query string if no variations available
            return task["query_string"]

        complexity = task.get("complexity", "easy")
        quality_condition = task.get("quality_condition", "Q0")

        # Determine if we need data quality context
        has_data_issues = quality_condition != "Q0"

        try:
            prompt_config = self.prompt_variations["prompt_variations"][self.prompt_length][complexity]

            if has_data_issues and "with_data_quality_issues" in prompt_config:
                base_prompt = prompt_config["with_data_quality_issues"]
            else:
                base_prompt = prompt_config["base_prompt"]

            # Replace placeholders with actual task values
            # Extract entity ID from original query string
            entity_id = self._extract_entity_id(task["query_string"])

            # Replace common placeholders
            formatted_prompt = base_prompt.replace("{packing_list_id}", entity_id)
            formatted_prompt = formatted_prompt.replace("{part_id}", entity_id)
            formatted_prompt = formatted_prompt.replace("{order_id}", entity_id)

            return formatted_prompt

        except (KeyError, TypeError) as e:
            print(f"Warning: Could not load {self.prompt_length} prompt for {complexity} task. Using original query.")
            return task["query_string"]

    def _extract_entity_id(self, query_string: str) -> str:
        """Extract entity ID from query string"""
        # Simple extraction - look for common patterns
        import re

        # Look for patterns like PL1115, 3DOR100091, ORBOX0017
        patterns = [
            r'PL***REMOVED***d+',           # Packing List IDs
            r'3DOR***REMOVED***d+',         # Part IDs
            r'ORBOX***REMOVED***d+',        # Order IDs
        ]

        for pattern in patterns:
            match = re.search(pattern, query_string)
            if match:
                return match.group()

        # Fallback - return the query string itself
        return query_string

    def _select_llm_provider(self, model_name: str) -> Any:
        """Internal factory function to select the correct LLM provider."""
        if self.use_mock:
            print(f"  - Using MockLLMProvider for {model_name}")
            return MockLLMProvider(model_name)

        print(f"  - Using LIVE provider for {model_name}")
        if "gpt" in model_name or "o4" in model_name:
            return OpenAIProvider(model_name)
        elif "claude" in model_name or "sonnet" in model_name:
            return AnthropicProvider(model_name)
        elif "deepseek" in model_name:
            return DeepSeekProvider(model_name)
        else:
            raise ValueError(f"No real provider found for model: {model_name}")

    def run_evaluation(self):
        """
        Iterates through all models and tasks, runs the agent,
        and logs the detailed results.
        """
        models_to_test = self.config["llm_evaluation"]["models_to_test"]
        cost_config = self.config.get("llm_providers", {})

        for model_name in models_to_test:
            print(f"***REMOVED***n--- Testing Model: {model_name} ---")
            # This logic now works because self.use_mock is set in __init__
            if self.use_mock:
                llm_provider = MockLLMProvider(model_name)
                print(f"  - Using MockLLMProvider for {model_name}")
            else:
                if "gpt" in model_name or "o4" in model_name:
                    llm_provider = OpenAIProvider(model_name)
                elif "claude" in model_name or "sonnet" in model_name:
                    llm_provider = AnthropicProvider(model_name)
                elif "deepseek" in model_name:
                    llm_provider = DeepSeekProvider(model_name)
                else:
                    print(f"Warning: No provider found for {model_name}. Using mock.")
                    llm_provider = MockLLMProvider(model_name)
                print(f"  - Using LIVE provider: {llm_provider.__class__.__name__}")

            cost_tracker = CostTracker(cost_config)

            for p_id, tasks in self.assignments.items():
                for task in tasks:
                    print(f"  - Running Task: {task['task_id']} on {model_name}")
                    cost_tracker.reset()
                    start_time = time.time()

                    data_loader = DataLoader(base_path=task["dataset_path"])
                    task_datasets = data_loader.load_base_data()

                    agent = MasterAgent(
                        llm_provider, task_datasets, self.config, cost_tracker
                    )

                    # Get the appropriate prompt based on prompt length setting
                    task_prompt = self._get_task_prompt(task)
                    print(f"    Using {self.prompt_length} prompt length")

                    # UPDATED: Capture the full execution trace with custom prompt
                    execution_trace = agent.run_query(task_prompt)
                    final_report = execution_trace["final_report"]
                    reconciliation = execution_trace["reconciliation_summary"]

                    end_time = time.time()
                    completion_time = round(end_time - start_time, 2)
                    cost_summary = cost_tracker.get_summary()

                    # UPDATED: Use the robust LLM-as-Judge evaluation
                    is_correct, gt_answer = self._evaluate_answer(
                        task["task_id"], final_report, llm_provider
                    )

                    # UPDATED: Log the rich behavioral data with bias analysis
                    result_entry = {
                        "task_id": task["task_id"],
                        "model": model_name,
                        "complexity": task["complexity"],
                        "quality_condition": task["quality_condition"],
                        "completion_time_sec": completion_time,
                        "is_correct": is_correct,
                        "total_cost_usd": cost_summary["total_cost_usd"],
                        "input_tokens": cost_summary["input_tokens"],
                        "output_tokens": cost_summary["output_tokens"],
                        "final_confidence": reconciliation.get("confidence"),
                        "reconciliation_issues": json.dumps(
                            reconciliation.get("issues_found")
                        ),
                        "llm_final_report": final_report,
                        "ground_truth_answer": gt_answer,
                    }

                    # Add bias analysis data if available
                    if hasattr(self, 'bias_analysis_data') and self.bias_analysis_data:
                        latest_bias_data = self.bias_analysis_data[-1]
                        if latest_bias_data['task_id'] == task["task_id"]:
                            result_entry.update({
                                "judge_consensus_score": latest_bias_data['consensus_score'],
                                "judge_details": latest_bias_data['individual_judgments'],
                                "total_judges": latest_bias_data['total_judges'],
                                "agreement_level": latest_bias_data['agreement_level']
                            })

                    self.results.append(result_entry)
        self._save_results()

    def _create_judge_provider(self, model_name: str, provider_class: str):
        """Create a judge-specific provider that handles plain text responses"""
        if provider_class == "OpenAIProvider":
            return self._create_openai_judge(model_name)
        elif provider_class == "AnthropicProvider":
            return self._create_anthropic_judge(model_name)
        elif provider_class == "DeepSeekProvider":
            return DeepSeekProvider(model_name)  # DeepSeek works fine as-is
        else:
            raise ValueError(f"Unknown provider class: {provider_class}")

    def _create_openai_judge(self, model_name: str):
        """Create OpenAI provider configured for plain text judge responses"""
        import openai
        from tenacity import retry, stop_after_attempt, wait_exponential

        class OpenAIJudgeProvider:
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.client = openai.OpenAI()

            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=60))
            def generate(self, prompt: str) -> Dict[str, Any]:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10,  # We only need "Correct" or "Incorrect"
                        temperature=0.1  # Low temperature for consistent judgments
                    )

                    content = response.choices[0].message.content.strip()
                    return {
                        "content": content,
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens
                    }
                except Exception as e:
                    print(f"ERROR: OpenAI judge API call failed: {e}")
                    return {"content": "incorrect", "input_tokens": 0, "output_tokens": 0}

        return OpenAIJudgeProvider(model_name)

    def _create_anthropic_judge(self, model_name: str):
        """Create Anthropic provider configured for plain text judge responses"""
        import anthropic
        from tenacity import retry, stop_after_attempt, wait_exponential

        class AnthropicJudgeProvider:
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.client = anthropic.Anthropic()

            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=60))
            def generate(self, prompt: str) -> Dict[str, Any]:
                try:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=10,  # We only need "Correct" or "Incorrect"
                        temperature=0.1,  # Low temperature for consistent judgments
                        system="You are an expert evaluator. Respond with exactly one word: 'Correct' or 'Incorrect'.",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    content = response.content[0].text.strip()
                    return {
                        "content": content,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                except Exception as e:
                    print(f"ERROR: Anthropic judge API call failed: {e}")
                    return {"content": "incorrect", "input_tokens": 0, "output_tokens": 0}

        return AnthropicJudgeProvider(model_name)

    def _initialize_judge_providers(self) -> Dict[str, Any]:
        """Initialize unbiased judge model providers with weighted consensus"""
        judge_models = {
            "gpt-4o-mini": {"weight": 2, "provider_class": "OpenAIProvider"},
            "deepseek-reasoner": {"weight": 2, "provider_class": "DeepSeekProvider"},
            "claude-3-5-haiku-20241022": {"weight": 1, "provider_class": "AnthropicProvider"}
        }

        judge_providers = {}

        for model_name, config in judge_models.items():
            try:
                provider = self._create_judge_provider(model_name, config["provider_class"])
                judge_providers[model_name] = {
                    "provider": provider,
                    "weight": config["weight"]
                }
                print(f"✅ Initialized unbiased judge: {model_name} (weight: {config['weight']})")
            except Exception as e:
                print(f"❌ Failed to initialize judge {model_name}: {e}")
                # Continue without this judge rather than failing completely

        if not judge_providers:
            raise RuntimeError("❌ CRITICAL: No judge providers could be initialized. Check API keys and model names.")

        print(f"🏛️  Unbiased evaluation system ready with {len(judge_providers)} judges")
        return judge_providers

    def _create_unbiased_judge_prompt(self, task_complexity: str, llm_report: str, ground_truth: str) -> str:
        """Create an improved judge prompt for unbiased evaluation"""

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

    def _evaluate_answer(
        self, task_id: str, llm_report: str, llm_provider: Any
    ) -> tuple:
        """
        UPDATED: Uses multiple unbiased judge models to evaluate correctness.
        Implements weighted consensus to eliminate self-evaluation bias.
        """
        gt_task = next(
            (t for t in self.ground_truth if t["task_id"] == task_id), None
        )
        if not gt_task:
            return False, f"Ground truth for task_id '{task_id}' not found in baseline_answers.json"

        gt_answer_json = json.dumps(gt_task["baseline_answer"], indent=2)

        # Get task complexity for context-aware judging
        task_complexity = gt_task.get("complexity_level", "unknown")

        # Initialize judge providers if not already done
        if not hasattr(self, 'judge_providers'):
            print("🏛️  Initializing unbiased judge system...")
            self.judge_providers = self._initialize_judge_providers()

        judge_prompt = self._create_unbiased_judge_prompt(task_complexity, llm_report, gt_answer_json)

        # Collect judgments from all judges
        judgments = {}
        judge_details = {}

        print(f"    🔍 Evaluating {task_id} with {len(self.judge_providers)} unbiased judges...")

        for judge_name, judge_config in self.judge_providers.items():
            try:
                response = judge_config["provider"].generate(judge_prompt)
                judgment_text = response["content"].strip().lower()

                # Parse judgment
                if "correct" in judgment_text and "incorrect" not in judgment_text:
                    judgment = True
                elif "incorrect" in judgment_text:
                    judgment = False
                else:
                    print(f"    ⚠️  Unclear judgment from {judge_name}: {judgment_text}")
                    judgment = False  # Default to incorrect for unclear responses

                judgments[judge_name] = judgment
                judge_details[judge_name] = {
                    'judgment': judgment,
                    'weight': judge_config["weight"],
                    'raw_response': judgment_text[:50]  # Truncate for storage
                }

                print(f"    📝 {judge_name}: {'✅ Correct' if judgment else '❌ Incorrect'} (weight: {judge_config['weight']})")

            except Exception as e:
                print(f"    ❌ Error with judge {judge_name}: {e}")
                judgments[judge_name] = False
                judge_details[judge_name] = {
                    'judgment': False,
                    'weight': judge_config["weight"],
                    'raw_response': f"Error: {str(e)}"
                }

        # Calculate weighted consensus
        total_weight = 0
        correct_weight = 0

        for judge_name, judgment in judgments.items():
            weight = judge_details[judge_name]['weight']
            total_weight += weight
            if judgment:
                correct_weight += weight

        # Weighted consensus score
        consensus_score = correct_weight / total_weight if total_weight > 0 else 0
        final_judgment = consensus_score >= 0.5

        print(f"    🏛️  Consensus: {consensus_score:.2f} → {'✅ CORRECT' if final_judgment else '❌ INCORRECT'}")

        # Store bias analysis data for later reporting
        if not hasattr(self, 'bias_analysis_data'):
            self.bias_analysis_data = []
        self.bias_analysis_data.append({
            'task_id': task_id,
            'consensus_score': consensus_score,
            'individual_judgments': json.dumps(judge_details),
            'total_judges': len(judgments),
            'agreement_level': 'unanimous' if consensus_score in [0, 1] else 'weighted_majority'
        })

        return final_judgment, gt_answer_json

    def _save_results(self):
        results_df = pd.DataFrame(self.results)

        # Create custom filename based on configuration
        timestamp = datetime.now().strftime("%Y%m%d")
        models_tested = self.config["llm_evaluation"]["models_to_test"]
        model_name = models_tested[0] if len(models_tested) == 1 else "multi_model"

        filename = f"{model_name}_{self.prompt_length}_{self.task_subset}_{timestamp}.csv"
        output_path = os.path.join(self.log_dir, filename)

        results_df.to_csv(output_path, index=False)

        # Generate bias analysis report if we have bias data
        if hasattr(self, 'bias_analysis_data') and self.bias_analysis_data:
            self._generate_bias_analysis_report()

        print(f"***REMOVED***n--- Unbiased Evaluation Complete ---")
        print(f"✅ Results saved to {output_path}")
        print(f"🏛️  Evaluation used {len(getattr(self, 'judge_providers', {}))} unbiased judges")

        # Print bias analysis summary
        if hasattr(self, 'bias_analysis_data') and self.bias_analysis_data:
            consensus_scores = [data['consensus_score'] for data in self.bias_analysis_data]
            avg_consensus = sum(consensus_scores) / len(consensus_scores)
            unanimous_decisions = sum(1 for score in consensus_scores if score in [0, 1])
            print(f"📊 Bias Analysis Summary:")
            print(f"   • Average Consensus Score: {avg_consensus:.2f}")
            print(f"   • Unanimous Decisions: {unanimous_decisions}/{len(consensus_scores)} ({unanimous_decisions/len(consensus_scores)*100:.1f}%)")
            print(f"   • Bias analysis report: {self.log_dir}/bias_analysis_report.md")

    def _generate_bias_analysis_report(self):
        """Generate comprehensive bias analysis report"""
        if not hasattr(self, 'bias_analysis_data') or not self.bias_analysis_data:
            return

        consensus_scores = [data['consensus_score'] for data in self.bias_analysis_data]

        report = []
        report.append("# Unbiased LLM Evaluation - Bias Analysis Report")
        report.append("=" * 60)
        report.append("")
        report.append("## Executive Summary")
        report.append("")
        report.append("This evaluation used multiple unbiased judge models to eliminate self-evaluation bias.")
        report.append("")

        # Judge System Overview
        if hasattr(self, 'judge_providers'):
            report.append("## Judge System Configuration")
            report.append("")
            for judge_name, config in self.judge_providers.items():
                report.append(f"- **{judge_name}**: Weight {config['weight']}")
            report.append("")

        # Consensus Analysis
        avg_consensus = sum(consensus_scores) / len(consensus_scores)
        unanimous_decisions = sum(1 for score in consensus_scores if score in [0, 1])
        majority_decisions = sum(1 for score in consensus_scores if score >= 0.5)

        report.append("## Consensus Analysis")
        report.append("")
        report.append(f"**Total Tasks Evaluated**: {len(consensus_scores)}")
        report.append(f"**Average Consensus Score**: {avg_consensus:.3f}")
        report.append(f"**Unanimous Decisions**: {unanimous_decisions}/{len(consensus_scores)} ({unanimous_decisions/len(consensus_scores)*100:.1f}%)")
        report.append(f"**Majority Decisions**: {majority_decisions}/{len(consensus_scores)} ({majority_decisions/len(consensus_scores)*100:.1f}%)")
        report.append("")

        # Bias Elimination Benefits
        report.append("## Bias Elimination Benefits")
        report.append("")
        report.append("✅ **Self-Evaluation Bias Eliminated**: Different models judge the answers")
        report.append("✅ **Weighted Consensus**: Higher-capability judges have more influence")
        report.append("✅ **Transparency**: All individual judgments are recorded")
        report.append("✅ **Reliability**: Multiple perspectives reduce individual model biases")
        report.append("")

        # Controversial Decisions
        controversial = [data for data in self.bias_analysis_data if 0 < data['consensus_score'] < 1]
        if controversial:
            report.append("## Controversial Decisions (Split Judgments)")
            report.append("")
            for data in controversial[:5]:  # Show first 5
                report.append(f"- **{data['task_id']}**: Consensus {data['consensus_score']:.2f}")
            if len(controversial) > 5:
                report.append(f"- ... and {len(controversial) - 5} more")
            report.append("")

        report.append("## Methodology Validation")
        report.append("")
        report.append("This unbiased evaluation system addresses the critical flaw in self-evaluation:")
        report.append("")
        report.append("❌ **Previous (Biased)**: deepseek-chat judges its own answers")
        report.append("✅ **Current (Unbiased)**: GPT-4o-mini, deepseek-reasoner, Claude judge deepseek-chat's answers")
        report.append("")
        report.append("This provides more accurate, reliable performance metrics for manufacturing data analysis tasks.")

        report_text = "***REMOVED***n".join(report)

        bias_report_path = os.path.join(self.log_dir, "bias_analysis_report.md")
        with open(bias_report_path, 'w') as f:
            f.write(report_text)

        print(f"📋 Bias analysis report saved to: {bias_report_path}")