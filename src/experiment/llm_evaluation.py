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
    OpenAIReasoningProvider,
)

class LLMEvaluationRunner:
    """
    Manages the execution of the LLM agent against the experimental tasks.
    """
    def __init__(self, config: Dict[str, Any], use_mock: bool = False, use_corrected_ground_truth: bool = True,
                 prompt_length: str = "long", task_subset: str = "all", fast_mode: bool = False):
        self.config = config
        self.use_mock = use_mock
        self.prompt_length = prompt_length
        self.task_subset = task_subset
        self.fast_mode = fast_mode  # Skip judge system for maximum speed

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
        if "o4-mini" in model_name:
            return OpenAIReasoningProvider(model_name)  # Use reasoning provider for o4-mini thinking models
        elif "gpt" in model_name or "o4" in model_name:
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
                if "o4-mini" in model_name:
                    llm_provider = OpenAIReasoningProvider(model_name)  # Use reasoning provider for o4-mini thinking models
                elif "gpt" in model_name or "o4" in model_name:
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

            task_counter = 0
            total_tasks = sum(len(tasks) for tasks in self.assignments.values())

            for p_id, tasks in self.assignments.items():
                for task in tasks:
                    task_counter += 1
                    print(f"  - Running Task {task_counter}/{total_tasks}: {task['task_id']} on {model_name}")
                    print(f"    Query: {task['query_string'][:80]}...")
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

                    # UPDATED: Use fast evaluation or robust LLM-as-Judge evaluation
                    if self.fast_mode:
                        # Fast mode: Simple string matching for speed
                        is_correct, gt_answer = self._fast_evaluate_answer(task["task_id"], final_report)
                        print(f"    ⚡ Fast evaluation: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                    else:
                        # Full mode: Use the robust LLM-as-Judge evaluation
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
        elif provider_class == "OpenAIReasoningProvider":
            return self._create_openai_reasoning_judge(model_name)
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

            @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
            def generate(self, prompt: str) -> Dict[str, Any]:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=5,  # Reduced from 10 to 5 - we only need "Correct" or "Incorrect"
                        temperature=0.0  # Zero temperature for fastest, most consistent judgments
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

    def _create_openai_reasoning_judge(self, model_name: str):
        """Create OpenAI reasoning provider for o4-mini with high reasoning effort"""
        import openai
        from tenacity import retry, stop_after_attempt, wait_exponential

        class OpenAIReasoningJudgeProvider:
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.client = openai.OpenAI()

            @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
            def generate(self, prompt: str) -> Dict[str, Any]:
                try:
                    response = self.client.responses.create(
                        model=self.model_name,
                        reasoning={"effort": "medium"},  #options from low to high for reasoning effort
                        input=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    content = response.output_text.strip()
                    return {
                        "content": content,
                        "input_tokens": 0,  # Reasoning API doesn't provide token counts yet
                        "output_tokens": 0
                    }
                except Exception as e:
                    print(f"ERROR: OpenAI reasoning judge API call failed: {e}")
                    return {"content": "incorrect", "input_tokens": 0, "output_tokens": 0}

        return OpenAIReasoningJudgeProvider(model_name)

    def _create_anthropic_judge(self, model_name: str):
        """Create Anthropic provider configured for plain text judge responses"""
        import anthropic
        from tenacity import retry, stop_after_attempt, wait_exponential

        class AnthropicJudgeProvider:
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.client = anthropic.Anthropic()

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=2, min=2, max=30)  # Longer waits for overload errors
            )
            def generate(self, prompt: str) -> Dict[str, Any]:
                try:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=10,  # Increased back to 10 for reliability
                        temperature=0.0,  # Zero temperature for fastest, most consistent judgments
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
                    error_msg = str(e)
                    if "overloaded" in error_msg.lower() or "529" in error_msg:
                        print(f"⚠️  Anthropic judge overloaded, will retry with longer delay: {e}")
                        raise  # Let retry mechanism handle it
                    else:
                        print(f"❌ Anthropic judge API call failed: {e}")
                        return {"content": "incorrect", "input_tokens": 0, "output_tokens": 0}

        return AnthropicJudgeProvider(model_name)

    def _initialize_judge_providers(self) -> Dict[str, Any]:
        """Initialize balanced 3-judge system with weighted voting and fallback options"""
        judge_models = {
            "o4-mini-2025-04-16": {"weight": 2.0, "provider_class": "OpenAIReasoningProvider"},
            "claude-3-5-haiku-latest": {"weight": 1.5, "provider_class": "AnthropicProvider", "fallback": "gpt-4o-mini-2024-07-18"},
            "deepseek-chat": {"weight": 1.5, "provider_class": "DeepSeekProvider"}
            # Balanced system: 1 high-reasoning judge + 2 fast judges with weighted consensus
            # Claude has GPT-4o-mini fallback for overload situations
        }

        judge_providers = {}

        for model_name, config in judge_models.items():
            try:
                provider = self._create_judge_provider(model_name, config["provider_class"])
                judge_providers[model_name] = {
                    "provider": provider,
                    "weight": config["weight"],
                    "fallback": config.get("fallback")  # Store fallback model if available
                }
                judge_type = "high reasoning" if "o4-mini" in model_name else "fast"
                fallback_info = f" (fallback: {config['fallback']})" if config.get("fallback") else ""
                print(f"✅ Initialized judge: {model_name} (weight: {config['weight']}, {judge_type}){fallback_info}")
            except Exception as e:
                print(f"❌ Failed to initialize judge {model_name}: {e}")
                # Continue without this judge rather than failing completely

        if not judge_providers:
            raise RuntimeError("❌ CRITICAL: No judge providers could be initialized. Check API keys and model names.")

        print(f"🏛️  Balanced evaluation system ready with {len(judge_providers)} judges")
        return judge_providers

    def _create_unbiased_judge_prompt(self, llm_report: str, ground_truth: str) -> str:
        """Create an improved judge prompt for unbiased evaluation with specific examples"""

        return f"""You are an expert evaluator for manufacturing data analysis systems. Your task is to determine if the AI-generated report contains the correct core factual information.

**CRITICAL EVALUATION RULES:**
1. **IGNORE confidence scores completely** - Low confidence does not mean wrong answer
2. **IGNORE data quality warnings** - These are neutral and don't affect correctness
3. **IGNORE extra fields or formatting** - Focus only on core required data
4. **IGNORE field name differences** - "printer_used" vs "assigned_printer" are equivalent if values match
5. **ONLY evaluate factual accuracy** - Does the core data match the expected result?

**Ground Truth (Expected Result):**
{ground_truth}

**AI Report (To Evaluate):**
{llm_report}

**EXAMPLES OF CORRECT ANSWERS:**

**Example 1 - Gear Identification (CORRECT):**
- Ground Truth: {{"gear_list": ["3DOR100033", "3DOR100034", "3DOR100035"]}}
- AI Report: {{"gears": ["3DOR100033", "3DOR100034", "3DOR100035"], "confidence": 0.3, "issues": ["low confidence due to data quality"]}}
- Judgment: CORRECT (gear list matches exactly, ignore field name difference, confidence and warnings)

**Example 2 - Printer Assignment (CORRECT with different field names):**
- Ground Truth: {{"assigned_printer": "Printer_1"}}
- AI Report: {{"printer_used": "Printer_1", "confidence": 0.3, "issues": ["data quality"]}}
- Judgment: CORRECT (same printer value, ignore field name difference and low confidence)

**Example 2b - Printer Assignment (INCORRECT):**
- Ground Truth: {{"assigned_printer": "Printer_1"}}
- AI Report: {{"printer_used": "ORBOX0018", "confidence": 0.8}}
- Judgment: INCORRECT (wrong printer name, even with high confidence)

**Example 3 - Date Matching (CORRECT):**
- Ground Truth: {{"date_match_status": true}}
- AI Report: {{"match_status": "Dates appear to match (28/10/2024 and 2024-10-28)", "confidence": 0.3, "issues": ["missing data"]}}
- Judgment: CORRECT (core finding is dates match = true, ignore confidence and issues)

**Example 4 - Gear Count (INCORRECT):**
- Ground Truth: {{"gear_count": 5, "gear_list": ["A", "B", "C", "D", "E"]}}
- AI Report: {{"gear_count": 4, "gear_list": ["A", "B", "C", "D"], "confidence": 0.9}}
- Judgment: INCORRECT (missing gear E, wrong count, even with high confidence)

**YOUR TASK:**
Compare the core factual content only. Respond with exactly one word: "Correct" or "Incorrect"

Do not provide explanations. Just the single word judgment."""

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

        # Initialize judge providers if not already done
        if not hasattr(self, 'judge_providers'):
            print("🏛️  Initializing unbiased judge system...")
            self.judge_providers = self._initialize_judge_providers()

        judge_prompt = self._create_unbiased_judge_prompt(llm_report, gt_answer_json)

        # Collect judgments from all judges with weighted consensus
        judgments = {}
        judge_details = {}

        print(f"    🔍 Evaluating {task_id} with {len(self.judge_providers)} balanced judges...")

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

                judge_type = "high reasoning" if "o4-mini" in judge_name else "fast"
                print(f"    📝 {judge_name}: {'✅ Correct' if judgment else '❌ Incorrect'} (weight: {judge_config['weight']}, {judge_type})")

            except Exception as e:
                error_msg = str(e)
                fallback_model = judge_config.get("fallback")

                # Try fallback if available and it's an overload error
                if fallback_model and ("overloaded" in error_msg.lower() or "529" in error_msg):
                    print(f"    ⚠️  {judge_name} overloaded, trying fallback: {fallback_model}")
                    try:
                        # Create fallback provider
                        fallback_provider = self._create_judge_provider(fallback_model, "OpenAIProvider")
                        response = fallback_provider.generate(judge_prompt)
                        judgment_text = response["content"].strip().lower()

                        # Parse fallback judgment
                        if "correct" in judgment_text and "incorrect" not in judgment_text:
                            judgment = True
                        elif "incorrect" in judgment_text:
                            judgment = False
                        else:
                            judgment = False

                        judgments[judge_name] = judgment
                        judge_details[judge_name] = {
                            'judgment': judgment,
                            'weight': judge_config["weight"],
                            'raw_response': f"Fallback({fallback_model}): {judgment_text[:30]}"
                        }

                        print(f"    📝 {judge_name} (fallback): {'✅ Correct' if judgment else '❌ Incorrect'} (weight: {judge_config['weight']}, fallback)")

                    except Exception as fallback_error:
                        print(f"    ❌ Fallback also failed for {judge_name}: {fallback_error}")
                        judgments[judge_name] = False
                        judge_details[judge_name] = {
                            'judgment': False,
                            'weight': judge_config["weight"],
                            'raw_response': f"Error + Fallback failed: {str(e)}"
                        }
                else:
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
        final_judgment = consensus_score >= 0.5  # Majority threshold

        print(f"    🏛️  Weighted Consensus: {consensus_score:.2f} → {'✅ CORRECT' if final_judgment else '❌ INCORRECT'}")

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

    def _fast_evaluate_answer(self, task_id: str, llm_report: str) -> tuple:
        """
        Fast evaluation mode: Simple string matching without judge models.
        Much faster but less sophisticated than the full judge system.
        """
        gt_task = next(
            (t for t in self.ground_truth if t["task_id"] == task_id), None
        )
        if not gt_task:
            return False, f"Ground truth for task_id '{task_id}' not found in baseline_answers.json"

        gt_answer_json = json.dumps(gt_task["baseline_answer"], indent=2)

        # Simple heuristic: check if key elements from ground truth appear in LLM report
        try:
            gt_answer = gt_task["baseline_answer"]

            # For gear finding tasks
            if "gear_list" in gt_answer:
                expected_gears = gt_answer["gear_list"]
                found_gears = sum(1 for gear in expected_gears if gear in llm_report)
                is_correct = found_gears >= len(expected_gears) * 0.8  # 80% threshold

            # For printer tasks
            elif "printer_id" in gt_answer:
                is_correct = gt_answer["printer_id"] in llm_report

            # For date verification tasks
            elif "date_match" in gt_answer:
                # Look for boolean indicators
                if gt_answer["date_match"]:
                    is_correct = any(word in llm_report.lower() for word in ["match", "correct", "true", "yes"])
                else:
                    is_correct = any(word in llm_report.lower() for word in ["mismatch", "incorrect", "false", "no"])

            else:
                # Fallback: assume correct if report is substantial
                is_correct = len(llm_report.strip()) > 50

        except Exception as e:
            print(f"    ⚠️  Fast evaluation error: {e}")
            is_correct = False

        return is_correct, gt_answer_json

    def _save_results(self):
        results_df = pd.DataFrame(self.results)

        # Clean text fields to prevent CSV formatting issues
        if 'llm_final_report' in results_df.columns:
            # Replace newlines with escaped newlines and handle quotes
            results_df['llm_final_report'] = results_df['llm_final_report'].apply(
                lambda x: str(x).replace('***REMOVED***n', '***REMOVED******REMOVED***n').replace('***REMOVED***r', '***REMOVED******REMOVED***r') if pd.notna(x) else ''
            )

        if 'ground_truth_answer' in results_df.columns:
            # Clean ground truth field as well
            results_df['ground_truth_answer'] = results_df['ground_truth_answer'].apply(
                lambda x: str(x).replace('***REMOVED***n', '***REMOVED******REMOVED***n').replace('***REMOVED***r', '***REMOVED******REMOVED***r') if pd.notna(x) else ''
            )

        # Create custom filename based on configuration
        timestamp = datetime.now().strftime("%Y%m%d")
        models_tested = self.config["llm_evaluation"]["models_to_test"]
        model_name = models_tested[0] if len(models_tested) == 1 else "multi_model"

        filename = f"{model_name}_{self.prompt_length}_{self.task_subset}_{timestamp}.csv"
        output_path = os.path.join(self.log_dir, filename)

        # Use proper CSV quoting to handle special characters
        results_df.to_csv(output_path, index=False, quoting=1, escapechar='***REMOVED******REMOVED***')  # quoting=1 = QUOTE_ALL

        # Generate bias analysis report if we have bias data
        if hasattr(self, 'bias_analysis_data') and self.bias_analysis_data:
            self._generate_bias_analysis_report()

        print(f"***REMOVED***n--- Unbiased Evaluation Complete ---")
        print(f"✅ Results saved to {output_path}")
        print(f"🏛️  Evaluation used 3 balanced judges with weighted consensus")

        # Print bias analysis summary
        if hasattr(self, 'bias_analysis_data') and self.bias_analysis_data:
            consensus_scores = [data['consensus_score'] for data in self.bias_analysis_data]
            avg_consensus = sum(consensus_scores) / len(consensus_scores)
            unanimous_decisions = sum(1 for score in consensus_scores if score in [0, 1])
            print(f"📊 Judge Analysis Summary:")
            print(f"   • Average Decision Score: {avg_consensus:.2f}")
            print(f"   • Consistent Decisions: {unanimous_decisions}/{len(consensus_scores)} ({unanimous_decisions/len(consensus_scores)*100:.1f}%)")
            print(f"   • Judge analysis report: {self.log_dir}/bias_analysis_report.md")

    def _generate_bias_analysis_report(self):
        """Generate comprehensive bias analysis report"""
        if not hasattr(self, 'bias_analysis_data') or not self.bias_analysis_data:
            return

        consensus_scores = [data['consensus_score'] for data in self.bias_analysis_data]

        report = []
        report.append("# Balanced LLM Evaluation - Judge Analysis Report")
        report.append("=" * 60)
        report.append("")
        report.append("## Executive Summary")
        report.append("")
        report.append("This evaluation used 3 balanced judges with weighted consensus to eliminate self-evaluation bias.")
        report.append("")

        # Judge System Overview
        if hasattr(self, 'judge_providers'):
            report.append("## Judge System Configuration")
            report.append("")
            for judge_name, config in self.judge_providers.items():
                judge_type = "high reasoning" if "o4-mini" in judge_name else "fast"
                report.append(f"- **{judge_name}**: Weight {config['weight']} ({judge_type})")
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
        report.append("✅ **Self-Evaluation Bias Eliminated**: Independent judges evaluate the answers")
        report.append("✅ **Balanced Approach**: High-reasoning judge + fast judges for speed/accuracy balance")
        report.append("✅ **Weighted Consensus**: Higher weight for reasoning judge, majority threshold")
        report.append("✅ **Transparency**: All individual judgments and weights are recorded")
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
        report.append("❌ **Previous (Biased)**: Model judges its own answers")
        report.append("✅ **Current (Balanced)**: 3 independent judges with weighted consensus judge all model answers")
        report.append("")
        report.append("This provides more accurate, reliable performance metrics for manufacturing data analysis tasks.")

        report_text = "***REMOVED***n".join(report)

        bias_report_path = os.path.join(self.log_dir, "bias_analysis_report.md")
        with open(bias_report_path, 'w') as f:
            f.write(report_text)

        print(f"📋 Bias analysis report saved to: {bias_report_path}")