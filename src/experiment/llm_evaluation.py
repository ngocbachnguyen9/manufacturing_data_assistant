# In src/experiment/llm_evaluation.py

import pandas as pd
import os
import json
import time
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
    def __init__(self, config: Dict[str, Any], use_mock: bool = False, use_corrected_ground_truth: bool = True):
        self.config = config
        self.use_mock = use_mock
        self.assignments = self._load_json("experiments/human_study/participant_assignments.json")

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
                    # UPDATED: Capture the full execution trace
                    execution_trace = agent.run_query(task["query_string"])
                    final_report = execution_trace["final_report"]
                    reconciliation = execution_trace["reconciliation_summary"]

                    end_time = time.time()
                    completion_time = round(end_time - start_time, 2)
                    cost_summary = cost_tracker.get_summary()

                    # UPDATED: Use the robust LLM-as-Judge evaluation
                    is_correct, gt_answer = self._evaluate_answer(
                        task["task_id"], final_report, llm_provider
                    )

                    # UPDATED: Log the rich behavioral data
                    self.results.append(
                        {
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
                    )
        self._save_results()

    def _evaluate_answer(
        self, task_id: str, llm_report: str, llm_provider: Any
    ) -> tuple:
        """
        UPDATED: Uses an LLM as a judge to evaluate the correctness of the report.
        """
        gt_task = next(
            (t for t in self.ground_truth if t["task_id"] == task_id), None
        )
        if not gt_task:
            # Return a more informative error message
            return False, f"Ground truth for task_id '{task_id}' not found in baseline_answers.json"

        gt_answer_json = json.dumps(gt_task["baseline_answer"], indent=2)

        judge_prompt = f"""
        You are an impartial judge evaluating manufacturing data analysis reports. Your task is to determine if the 'Generated Report' correctly answers the query based on the 'Ground Truth Answer'.

        **Ground Truth Answer:**
        {gt_answer_json}

        **Generated Report:**
        {llm_report}

        **Evaluation Guidelines:**
        1. **For Gear Lists**: The report is CORRECT if it contains ALL gears from the ground truth. Extra gears are acceptable if they belong to the same order.
        2. **For Printer Assignment**: The report is CORRECT if it identifies the correct printer name, even if the data contains corruption (spaces, missing characters).
        3. **For Date Verification**: The report is CORRECT if it correctly identifies whether dates match or not.
        4. **For Data Quality Issues**: Reports with low confidence due to detected corruption or missing data should be considered CORRECT if they appropriately flag the issues.
        5. **For Corruption Recovery**: Reports that successfully work through data corruption using fuzzy matching or alternative data sources should be considered CORRECT even if confidence is reduced.

        **Decision Rules:**
        - If the report contains all required information from ground truth: CORRECT
        - If the report correctly identifies data quality issues with appropriate low confidence: CORRECT
        - If the report successfully recovers correct information despite data corruption: CORRECT
        - If the report provides wrong factual information: INCORRECT

        **Special Cases for Corrupted Data:**
        - Q1 (Whitespace): Report is CORRECT if it finds the right answer despite extra spaces
        - Q2 (Missing Characters): Report is CORRECT if it finds the right answer despite missing characters
        - Q3 (Missing Relationships): Report is CORRECT if it appropriately reports missing data with low confidence

        Your response MUST be a single word: either 'Correct' or 'Incorrect'.
        """

        # Use the LLM to make the judgment
        judge_response = llm_provider.generate(judge_prompt)
        judgment = judge_response["content"].strip().lower()

        return judgment == "correct", gt_answer_json

    def _save_results(self):
        results_df = pd.DataFrame(self.results)
        output_path = os.path.join(
            self.log_dir, "llm_performance_results.csv"
        )
        results_df.to_csv(output_path, index=False)
        print(f"***REMOVED***n--- Evaluation Complete ---")
        print(f"Results for all models saved to {output_path}")