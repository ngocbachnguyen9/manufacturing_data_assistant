import pandas as pd
import os
import json
import time
from typing import Dict, List, Any

class StudyPlatform:
    """
    A CLI platform to guide a researcher through conducting a human study session.
    """

    def __init__(self, assignments_path: str):
        if not os.path.exists(assignments_path):
            raise FileNotFoundError(
                f"Assignments file not found: {assignments_path}"
            )
        with open(assignments_path, "r") as f:
            self.assignments = json.load(f)
        self.log_dir = "experiments/human_study/session_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def run_session(self):
        """Starts and manages a single participant session."""
        print("--- Manufacturing Data Assistant: Human Study Platform ---")
        participant_id = input(
            f"Enter Participant ID (e.g., P1, P2...): "
        ).upper()

        if participant_id not in self.assignments:
            print(f"Error: Participant ID '{participant_id}' not found.")
            return

        tasks = self.assignments[participant_id]
        results = []
        print(f"***REMOVED***nStarting session for {participant_id}. Total tasks: {len(tasks)}")

        for i, task in enumerate(tasks):
            os.system("cls||clear")  # Clear screen
            print(f"--- Task {i+1} of {len(tasks)} for {participant_id} ---")
            print(f"Complexity: {task['complexity'].upper()}")
            print(f"Data Quality: {task['quality_condition']}")
            print(f"***REMOVED***nDataset to use: {task['dataset_path']}")
            print(f"***REMOVED***nQuery: ***REMOVED***"{task['query_string']}***REMOVED***"")
            print("-" * 50)

            input("Press Enter to start the timer and begin the task...")
            start_time = time.time()

            answer = input("Enter participant's final answer here: ")
            end_time = time.time()
            completion_time = round(end_time - start_time)

            accuracy = input("Enter accuracy score (1 for correct, 0 for incorrect): ")
            notes = input("Enter any notes on process or error detection: ")

            results.append(
                {
                    "task_id": task["task_id"],
                    "participant_id": participant_id,
                    "completion_time_sec": completion_time,
                    "accuracy": int(accuracy),
                    "participant_answer": answer,
                    "notes": notes,
                }
            )
            print(f"Task completed in {completion_time} seconds. Result logged.")
            input("***REMOVED***nPress Enter to proceed to the next task...")

        self._save_results(participant_id, results)

    def _save_results(self, participant_id: str, results: List[Dict]):
        """Saves the collected session results to a CSV file."""
        results_df = pd.DataFrame(results)
        output_path = os.path.join(self.log_dir, f"{participant_id}_results.csv")
        results_df.to_csv(output_path, index=False)
        print(f"***REMOVED***nSession for {participant_id} complete. Results saved to {output_path}")