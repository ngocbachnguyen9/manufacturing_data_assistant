import pandas as pd
import os
import json
import random
from typing import Dict, List, Any


class TaskGenerator:
    """
    Generates a counterbalanced set of tasks for each participant based on
    the 3x3 pattern-pair design.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the generator with experiment configuration.
        """
        self.config = config
        self.participants = config["human_study"]["participant_matrix"]
        self.quality_patterns = config["human_study"]["quality_patterns"]
        self.prompt_patterns = config["human_study"]["prompt_patterns"]
        self.task_templates = config["task_complexity"]
        self.valid_ids = self._get_valid_ids()
        random.seed(config["experiment"]["random_seed"])

    def _get_valid_ids(self) -> Dict[str, List[str]]:
        """
        Loads valid entity IDs from the Q0 baseline dataset to create
        concrete task queries.
        """
        print("Loading valid IDs from Q0 baseline dataset...")
        baseline_path = "data/experimental_datasets/Q0_baseline"
        rel_df = pd.read_csv(
            os.path.join(baseline_path, "relationship_data.csv")
        )

        # Easy/Hard tasks use Order IDs
        orders = sorted(
            rel_df[rel_df["parent"].str.startswith("ORBOX", na=False)][
                "parent"
            ].unique()
        )
        # Medium tasks use Gear IDs
        gears = sorted(
            rel_df[rel_df["child"].str.startswith("3DOR", na=False)][
                "child"
            ].unique()
        )

        return {"easy": orders, "hard": orders.copy(), "medium": gears}

    def _create_task_list(
        self, pattern_type: str, pattern_name: str
    ) -> List[str]:
        """Creates a shuffled list of conditions based on a pattern."""
        patterns = (
            self.quality_patterns
            if pattern_type == "quality"
            else self.prompt_patterns
        )
        pattern = patterns[pattern_name]
        task_list = []
        for key, count in pattern.items():
            task_list.extend([key] * count)
        random.shuffle(task_list)
        return task_list

    def generate_all_assignments(self) -> Dict[str, List[Dict]]:
        """
        Generates and returns the full assignment dictionary for all participants.
        """
        all_assignments = {}
        print("Generating task assignments for all participants...")

        for p_id, patterns in self.participants.items():
            quality_list = self._create_task_list(
                "quality", patterns["quality_pattern"]
            )
            complexity_list = self._create_task_list(
                "prompt", patterns["prompt_pattern"]
            )

            participant_tasks = []
            for i in range(len(quality_list)):
                complexity = complexity_list[i].lower()  # E -> easy
                quality = quality_list[i]

                # Select a random ID and remove it to ensure variety
                if not self.valid_ids[complexity]:
                    raise ValueError(f"Not enough unique IDs for {complexity} tasks.")
                entity_id = self.valid_ids[complexity].pop(
                    random.randrange(len(self.valid_ids[complexity]))
                )

                # Construct the task object
                task_id = f"{p_id}_task_{i+1}"
                query_template = self.task_templates[complexity]["description"]
                query_string = query_template.replace(
                    "{...}", entity_id
                )  # Simple replace
                dataset_path = (
                    f"data/experimental_datasets/{quality}_baseline"
                    if quality == "Q0"
                    else f"data/experimental_datasets/{quality}_dataset"
                )

                participant_tasks.append(
                    {
                        "task_id": task_id,
                        "participant_id": p_id,
                        "complexity": complexity,
                        "quality_condition": quality,
                        "query_string": query_string,
                        "dataset_path": dataset_path,
                    }
                )
            all_assignments[p_id] = participant_tasks
            print(f"  - Generated 10 tasks for participant {p_id}")

        return all_assignments

    def save_assignments(self, assignments: Dict):
        """Saves the generated assignments to a JSON file."""
        output_path = "experiments/human_study/participant_assignments.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(assignments, f, indent=2)
        print(f"***REMOVED***nParticipant assignments saved to {output_path}")