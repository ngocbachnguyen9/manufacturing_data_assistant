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
        self.clean_ids = self._get_clean_ids() # NEW: Pool of clean IDs
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
    
    def _get_clean_ids(self) -> Dict[str, List[str]]:
        """NEW: Creates pools of IDs that were NOT corrupted."""
        all_dirty_orders = set(self.dirty_ids.get("Q1", []) + self.dirty_ids.get("Q2", []) + self.dirty_ids.get("Q3", []))
        
        clean_orders = [oid for oid in self.valid_ids["easy"] if oid not in all_dirty_orders]
        # For medium tasks, we assume all gears are clean unless explicitly targeted
        clean_gears = self.valid_ids["medium"][:]
        
        return {"easy": clean_orders, "hard": clean_orders.copy(), "medium": clean_gears}

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
        UPDATED: Generates assignments ensuring tasks match their data quality.
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
            # Clone the valid IDs so each participant gets a fresh pool
            available_ids = {
                "easy": self.valid_ids["easy"][:],
                "medium": self.valid_ids["medium"][:],
                "hard": self.valid_ids["hard"][:],
            }

            # NEW: Create disposable pools of IDs for this participant
            id_pools = {
                "Q0": self.clean_ids.copy(),
                "Q1": {"easy": self.dirty_ids.get("Q1", []), "hard": self.dirty_ids.get("Q1", [])},
                "Q2": {"easy": self.dirty_ids.get("Q2", []), "hard": self.dirty_ids.get("Q2", [])},
                "Q3": {"easy": self.dirty_ids.get("Q3", []), "hard": self.dirty_ids.get("Q3", [])},
            }

            participant_tasks = []

            for i in range(len(quality_list)):
                # Map the single-letter code to the full complexity name
                letter = complexity_list[i].lower()  # 'e','m','h'
                mapping = {"e": "easy", "m": "medium", "h": "hard"}
                if letter not in mapping:
                    raise ValueError(f"Unknown complexity code: {complexity_list[i]}")
                complexity = mapping[letter]

                # NEW: Select entity ID from the correct pool (clean or dirty)
                quality = quality_list[i]

                if quality == "Q0":
                    pool = id_pools[quality][complexity]
                else: # For Q1, Q2, Q3, medium tasks still use clean IDs
                    pool = id_pools[quality].get(complexity, self.clean_ids[complexity])

                if not pool:
                    raise ValueError(f"Not enough unique IDs for {p_id}, {quality}, {complexity} task.")
                entity_id = pool.pop(random.randrange(len(pool)))

                # Construct the task object
                task_id = f"{p_id}_task_{i+1}"
                query_template = self.task_templates[complexity]["description"]
                # Replace ENTITY_ID placeholder with the actual ID through accessing dataset
                query_string = query_template.format(ENTITY_ID=entity_id)
               
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