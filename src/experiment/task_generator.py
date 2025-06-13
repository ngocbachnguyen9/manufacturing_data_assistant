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

    def __init__(self, config: Dict[str,Any], dirty_ids: Dict[str,List[str]]):
        """
        Initializes the generator with experiment configuration.
        """
        self.config = config
        self.dirty_ids = dirty_ids # <— store the incoming dirty‐ID map
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

        # Hard tasks use Order IDs
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
        
        # Fallback: Use orders for easy tasks if packing lists are missing
        packing_list_dir = "data/generated_documents/packing_lists"
        packing_lists = []
        if os.path.exists(packing_list_dir):
            for f in os.listdir(packing_list_dir):
                if f.startswith("PackingList-") and f.endswith(".pdf"):
                    pl_id = f.replace("PackingList-", "").replace(".pdf", "")
                    packing_lists.append(pl_id)
        else:
            print(f"Warning: Packing list directory not found at {packing_list_dir}")
        
        if not packing_lists:
            print("Warning: No packing lists found. Using orders for easy tasks")
            packing_lists = orders.copy()

        return {"easy": sorted(packing_lists),  "hard": orders, "medium": gears}
    
    def _get_clean_ids(self) -> Dict[str, List[str]]:
        # Gather all corrupted IDs across Q1–Q3
        all_dirty = set()
        for qc in ("Q1","Q2","Q3"):
            all_dirty.update(self.dirty_ids.get(qc, []))

        # Split into orders vs gears by intersecting with baseline lists
        dirty_orders = set(self.valid_ids["easy"]).intersection(all_dirty)
        dirty_gears  = set(self.valid_ids["medium"]).intersection(all_dirty)

        # Filter them out of the baseline valid IDs
        clean_orders = [
            o for o in self.valid_ids["easy"] if o not in dirty_orders
        ]
        clean_gears = [
            g for g in self.valid_ids["medium"] if g not in dirty_gears
        ]

        return {
            "easy": clean_orders,
            "hard": clean_orders.copy(),
            "medium": clean_gears,
        }

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

        # Build the ID pools dynamically based on ID prefixes
        id_pools_template = {"Q0": self._get_clean_ids()}
        for qc in ("Q1", "Q2", "Q3"):
            dirty = self.dirty_ids.get(qc, [])
            # Partition dirty IDs by complexity based on their prefix
            id_pools_template[qc] = {
                "easy": [i for i in dirty if i.startswith(("E","PL"))],
                "medium": [i for i in dirty if i.startswith(("M", "3DOR"))],
                "hard": [i for i in dirty if i.startswith(("H", "ORBOX", "PL"))],
            }

        for p_id, patterns in self.participants.items():
            # Give each participant their own copy of the ID pools
            id_pools = json.loads(json.dumps(id_pools_template))

            quality_list = self._create_task_list("quality", patterns["quality_pattern"])
            complexity_list = self._create_task_list("prompt", patterns["prompt_pattern"])
            
            participant_tasks = []
            for i in range(len(quality_list)):
                complexity = {"e": "easy", "m": "medium", "h": "hard"}[complexity_list[i].lower()]
                quality = quality_list[i]

                # Select entity ID from the correct, isolated pool
                pool = id_pools[quality][complexity]
                if not pool:
                    raise ValueError(f"Not enough unique IDs for {p_id}, {quality}, {complexity} task.")
                
                entity_id = pool.pop(random.randrange(len(pool)))

                # Use the entity ID and an index to create a stable, lookup-friendly task_id
                task_id = f"{complexity}_{entity_id}_{i}"
                
                query_template = self.task_templates[complexity]["description"]
                query_string = query_template.format(ENTITY_ID=entity_id)
            
                if quality == "Q0":
                    dataset_path = f"data/experimental_datasets/Q0_baseline"
                else:
                    dataset_path = f"data/experimental_datasets/{quality}_dataset"

                participant_tasks.append({
                    "task_id": task_id,
                    "participant_id": p_id,
                    "complexity": complexity,
                    "quality_condition": quality,
                    "query_string": query_string,
                    "dataset_path": dataset_path,
                })
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