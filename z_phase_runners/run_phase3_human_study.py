import yaml
import os
import json
from typing import Dict, Any
from argparse import ArgumentParser

from src.experiment.task_generator import TaskGenerator
from src.human_study.study_platform import StudyPlatform

def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    p = ArgumentParser()
    p.add_argument("--pilot", action="store_true",
                   help="Run only pilot participants")
    p.add_argument("--seed", type=int,
                   help="Override random seed (for pilot or full run)")
    args = p.parse_args()

    # 1) Load config
    cfg = load_config("config/experiment_config.yaml")
    # --- load the dirty‐ID map produced in Phase 1 ---
    dirty_path = "experiments/human_study/dirty_ids.json"
    if not os.path.exists(dirty_path):
        raise FileNotFoundError(f"Dirty‐ID map not found: {dirty_path}")
    dirty_ids = json.load(open(dirty_path))

    # 2) Decide seed & participant set
    if args.seed is not None:
        seed = args.seed
    elif args.pilot:
        seed = cfg["experiment"].get("pilot_random_seed", cfg["experiment"]["random_seed"])
    else:
        seed = cfg["experiment"]["random_seed"]
    cfg["experiment"]["random_seed"] = seed

    all_parts = cfg["human_study"]["participant_matrix"].keys()
    if args.pilot:
        # define your pilot IDs here or in config under human_study['pilot_participants']
        pilot_ids = cfg["human_study"].get("pilot_participants", list(all_parts)[:2])
        participants_to_run = pilot_ids
        out_file = "experiments/human_study/pilot_assignments.json"
    else:
        participants_to_run = all_parts
        out_file = "experiments/human_study/participant_assignments.json"

    # 3) Generate all assignments (same logic, seeded) using dirty IDs
    tg = TaskGenerator(cfg, dirty_ids)
    full_assignments = tg.generate_all_assignments()

    # 4) Slice as needed and write JSON
    subset = {pid: full_assignments[pid] for pid in participants_to_run}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Wrote {len(subset)} participant(s) → {out_file}")

    # 5) Launch the CLI only on that file
    input("Press ENTER to start study session…")
    sp = StudyPlatform(out_file)
    sp.run_session()

if __name__=="__main__":
    main()