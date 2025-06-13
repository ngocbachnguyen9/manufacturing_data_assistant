import json
import re
import pandas as pd

def extract_entity_id(query_string: str) -> str:
    """Extracts the entity ID from a query string using regex."""
    # Patterns for different ID types
    patterns = {
        "PL": r"Packing List (PL***REMOVED***w+)",
        "3DOR": r"Part (3DOR***REMOVED***w+)",
        "ORBOX": r"Order (ORBOX***REMOVED***w+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, query_string)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract entity ID from query: {query_string}")


def pl_to_orbox(pl_id: str) -> str:
    """Converts a Packing List ID to an Order Box ID based on the rule."""
    # e.g., PL1011 -> ORBOX0011, PL1115 -> ORBOX0115
    numeric_part = pl_id.replace("PL", "")
    if len(numeric_part) == 4 and numeric_part.startswith("1"):
        # Handle 4-digit numbers like 1011, 1115
        return f"ORBOX0{numeric_part[1:]}"
    # Add other rules if they exist
    return f"ORBOX{numeric_part.zfill(5)}"


# 1. Load all necessary files
with open("participant_assignments.json", "r") as f:
    assignments = json.load(f)

with open("baseline_answers_old.json", "r") as f:
    old_ground_truth = json.load(f)

# 2. Create the Entity-to-Answer Map from the old ground truth
entity_answer_map = {}
for item in old_ground_truth:
    entity_id = extract_entity_id(item["query_instance"])
    # Apply the PL->ORBOX mapping for easy tasks
    if entity_id.startswith("PL"):
        entity_id = pl_to_orbox(entity_id)
    entity_answer_map[entity_id] = item["baseline_answer"]

# 3. Synthesize new answers for entities not in the old ground truth
#    (Based on manual analysis of the provided data images)
synthesized_answers = {
    # Easy Tasks (PL -> ORBOX)
    "ORBOX0115": { # From PL1115
        "packing_list_id": "PL1115", "gear_count": 1, "gear_list": ["3DOR100033"],
    },
    "ORBOX0122": { # From PL1122
        "packing_list_id": "PL1122", "gear_count": 1, "gear_list": ["3DOR100071"],
    },
    "ORBOX0121": { # From PL1121
        "packing_list_id": "PL1121", "gear_count": 4, "gear_list": ["3DOR100061", "3DOR100062", "3DOR100065", "3DOR100068"],
    },
    "ORBOX0012": { # From PL1012
        "packing_list_id": "PL1012", "gear_count": 4, "gear_list": ["3DOR100041", "3DOR100043", "3DOR100045", "3DOR100047"],
    },
     "ORBOX0017": { # From PL1017
        "packing_list_id": "PL1017", "gear_count": 5, "gear_list": ["3DOR100021", "3DOR100022", "3DOR100023", "3DOR100024", "3DOR100030"],
    },
    # Medium Tasks
    "3DOR100091": {"part_id": "3DOR100091", "assigned_printer": "Printer_1"},
    "3DOR100056": {"part_id": "3DOR100056", "assigned_printer": "Printer_6"},
    "3DOR100098": {"part_id": "3DOR100098", "assigned_printer": "Printer_8"},
    "3DOR100017": {"part_id": "3DOR100017", "assigned_printer": "Printer_7"},
    "3DOR100041": {"part_id": "3DOR100041", "assigned_printer": "Printer_1"},
    "3DOR100012": {"part_id": "3DOR100012", "assigned_printer": "Printer_2"},
    "3DOR100026": {"part_id": "3DOR100026", "assigned_printer": "Printer_6"},
    "3DOR100061": {"part_id": "3DOR100061", "assigned_printer": "Printer_1"},
    "3DOR100095": {"part_id": "3DOR100095", "assigned_printer": "Printer_5"},
    "3DOR100054": {"part_id": "3DOR100054", "assigned_printer": "Printer_4"},
    # Hard Tasks
    "ORBOX0017": {
        "product_id": "ORBOX0017", "certificate_date": "2024-10-28", "warehouse_arrival_date": "2024-10-28", "date_match_status": True,
    },
    "ORBOX0015": {
        "product_id": "ORBOX0015", "certificate_date": "2024-10-28", "warehouse_arrival_date": "2024-10-28", "date_match_status": True,
    },
    "ORBOX0117": {
        "product_id": "ORBOX0117", "certificate_date": "2024-10-28", "warehouse_arrival_date": "2024-10-28", "date_match_status": True,
    },
}
# Merge synthesized answers into the main map
entity_answer_map.update(synthesized_answers)


# 4. Generate the new ground truth list
new_ground_truth = []
for p_id, tasks in assignments.items():
    for task in tasks:
        task_id = task["task_id"]
        query = task["query_string"]
        entity_id = extract_entity_id(query)

        # Apply mapping for easy tasks
        lookup_id = pl_to_orbox(entity_id) if entity_id.startswith("PL") else entity_id

        if lookup_id in entity_answer_map:
            new_ground_truth.append({
                "task_id": task_id,
                "complexity_level": task["complexity"],
                "query_instance": query,
                "baseline_answer": entity_answer_map[lookup_id]
            })
        else:
            print(f"WARNING: No ground truth found for entity {lookup_id} (from task {task_id})")

# 5. Save the new file
with open("baseline_answers_aligned.json", "w") as f:
    json.dump(new_ground_truth, f, indent=2)

print("Aligned ground truth file 'baseline_answers_aligned.json' has been created.")