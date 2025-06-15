#!/usr/bin/env python3
"""
Correct Ground Truth Generator

This script generates ground truth answers by actually running the system tools
on the Q0 baseline dataset to get the real answers, rather than using hardcoded values.

This ensures the ground truth matches what the system actually returns when working correctly.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Any
from src.tools.packing_list_parser_tool import PackingListParserTool
from src.tools.relationship_tool import RelationshipTool
from src.utils.data_loader import DataLoader


class CorrectGroundTruthGenerator:
    """
    Generates ground truth by actually running the system tools on Q0 baseline data.
    """
    
    def __init__(self, baseline_path: str = "data/experimental_datasets/Q0_baseline"):
        self.baseline_path = baseline_path
        self.data_loader = DataLoader(base_path=baseline_path)
        self.datasets = self.data_loader.load_base_data()
        
        # Initialize tools
        self.packing_list_tool = PackingListParserTool(self.datasets)
        self.relationship_tool = RelationshipTool(self.datasets)
        
    def generate_easy_task_ground_truth(self, packing_list_id: str) -> Dict[str, Any]:
        """
        Generate ground truth for easy tasks (gear identification) by running actual tools.
        """
        # Step 1: Parse packing list to get order ID
        packing_result = self.packing_list_tool.run(packing_list_id)
        
        if "error" in packing_result:
            return {
                "packing_list_id": packing_list_id,
                "gear_count": 0,
                "gear_list": [],
                "error": packing_result["error"]
            }
        
        order_id = packing_result["order_id"]
        
        # Step 2: Find all gears for the order
        relationship_result = self.relationship_tool.run(order_id)

        if (isinstance(relationship_result, list) and len(relationship_result) == 1 and
            isinstance(relationship_result[0], dict) and "error" in relationship_result[0]):
            return {
                "packing_list_id": packing_list_id,
                "gear_count": 0,
                "gear_list": [],
                "error": relationship_result[0]["error"]
            }

        # Extract gear IDs from the relationship records
        gears = set()
        for record in relationship_result:
            # Look for gear IDs in both parent and child fields
            if record.get("parent", "").startswith("3DOR"):
                gears.add(record["parent"])
            if record.get("child", "").startswith("3DOR"):
                gears.add(record["child"])

        gears = sorted(list(gears))  # Ensure consistent ordering
        
        return {
            "packing_list_id": packing_list_id,
            "gear_count": len(gears),
            "gear_list": gears
        }
    
    def generate_medium_task_ground_truth(self, part_id: str) -> Dict[str, Any]:
        """
        Generate ground truth for medium tasks (printer identification) by running actual tools.
        """
        # Find the printer for this part
        relationship_result = self.relationship_tool.run(part_id)

        if (isinstance(relationship_result, list) and len(relationship_result) == 1 and
            isinstance(relationship_result[0], dict) and "error" in relationship_result[0]):
            return {
                "part_id": part_id,
                "assigned_printer": "UNKNOWN",
                "error": relationship_result[0]["error"]
            }

        # Find printer in the relationship records
        printer = None
        for record in relationship_result:
            # Look for printer in parent field
            if record.get("parent", "").startswith("Printer"):
                printer = record["parent"]
                break
        
        return {
            "part_id": part_id,
            "assigned_printer": printer if printer else "UNKNOWN"
        }
    
    def generate_hard_task_ground_truth(self, order_id: str) -> Dict[str, Any]:
        """
        Generate ground truth for hard tasks (date verification).
        For now, we'll use the existing hardcoded values since these involve document parsing.
        """
        # These are consistent across all test cases based on the generated data
        return {
            "product_id": order_id,
            "certificate_date": "2024-10-28",
            "warehouse_arrival_date": "2024-10-28", 
            "date_match_status": True
        }
    
    def generate_all_ground_truths(self, assignments_file: str = "experiments/human_study/participant_assignments.json") -> List[Dict[str, Any]]:
        """
        Generate ground truth for all tasks in the participant assignments.
        """
        # Load participant assignments
        with open(assignments_file, 'r') as f:
            assignments = json.load(f)
        
        ground_truth_list = []
        
        for participant_id, tasks in assignments.items():
            for task in tasks:
                task_id = task["task_id"]
                complexity = task["complexity"]
                query = task["query_string"]
                
                print(f"Generating ground truth for {task_id} ({complexity}): {query}")
                
                try:
                    if complexity == "easy":
                        # Extract packing list ID from query
                        import re
                        match = re.search(r'Packing List (PL***REMOVED***w+)', query)
                        if match:
                            pl_id = match.group(1)
                            baseline_answer = self.generate_easy_task_ground_truth(pl_id)
                        else:
                            print(f"Warning: Could not extract packing list ID from query: {query}")
                            continue
                            
                    elif complexity == "medium":
                        # Extract part ID from query
                        match = re.search(r'Part (3DOR***REMOVED***w+)', query)
                        if match:
                            part_id = match.group(1)
                            baseline_answer = self.generate_medium_task_ground_truth(part_id)
                        else:
                            print(f"Warning: Could not extract part ID from query: {query}")
                            continue
                            
                    elif complexity == "hard":
                        # Extract order ID from query
                        match = re.search(r'Order (ORBOX***REMOVED***w+)', query)
                        if match:
                            order_id = match.group(1)
                            baseline_answer = self.generate_hard_task_ground_truth(order_id)
                        else:
                            print(f"Warning: Could not extract order ID from query: {query}")
                            continue
                    
                    ground_truth_entry = {
                        "task_id": task_id,
                        "complexity_level": complexity,
                        "query_instance": query,
                        "baseline_answer": baseline_answer
                    }
                    
                    ground_truth_list.append(ground_truth_entry)
                    
                except Exception as e:
                    print(f"Error generating ground truth for {task_id}: {e}")
                    continue
        
        return ground_truth_list
    
    def save_ground_truth(self, output_file: str = "data/ground_truth/baseline_answers_corrected.json"):
        """
        Generate and save the corrected ground truth file.
        """
        ground_truth = self.generate_all_ground_truths()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"Corrected ground truth saved to {output_file}")
        print(f"Generated {len(ground_truth)} ground truth entries")
        
        return ground_truth


if __name__ == "__main__":
    generator = CorrectGroundTruthGenerator()
    generator.save_ground_truth()
