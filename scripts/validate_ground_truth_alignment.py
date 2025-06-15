#!/usr/bin/env python3
"""
Ground Truth Alignment Validation Script

This script validates the alignment between:
1. data/ground_truth/baseline_answers.json
2. experiments/human_study/participant_assignments.json

It checks for:
- Task ID consistency
- Query string matching
- Complexity level alignment
- Missing tasks
- Data type consistency
- Answer format validation

Usage:
    python scripts/validate_ground_truth_alignment.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Any


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in {file_path}: {e}")
        sys.exit(1)


def extract_assignments_data(assignments: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Extract task data from participant assignments."""
    tasks = {}
    for participant_id, task_list in assignments.items():
        for task in task_list:
            task_id = task["task_id"]
            tasks[task_id] = {
                "participant_id": participant_id,
                "complexity": task["complexity"],
                "quality_condition": task["quality_condition"],
                "query_string": task["query_string"],
                "dataset_path": task["dataset_path"]
            }
    return tasks


def extract_ground_truth_data(ground_truth: List[Dict]) -> Dict[str, Dict]:
    """Extract task data from ground truth."""
    tasks = {}
    for entry in ground_truth:
        task_id = entry["task_id"]
        tasks[task_id] = {
            "complexity_level": entry["complexity_level"],
            "query_instance": entry["query_instance"],
            "baseline_answer": entry["baseline_answer"]
        }
    return tasks


def validate_task_ids(assignments_tasks: Dict, ground_truth_tasks: Dict) -> List[str]:
    """Validate task ID consistency."""
    issues = []
    
    assignment_ids = set(assignments_tasks.keys())
    ground_truth_ids = set(ground_truth_tasks.keys())
    
    # Check for missing tasks
    missing_in_gt = assignment_ids - ground_truth_ids
    missing_in_assignments = ground_truth_ids - assignment_ids
    
    if missing_in_gt:
        issues.append(f"❌ Tasks in assignments but missing in ground truth: {sorted(missing_in_gt)}")
    
    if missing_in_assignments:
        issues.append(f"❌ Tasks in ground truth but missing in assignments: {sorted(missing_in_assignments)}")
    
    if not missing_in_gt and not missing_in_assignments:
        issues.append(f"✅ Task ID alignment: All {len(assignment_ids)} tasks present in both files")
    
    return issues


def validate_query_strings(assignments_tasks: Dict, ground_truth_tasks: Dict) -> List[str]:
    """Validate query string consistency."""
    issues = []
    mismatches = []
    
    common_tasks = set(assignments_tasks.keys()) & set(ground_truth_tasks.keys())
    
    for task_id in sorted(common_tasks):
        assignment_query = assignments_tasks[task_id]["query_string"]
        gt_query = ground_truth_tasks[task_id]["query_instance"]
        
        if assignment_query != gt_query:
            mismatches.append({
                "task_id": task_id,
                "assignment_query": assignment_query,
                "ground_truth_query": gt_query
            })
    
    if mismatches:
        issues.append(f"❌ Query string mismatches found in {len(mismatches)} tasks:")
        for mismatch in mismatches[:5]:  # Show first 5 mismatches
            issues.append(f"   {mismatch['task_id']}:")
            issues.append(f"     Assignment: '{mismatch['assignment_query']}'")
            issues.append(f"     Ground Truth: '{mismatch['ground_truth_query']}'")
        if len(mismatches) > 5:
            issues.append(f"   ... and {len(mismatches) - 5} more mismatches")
    else:
        issues.append(f"✅ Query string alignment: All {len(common_tasks)} queries match perfectly")
    
    return issues


def validate_complexity_levels(assignments_tasks: Dict, ground_truth_tasks: Dict) -> List[str]:
    """Validate complexity level consistency."""
    issues = []
    mismatches = []
    
    common_tasks = set(assignments_tasks.keys()) & set(ground_truth_tasks.keys())
    
    for task_id in sorted(common_tasks):
        assignment_complexity = assignments_tasks[task_id]["complexity"]
        gt_complexity = ground_truth_tasks[task_id]["complexity_level"]
        
        if assignment_complexity != gt_complexity:
            mismatches.append({
                "task_id": task_id,
                "assignment_complexity": assignment_complexity,
                "ground_truth_complexity": gt_complexity
            })
    
    if mismatches:
        issues.append(f"❌ Complexity level mismatches found in {len(mismatches)} tasks:")
        for mismatch in mismatches[:5]:  # Show first 5 mismatches
            issues.append(f"   {mismatch['task_id']}: {mismatch['assignment_complexity']} vs {mismatch['ground_truth_complexity']}")
        if len(mismatches) > 5:
            issues.append(f"   ... and {len(mismatches) - 5} more mismatches")
    else:
        issues.append(f"✅ Complexity level alignment: All {len(common_tasks)} complexity levels match")
    
    return issues


def validate_answer_formats(ground_truth_tasks: Dict) -> List[str]:
    """Validate ground truth answer format consistency."""
    issues = []
    
    # Group tasks by complexity
    easy_tasks = []
    medium_tasks = []
    hard_tasks = []
    
    for task_id, task_data in ground_truth_tasks.items():
        complexity = task_data["complexity_level"]
        answer = task_data["baseline_answer"]
        
        if complexity == "easy":
            easy_tasks.append((task_id, answer))
        elif complexity == "medium":
            medium_tasks.append((task_id, answer))
        elif complexity == "hard":
            hard_tasks.append((task_id, answer))
    
    # Validate easy task format (should have gear-related fields)
    easy_format_issues = []
    for task_id, answer in easy_tasks:
        if "gear_list" not in answer and "error" not in answer:
            easy_format_issues.append(task_id)
    
    if easy_format_issues:
        issues.append(f"❌ Easy tasks with incorrect format: {easy_format_issues}")
    else:
        issues.append(f"✅ Easy task format: All {len(easy_tasks)} tasks have correct gear-related format")
    
    # Validate medium task format (should have printer-related fields)
    medium_format_issues = []
    for task_id, answer in medium_tasks:
        if "assigned_printer" not in answer and "error" not in answer:
            medium_format_issues.append(task_id)
    
    if medium_format_issues:
        issues.append(f"❌ Medium tasks with incorrect format: {medium_format_issues}")
    else:
        issues.append(f"✅ Medium task format: All {len(medium_tasks)} tasks have correct printer-related format")
    
    # Validate hard task format (should have date-related fields)
    hard_format_issues = []
    for task_id, answer in hard_tasks:
        if "date_match_status" not in answer and "error" not in answer:
            hard_format_issues.append(task_id)
    
    if hard_format_issues:
        issues.append(f"❌ Hard tasks with incorrect format: {hard_format_issues}")
    else:
        issues.append(f"✅ Hard task format: All {len(hard_tasks)} tasks have correct date-related format")
    
    return issues


def generate_summary_statistics(assignments_tasks: Dict, ground_truth_tasks: Dict) -> List[str]:
    """Generate summary statistics."""
    stats = []
    
    # Task count statistics
    stats.append(f"📊 SUMMARY STATISTICS:")
    stats.append(f"   Total tasks in assignments: {len(assignments_tasks)}")
    stats.append(f"   Total tasks in ground truth: {len(ground_truth_tasks)}")
    
    # Complexity distribution in assignments
    complexity_counts = {}
    for task_data in assignments_tasks.values():
        complexity = task_data["complexity"]
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    stats.append(f"   Complexity distribution (assignments):")
    for complexity, count in sorted(complexity_counts.items()):
        stats.append(f"     {complexity}: {count} tasks")
    
    # Quality condition distribution
    quality_counts = {}
    for task_data in assignments_tasks.values():
        quality = task_data["quality_condition"]
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    stats.append(f"   Quality condition distribution:")
    for quality, count in sorted(quality_counts.items()):
        stats.append(f"     {quality}: {count} tasks")
    
    return stats


def main():
    """Main validation function."""
    print("🔍 GROUND TRUTH ALIGNMENT VALIDATION")
    print("=" * 60)
    
    # Load files
    print("***REMOVED***n📁 Loading files...")
    assignments = load_json_file("experiments/human_study/participant_assignments.json")
    ground_truth = load_json_file("data/ground_truth/baseline_answers.json")
    
    # Extract task data
    assignments_tasks = extract_assignments_data(assignments)
    ground_truth_tasks = extract_ground_truth_data(ground_truth)
    
    print(f"   Loaded {len(assignments_tasks)} tasks from participant assignments")
    print(f"   Loaded {len(ground_truth_tasks)} tasks from ground truth")
    
    # Run validations
    all_issues = []
    
    print("***REMOVED***n🔍 Validating task ID alignment...")
    all_issues.extend(validate_task_ids(assignments_tasks, ground_truth_tasks))
    
    print("***REMOVED***n🔍 Validating query string alignment...")
    all_issues.extend(validate_query_strings(assignments_tasks, ground_truth_tasks))
    
    print("***REMOVED***n🔍 Validating complexity level alignment...")
    all_issues.extend(validate_complexity_levels(assignments_tasks, ground_truth_tasks))
    
    print("***REMOVED***n🔍 Validating answer format consistency...")
    all_issues.extend(validate_answer_formats(ground_truth_tasks))
    
    # Generate summary
    print("***REMOVED***n📊 Generating summary statistics...")
    all_issues.extend(generate_summary_statistics(assignments_tasks, ground_truth_tasks))
    
    # Print results
    print("***REMOVED***n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    for issue in all_issues:
        print(issue)
    
    # Final assessment
    error_count = sum(1 for issue in all_issues if issue.startswith("❌"))
    success_count = sum(1 for issue in all_issues if issue.startswith("✅"))
    
    print(f"***REMOVED***n🎯 FINAL ASSESSMENT:")
    print(f"   ✅ Successful validations: {success_count}")
    print(f"   ❌ Issues found: {error_count}")
    
    if error_count == 0:
        print(f"***REMOVED***n🎉 EXCELLENT: Perfect alignment between ground truth and participant assignments!")
        return 0
    else:
        print(f"***REMOVED***n⚠️  WARNING: {error_count} alignment issues found that need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
