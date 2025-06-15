#!/usr/bin/env python3
"""
Fix Ground Truth Script

This script:
1. Generates corrected ground truth using actual system behavior
2. Compares old vs new ground truth to show differences
3. Runs a test evaluation to verify the fixes work

Usage:
    python scripts/fix_ground_truth.py
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation.correct_ground_truth_generator import CorrectGroundTruthGenerator


def compare_ground_truths(old_file: str, new_file: str):
    """
    Compare old and new ground truth files to show what changed.
    """
    print("***REMOVED***n" + "="*60)
    print("GROUND TRUTH COMPARISON")
    print("="*60)
    
    # Load both files
    with open(old_file, 'r') as f:
        old_gt = json.load(f)
    
    with open(new_file, 'r') as f:
        new_gt = json.load(f)
    
    # Create lookup dictionaries
    old_lookup = {item["task_id"]: item for item in old_gt}
    new_lookup = {item["task_id"]: item for item in new_gt}
    
    changes_found = 0
    
    for task_id in sorted(old_lookup.keys()):
        if task_id not in new_lookup:
            print(f"❌ Task {task_id} missing in new ground truth")
            continue
            
        old_answer = old_lookup[task_id]["baseline_answer"]
        new_answer = new_lookup[task_id]["baseline_answer"]
        
        # Check for differences
        if old_answer != new_answer:
            changes_found += 1
            print(f"***REMOVED***n🔄 CHANGED: {task_id}")
            print(f"   Query: {old_lookup[task_id]['query_instance']}")
            
            # Show specific differences
            if "gear_list" in old_answer and "gear_list" in new_answer:
                old_gears = old_answer["gear_list"]
                new_gears = new_answer["gear_list"]
                print(f"   OLD: {len(old_gears)} gears: {old_gears}")
                print(f"   NEW: {len(new_gears)} gears: {new_gears}")
                
            elif "assigned_printer" in old_answer and "assigned_printer" in new_answer:
                print(f"   OLD: Printer = {old_answer['assigned_printer']}")
                print(f"   NEW: Printer = {new_answer['assigned_printer']}")
                
            else:
                print(f"   OLD: {old_answer}")
                print(f"   NEW: {new_answer}")
    
    print(f"***REMOVED***n📊 SUMMARY: {changes_found} tasks had ground truth changes")
    
    if changes_found == 0:
        print("✅ No changes needed - ground truth was already correct!")
    else:
        print(f"✅ Fixed {changes_found} ground truth mismatches")


def test_specific_cases():
    """
    Test specific cases that were failing before the fix.
    """
    print("***REMOVED***n" + "="*60)
    print("TESTING SPECIFIC CASES")
    print("="*60)
    
    generator = CorrectGroundTruthGenerator()
    
    # Test the problematic PL1115 case
    print("***REMOVED***n🧪 Testing PL1115 (was showing 1 gear, should show 5):")
    result = generator.generate_easy_task_ground_truth("PL1115")
    print(f"   Result: {result['gear_count']} gears: {result['gear_list']}")
    
    if result['gear_count'] == 5 and "3DOR100033" in result['gear_list']:
        print("   ✅ FIXED: Now correctly finds all 5 gears including the original one")
    else:
        print("   ❌ ISSUE: Still not finding the expected gears")
    
    # Test a medium task
    print("***REMOVED***n🧪 Testing 3DOR100056 printer assignment:")
    result = generator.generate_medium_task_ground_truth("3DOR100056")
    print(f"   Result: Assigned to {result['assigned_printer']}")
    
    if result['assigned_printer'] == "Printer_6":
        print("   ✅ CORRECT: Printer assignment working")
    else:
        print("   ❌ ISSUE: Printer assignment not working")


def main():
    """
    Main function to fix ground truth and verify the changes.
    """
    print("🔧 FIXING GROUND TRUTH ISSUES")
    print("="*60)
    
    # Step 1: Generate corrected ground truth
    print("***REMOVED***n1️⃣ Generating corrected ground truth using actual system behavior...")
    generator = CorrectGroundTruthGenerator()
    corrected_gt = generator.save_ground_truth()
    
    # Step 2: Compare with old ground truth
    print("***REMOVED***n2️⃣ Comparing with original ground truth...")
    old_file = "data/ground_truth/baseline_answers.json"
    new_file = "data/ground_truth/baseline_answers_corrected.json"
    
    if os.path.exists(old_file):
        compare_ground_truths(old_file, new_file)
    else:
        print(f"⚠️  Original ground truth file not found at {old_file}")
    
    # Step 3: Test specific cases
    print("***REMOVED***n3️⃣ Testing specific problematic cases...")
    test_specific_cases()
    
    # Step 4: Instructions for next steps
    print("***REMOVED***n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("✅ Corrected ground truth generated successfully!")
    print(f"📁 File saved to: {new_file}")
    print("***REMOVED***n🚀 To test the fixes:")
    print("   1. Run LLM evaluation with corrected ground truth:")
    print("      python run_phase4_llm_evaluation.py --models deepseek-chat")
    print("   2. Check if success rate improves significantly")
    print("   3. Verify that corruption detection is now properly evaluated")
    
    print("***REMOVED***n📋 Expected improvements:")
    print("   • Easy tasks should now show much higher success rates")
    print("   • Tasks with detected corruption should be marked as correct")
    print("   • Overall failure rate should drop from 100% to reasonable levels")


if __name__ == "__main__":
    main()
