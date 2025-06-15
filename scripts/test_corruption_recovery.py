#!/usr/bin/env python3
"""
Test Corruption Recovery Script

This script tests the system's ability to work through data corruption
using fuzzy matching and alternative data sources.

Usage:
    python scripts/test_corruption_recovery.py
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.relationship_tool import RelationshipTool
from src.utils.data_loader import DataLoader


def test_q1_whitespace_corruption():
    """Test Q1 dataset (whitespace injection) corruption recovery."""
    print("***REMOVED***n" + "="*60)
    print("TESTING Q1 WHITESPACE CORRUPTION RECOVERY")
    print("="*60)
    
    # Load Q1 corrupted data
    data_loader = DataLoader(base_path="data/experimental_datasets/Q1_dataset")
    datasets = data_loader.load_base_data()
    
    relationship_tool = RelationshipTool(datasets)
    
    # Test cases with known whitespace corruption
    test_cases = [
        ("Printer_6", "   Printer_6"),  # Leading spaces
        ("Printer_7", "  Printer_7  "),  # Leading and trailing spaces
    ]
    
    for clean_id, corrupted_id in test_cases:
        print(f"***REMOVED***n🧪 Testing: '{corrupted_id}' (should find '{clean_id}')")
        
        # Try exact match first (should fail)
        exact_result = relationship_tool.run(corrupted_id, fuzzy_enabled=False)
        print(f"   Exact match: {'✅ Found' if not any('error' in r for r in exact_result) else '❌ Failed'}")
        
        # Try fuzzy match (should succeed)
        fuzzy_result = relationship_tool.run(corrupted_id, fuzzy_enabled=True, threshold=0.7)
        if not any('error' in r for r in fuzzy_result):
            print(f"   Fuzzy match: ✅ Found {len(fuzzy_result)} relationships")
            if fuzzy_result and 'fuzzy_match_confidence' in fuzzy_result[0]:
                print(f"   Confidence: {fuzzy_result[0]['fuzzy_match_confidence']:.2f}")
        else:
            print(f"   Fuzzy match: ❌ Failed")


def test_q2_character_removal():
    """Test Q2 dataset (character removal) corruption recovery."""
    print("***REMOVED***n" + "="*60)
    print("TESTING Q2 CHARACTER REMOVAL RECOVERY")
    print("="*60)
    
    # Load Q2 corrupted data
    data_loader = DataLoader(base_path="data/experimental_datasets/Q2_dataset")
    datasets = data_loader.load_base_data()
    
    relationship_tool = RelationshipTool(datasets)
    
    # Test cases with known character removal
    test_cases = [
        ("Printer_6", "Priter_6"),   # Missing 'n'
        ("Printer_5", "Prnter_5"),   # Missing 'i'
    ]
    
    for clean_id, corrupted_id in test_cases:
        print(f"***REMOVED***n🧪 Testing: '{corrupted_id}' (should find '{clean_id}')")
        
        # Try exact match first (should fail)
        exact_result = relationship_tool.run(corrupted_id, fuzzy_enabled=False)
        print(f"   Exact match: {'✅ Found' if not any('error' in r for r in exact_result) else '❌ Failed'}")
        
        # Try fuzzy match (should succeed)
        fuzzy_result = relationship_tool.run(corrupted_id, fuzzy_enabled=True, threshold=0.7)
        if not any('error' in r for r in fuzzy_result):
            print(f"   Fuzzy match: ✅ Found {len(fuzzy_result)} relationships")
            if fuzzy_result and 'fuzzy_match_confidence' in fuzzy_result[0]:
                print(f"   Confidence: {fuzzy_result[0]['fuzzy_match_confidence']:.2f}")
                print(f"   Matched ID: {fuzzy_result[0].get('matched_id', 'N/A')}")
        else:
            print(f"   Fuzzy match: ❌ Failed")


def test_q3_missing_relationships():
    """Test Q3 dataset (missing relationships) detection."""
    print("***REMOVED***n" + "="*60)
    print("TESTING Q3 MISSING RELATIONSHIPS DETECTION")
    print("="*60)
    
    # Load Q3 corrupted data
    data_loader = DataLoader(base_path="data/experimental_datasets/Q3_dataset")
    datasets = data_loader.load_base_data()
    
    relationship_tool = RelationshipTool(datasets)
    
    # Load baseline for comparison
    baseline_loader = DataLoader(base_path="data/experimental_datasets/Q0_baseline")
    baseline_datasets = baseline_loader.load_base_data()
    baseline_tool = RelationshipTool(baseline_datasets)
    
    # Test specific IDs that should have relationships in baseline but not in Q3
    test_ids = ["3DOR100061", "ORBOX00115", "Printer_6"]
    
    for test_id in test_ids:
        print(f"***REMOVED***n🧪 Testing: '{test_id}'")
        
        # Check baseline
        baseline_result = baseline_tool.run(test_id, fuzzy_enabled=False)
        baseline_count = len(baseline_result) if not any('error' in r for r in baseline_result) else 0
        
        # Check Q3
        q3_result = relationship_tool.run(test_id, fuzzy_enabled=False)
        q3_count = len(q3_result) if not any('error' in r for r in q3_result) else 0
        
        print(f"   Baseline: {baseline_count} relationships")
        print(f"   Q3: {q3_count} relationships")
        
        if baseline_count > 0 and q3_count == 0:
            print(f"   ✅ Correctly detected missing relationships")
        elif baseline_count == q3_count:
            print(f"   ⚠️  No difference detected (may not be corrupted)")
        else:
            print(f"   ❓ Unexpected result")


def test_end_to_end_recovery():
    """Test end-to-end corruption recovery scenarios."""
    print("***REMOVED***n" + "="*60)
    print("TESTING END-TO-END CORRUPTION RECOVERY")
    print("="*60)
    
    # Test the specific cases that were failing in the performance results
    scenarios = [
        {
            "name": "P1_task_4 (Q2) - Printer with missing character",
            "dataset": "Q2_dataset",
            "query_id": "3DOR100056",
            "expected_printer": "Printer_6",
            "corruption_type": "character_removal"
        },
        {
            "name": "P3_task_3 (Q2) - Missing relationships",
            "dataset": "Q2_dataset", 
            "query_id": "3DOR100061",
            "expected_printer": "Printer_1",
            "corruption_type": "missing_relationships"
        }
    ]
    
    for scenario in scenarios:
        print(f"***REMOVED***n🎯 {scenario['name']}")
        
        # Load the corrupted dataset
        data_loader = DataLoader(base_path=f"data/experimental_datasets/{scenario['dataset']}")
        datasets = data_loader.load_base_data()
        relationship_tool = RelationshipTool(datasets)
        
        # Try to find the printer for the part
        result = relationship_tool.run(scenario['query_id'], fuzzy_enabled=True, threshold=0.7)
        
        if not any('error' in r for r in result):
            # Look for printer in the results
            printers = [r for r in result if r.get('parent', '').startswith('Printer') or r.get('child', '').startswith('Printer')]
            if printers:
                printer_name = printers[0].get('parent') or printers[0].get('child')
                confidence = printers[0].get('fuzzy_match_confidence', 1.0)
                print(f"   ✅ Found printer: {printer_name} (confidence: {confidence:.2f})")
                
                if printer_name == scenario['expected_printer']:
                    print(f"   ✅ CORRECT: Matches expected printer")
                else:
                    print(f"   ⚠️  DIFFERENT: Expected {scenario['expected_printer']}")
            else:
                print(f"   ❌ No printer found in relationships")
        else:
            print(f"   ❌ Failed to find any relationships")


def main():
    """Run all corruption recovery tests."""
    print("🔧 TESTING CORRUPTION RECOVERY CAPABILITIES")
    print("="*60)
    
    try:
        test_q1_whitespace_corruption()
        test_q2_character_removal()
        test_q3_missing_relationships()
        test_end_to_end_recovery()
        
        print("***REMOVED***n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✅ Corruption recovery tests completed")
        print("***REMOVED***n📋 Next steps:")
        print("   1. Run: python scripts/fix_ground_truth.py")
        print("   2. Run: python run_phase4_llm_evaluation.py --models deepseek-chat")
        print("   3. Check if performance results now show corruption recovery")
        
    except Exception as e:
        print(f"***REMOVED***n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
