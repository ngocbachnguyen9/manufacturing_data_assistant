#!/usr/bin/env python3
"""
Helper script to run benchmarking with unique output directories (Option 1 approach)
Automatically generates appropriate directory names based on CSV filename
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from benchmarking_suite import ComprehensiveBenchmarkingSuite

def extract_info_from_filename(csv_path: str) -> dict:
    """Extract model, prompt length, and other info from CSV filename"""
    csv_file = Path(csv_path)
    filename = csv_file.stem  # Remove .csv extension
    
    # Expected format: model_promptlength_subset_date.csv
    # Example: deepseek-chat_short_all_20250617.csv
    parts = filename.split('_')
    
    info = {
        'model': 'unknown',
        'prompt_length': 'unknown', 
        'subset': 'unknown',
        'date': 'unknown',
        'full_filename': filename
    }
    
    if len(parts) >= 4:
        info['model'] = parts[0]
        info['prompt_length'] = parts[1]
        info['subset'] = parts[2]
        info['date'] = parts[3]
    elif len(parts) >= 3:
        info['model'] = parts[0]
        info['prompt_length'] = parts[1]
        info['subset'] = parts[2]
    elif len(parts) >= 2:
        info['model'] = parts[0]
        info['prompt_length'] = parts[1]
    elif len(parts) >= 1:
        info['model'] = parts[0]
    
    return info

def generate_unique_output_dir(csv_path: str, custom_suffix: str = None) -> str:
    """Generate a unique output directory name based on CSV file info"""
    info = extract_info_from_filename(csv_path)
    
    # Base directory name
    base_name = f"benchmark_{info['model']}_{info['prompt_length']}_{info['subset']}"
    
    # Add custom suffix if provided
    if custom_suffix:
        base_name += f"_{custom_suffix}"
    
    # Add date if not already in filename
    if info['date'] == 'unknown':
        today = datetime.now().strftime("%Y%m%d")
        base_name += f"_{today}"
    
    return base_name

def run_benchmark_with_unique_output(csv_path: str, custom_suffix: str = None, 
                                   output_dir: str = None) -> tuple:
    """
    Run comprehensive benchmarking with unique output directory
    
    Args:
        csv_path: Path to the CSV file to benchmark
        custom_suffix: Optional custom suffix for output directory
        output_dir: Optional custom output directory (overrides auto-generation)
    
    Returns:
        tuple: (suite_result, json_path, summary_path, output_directory)
    """
    
    # Validate CSV file exists
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Generate unique output directory
    if output_dir is None:
        output_dir = generate_unique_output_dir(csv_path, custom_suffix)
    
    print(f"🚀 Running benchmark for: {csv_file.name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Input file: {csv_path}")
    print("-" * 60)
    
    # Initialize benchmarking suite with unique output directory
    suite = ComprehensiveBenchmarkingSuite(
        results_path=csv_path,
        output_dir=output_dir
    )
    
    # Run complete benchmark suite
    suite_result, json_path, summary_path = suite.run_and_export_complete_suite()
    
    print(f"***REMOVED***n✅ Benchmarking completed successfully!")
    print(f"📁 All files saved to: {output_dir}/")
    print(f"📊 JSON Results: {json_path}")
    print(f"📝 Executive Summary: {summary_path}")
    print(f"📈 Visualizations: {output_dir}/visualizations/")
    print(f"🎯 Prompt Analysis: {output_dir}/prompt_analysis/")
    
    return suite_result, json_path, summary_path, output_dir

def run_multiple_benchmarks(csv_files: list, custom_suffixes: list = None) -> dict:
    """
    Run benchmarks for multiple CSV files with unique output directories
    
    Args:
        csv_files: List of CSV file paths
        custom_suffixes: Optional list of custom suffixes (same length as csv_files)
    
    Returns:
        dict: Results for each file {csv_path: (suite_result, json_path, summary_path, output_dir)}
    """
    
    if custom_suffixes and len(custom_suffixes) != len(csv_files):
        raise ValueError("custom_suffixes must be same length as csv_files")
    
    results = {}
    
    for i, csv_path in enumerate(csv_files):
        suffix = custom_suffixes[i] if custom_suffixes else None
        
        print(f"***REMOVED***n{'='*60}")
        print(f"BENCHMARK {i+1}/{len(csv_files)}")
        print(f"{'='*60}")
        
        try:
            result = run_benchmark_with_unique_output(csv_path, suffix)
            results[csv_path] = result
            
        except Exception as e:
            print(f"❌ Error benchmarking {csv_path}: {e}")
            results[csv_path] = None
    
    # Summary
    print(f"***REMOVED***n{'='*60}")
    print("MULTIPLE BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results.values() if r is not None)
    total = len(csv_files)
    
    print(f"✅ Successful: {successful}/{total}")
    
    for csv_path, result in results.items():
        status = "✅" if result else "❌"
        filename = Path(csv_path).name
        print(f"{status} {filename}")
        if result:
            _, _, _, output_dir = result
            print(f"   📁 {output_dir}/")
    
    return results

def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python run_benchmark_with_unique_output.py <csv_file> [custom_suffix]")
        print("***REMOVED***nExamples:")
        print("  python run_benchmark_with_unique_output.py performance_logs/deepseek-chat_short_all_20250617.csv")
        print("  python run_benchmark_with_unique_output.py performance_logs/gpt4o_normal_all_20250617.csv experiment1")
        print("***REMOVED***nFor multiple files:")
        print("  python run_benchmark_with_unique_output.py file1.csv file2.csv file3.csv")
        return
    
    csv_files = sys.argv[1:]
    
    if len(csv_files) == 1:
        # Single file
        csv_path = csv_files[0]
        custom_suffix = sys.argv[2] if len(sys.argv) > 2 else None
        
        try:
            run_benchmark_with_unique_output(csv_path, custom_suffix)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
            
    else:
        # Multiple files
        try:
            run_multiple_benchmarks(csv_files)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
