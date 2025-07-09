#!/usr/bin/env python3
"""
Box and Whisker Plot Comparison: Human vs LLM Performance
=========================================================

This script creates comprehensive box and whisker plots comparing human participant
performance against aggregated LLM model performance across different dimensions:
- Overall comparison
- By task complexity (Easy/Medium/Hard)  
- By data quality condition (Q0/Q1/Q2/Q3)

Metrics analyzed: Accuracy and Completion Time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BoxWhiskerComparison:
    """
    Creates box and whisker plots comparing human vs LLM performance
    """
    
    def __init__(self, output_dir: str = "experiments/phase5_analysis/individual_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.human_data = None
        self.llm_data = None
        self.processed_data = None
        
        # Color scheme for consistent visualization
        self.colors = {
            'human': '#FF6B6B',      # Coral red
            'llm': '#4ECDC4',        # Teal
            'human_light': '#FFB3B3', # Light coral
            'llm_light': '#A8E6E1'    # Light teal
        }
    
    def load_human_data(self, human_csv_path: str) -> pd.DataFrame:
        """Load and preprocess human study results"""
        print(f"Loading human data from: {human_csv_path}")
        
        # Handle potential BOM and encoding issues
        try:
            self.human_data = pd.read_csv(human_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.human_data = pd.read_csv(human_csv_path, encoding='utf-8-sig')
        
        # Clean column names (remove BOM if present)
        self.human_data.columns = self.human_data.columns.str.strip()
        
        # Standardize column names
        column_mapping = {
            'data quality': 'quality_condition',
            'completion_time_sec': 'completion_time_sec',
            'accuracy': 'is_correct'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in self.human_data.columns:
                self.human_data = self.human_data.rename(columns={old_col: new_col})
        
        # Add source identifier
        self.human_data['source'] = 'Human'
        
        print(f"Loaded {len(self.human_data)} human task records")
        return self.human_data
    
    def load_llm_data(self, llm_results_dir: str) -> pd.DataFrame:
        """Load and aggregate LLM results from multiple model files"""
        print(f"Loading LLM data from: {llm_results_dir}")
        
        llm_files = glob.glob(f"{llm_results_dir}/*.csv")
        if not llm_files:
            raise FileNotFoundError(f"No CSV files found in {llm_results_dir}")
        
        all_llm_data = []
        
        for file_path in llm_files:
            print(f"  Processing: {Path(file_path).name}")
            try:
                df = pd.read_csv(file_path)
                
                # Convert boolean strings to actual booleans
                if 'is_correct' in df.columns:
                    df['is_correct'] = df['is_correct'].map({
                        'True': 1, 'False': 0, True: 1, False: 0, 1: 1, 0: 0
                    })
                
                all_llm_data.append(df)
            except Exception as e:
                print(f"    Warning: Could not load {file_path}: {e}")
                continue
        
        if not all_llm_data:
            raise ValueError("No valid LLM data files could be loaded")
        
        # Combine all LLM data
        self.llm_data = pd.concat(all_llm_data, ignore_index=True)
        
        print(f"Loaded {len(self.llm_data)} LLM task records from {len(all_llm_data)} files")
        print(f"Models included: {sorted(self.llm_data['model'].unique())}")
        
        return self.llm_data
    
    def process_data_for_comparison(self) -> pd.DataFrame:
        """
        Process and aggregate data for box plot comparison.
        For LLMs: Calculate average performance per model per task, then aggregate.
        For humans: Calculate average performance per participant to get meaningful quartiles.
        """
        print("Processing data for comparison...")

        if self.human_data is None or self.llm_data is None:
            raise ValueError("Both human and LLM data must be loaded first")

        # Get common task_ids
        human_tasks = set(self.human_data['task_id'].unique())
        llm_tasks = set(self.llm_data['task_id'].unique())
        common_tasks = human_tasks.intersection(llm_tasks)

        print(f"Common tasks for comparison: {len(common_tasks)}")

        if len(common_tasks) == 0:
            raise ValueError("No common task_ids found between human and LLM data")

        # Process human data (filter to common tasks)
        human_filtered = self.human_data[self.human_data['task_id'].isin(common_tasks)].copy()

        # Process LLM data: Average per model per task, then aggregate across models
        llm_filtered = self.llm_data[self.llm_data['task_id'].isin(common_tasks)].copy()

        # Step 1: Calculate average performance per model per task
        llm_model_averages = llm_filtered.groupby(['task_id', 'model']).agg({
            'completion_time_sec': 'mean',
            'is_correct': 'mean',
            'complexity': 'first',
            'quality_condition': 'first'
        }).reset_index()

        # Step 2: Calculate overall average across all models per task
        llm_aggregated = llm_model_averages.groupby('task_id').agg({
            'completion_time_sec': 'mean',
            'is_correct': 'mean',
            'complexity': 'first',
            'quality_condition': 'first'
        }).reset_index()

        # Add source identifier
        llm_aggregated['source'] = 'LLM'

        # Store both individual and aggregated data for different visualization needs
        self.individual_data = []
        self.aggregated_data = []

        # Add individual human data (for task-level analysis)
        for _, row in human_filtered.iterrows():
            self.individual_data.append({
                'task_id': row['task_id'],
                'participant_id': row.get('participant_id', 'Unknown'),
                'source': 'Human',
                'accuracy': row['is_correct'],
                'completion_time_sec': row['completion_time_sec'],
                'complexity': row['complexity'],
                'quality_condition': row['quality_condition']
            })

        # Add individual LLM data (for task-level analysis)
        for _, row in llm_aggregated.iterrows():
            self.individual_data.append({
                'task_id': row['task_id'],
                'participant_id': 'LLM_Aggregate',
                'source': 'LLM',
                'accuracy': row['is_correct'],
                'completion_time_sec': row['completion_time_sec'],
                'complexity': row['complexity'],
                'quality_condition': row['quality_condition']
            })

        # Create aggregated data for participant-level analysis (for proper box plots)
        # Human: Average per participant
        human_participant_avg = human_filtered.groupby('participant_id').agg({
            'is_correct': 'mean',
            'completion_time_sec': 'mean'
        }).reset_index()

        for _, row in human_participant_avg.iterrows():
            self.aggregated_data.append({
                'participant_id': row['participant_id'],
                'source': 'Human',
                'accuracy': row['is_correct'],
                'completion_time_sec': row['completion_time_sec']
            })

        # LLM: Average per task (already aggregated across models)
        for _, row in llm_aggregated.iterrows():
            self.aggregated_data.append({
                'participant_id': f"Task_{row['task_id']}",
                'source': 'LLM',
                'accuracy': row['is_correct'],
                'completion_time_sec': row['completion_time_sec']
            })

        # Store both datasets
        self.individual_data = pd.DataFrame(self.individual_data)
        self.aggregated_data = pd.DataFrame(self.aggregated_data)

        # For backward compatibility, set processed_data to individual_data
        self.processed_data = self.individual_data

        print(f"Individual data contains {len(self.individual_data)} records")
        print(f"  Human records: {len(self.individual_data[self.individual_data['source'] == 'Human'])}")
        print(f"  LLM records: {len(self.individual_data[self.individual_data['source'] == 'LLM'])}")

        print(f"Aggregated data contains {len(self.aggregated_data)} records")
        print(f"  Human participants: {len(self.aggregated_data[self.aggregated_data['source'] == 'Human'])}")
        print(f"  LLM tasks: {len(self.aggregated_data[self.aggregated_data['source'] == 'LLM'])}")

        return self.processed_data

    def create_overall_comparison_plots(self):
        """Create overall Human vs LLM comparison box plots using aggregated data"""
        print("Creating overall comparison plots...")

        if self.aggregated_data is None:
            raise ValueError("Data must be processed first")

        # Create figure with subplots for accuracy and time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Accuracy comparison using aggregated data (participant averages for humans, task averages for LLMs)
        accuracy_data = [
            self.aggregated_data[self.aggregated_data['source'] == 'Human']['accuracy'].values,
            self.aggregated_data[self.aggregated_data['source'] == 'LLM']['accuracy'].values
        ]

        bp1 = ax1.boxplot(accuracy_data, labels=['Human***REMOVED***n(Participant Avg)', 'LLM***REMOVED***n(Task Avg)'], patch_artist=True)
        bp1['boxes'][0].set_facecolor(self.colors['human'])
        bp1['boxes'][1].set_facecolor(self.colors['llm'])

        ax1.set_title('Accuracy Comparison: Human vs LLM', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy Rate', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        # Add sample size annotations
        human_n = len(accuracy_data[0])
        llm_n = len(accuracy_data[1])
        ax1.text(1, -0.02, f'n={human_n}', ha='center', va='top', fontsize=10)
        ax1.text(2, -0.02, f'n={llm_n}', ha='center', va='top', fontsize=10)

        # Completion time comparison using aggregated data
        time_data = [
            self.aggregated_data[self.aggregated_data['source'] == 'Human']['completion_time_sec'].values,
            self.aggregated_data[self.aggregated_data['source'] == 'LLM']['completion_time_sec'].values
        ]

        bp2 = ax2.boxplot(time_data, labels=['Human***REMOVED***n(Participant Avg)', 'LLM***REMOVED***n(Task Avg)'], patch_artist=True)
        bp2['boxes'][0].set_facecolor(self.colors['human'])
        bp2['boxes'][1].set_facecolor(self.colors['llm'])

        ax2.set_title('Completion Time Comparison: Human vs LLM', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Completion Time (seconds)', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add sample size annotations
        ax2.text(1, ax2.get_ylim()[0], f'n={human_n}', ha='center', va='top', fontsize=10)
        ax2.text(2, ax2.get_ylim()[0], f'n={llm_n}', ha='center', va='top', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / "overall_human_vs_llm_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("  Saved: overall_human_vs_llm_boxplots.png")

    def create_complexity_comparison_plots(self):
        """Create complexity-based Human vs LLM comparison box plots"""
        print("Creating complexity comparison plots...")

        if self.individual_data is None:
            raise ValueError("Data must be processed first")

        complexities = ['easy', 'medium', 'hard']

        # Create figure with subplots for accuracy and time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # For complexity analysis, we need to aggregate by participant within each complexity level
        # to get meaningful box plots

        # Accuracy by complexity
        accuracy_data_by_complexity = []
        labels = []
        colors = []

        for complexity in complexities:
            # Human data: aggregate by participant for this complexity
            human_complexity = self.individual_data[
                (self.individual_data['source'] == 'Human') &
                (self.individual_data['complexity'] == complexity)
            ]

            if len(human_complexity) > 0:
                human_participant_avg = human_complexity.groupby('participant_id')['accuracy'].mean().values
            else:
                human_participant_avg = np.array([])

            # LLM data: use task-level averages for this complexity
            llm_complexity = self.individual_data[
                (self.individual_data['source'] == 'LLM') &
                (self.individual_data['complexity'] == complexity)
            ]
            llm_acc = llm_complexity['accuracy'].values if len(llm_complexity) > 0 else np.array([])

            accuracy_data_by_complexity.extend([human_participant_avg, llm_acc])
            labels.extend([f'Human***REMOVED***n({complexity.title()})', f'LLM***REMOVED***n({complexity.title()})'])
            colors.extend([self.colors['human'], self.colors['llm']])

        bp1 = ax1.boxplot(accuracy_data_by_complexity, labels=labels, patch_artist=True)
        for i, box in enumerate(bp1['boxes']):
            box.set_facecolor(colors[i])

        ax1.set_title('Accuracy by Task Complexity: Human vs LLM', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy Rate', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        ax1.tick_params(axis='x', rotation=45)

        # Add sample size annotations
        for i, data in enumerate(accuracy_data_by_complexity):
            ax1.text(i+1, -0.02, f'n={len(data)}', ha='center', va='top', fontsize=9)

        # Completion time by complexity
        time_data_by_complexity = []

        for complexity in complexities:
            # Human data: aggregate by participant for this complexity
            human_complexity = self.individual_data[
                (self.individual_data['source'] == 'Human') &
                (self.individual_data['complexity'] == complexity)
            ]

            if len(human_complexity) > 0:
                human_participant_avg = human_complexity.groupby('participant_id')['completion_time_sec'].mean().values
            else:
                human_participant_avg = np.array([])

            # LLM data: use task-level averages for this complexity
            llm_complexity = self.individual_data[
                (self.individual_data['source'] == 'LLM') &
                (self.individual_data['complexity'] == complexity)
            ]
            llm_time = llm_complexity['completion_time_sec'].values if len(llm_complexity) > 0 else np.array([])

            time_data_by_complexity.extend([human_participant_avg, llm_time])

        bp2 = ax2.boxplot(time_data_by_complexity, labels=labels, patch_artist=True)
        for i, box in enumerate(bp2['boxes']):
            box.set_facecolor(colors[i])

        ax2.set_title('Completion Time by Task Complexity: Human vs LLM', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Completion Time (seconds)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # Add sample size annotations
        for i, data in enumerate(time_data_by_complexity):
            ax2.text(i+1, ax2.get_ylim()[0], f'n={len(data)}', ha='center', va='top', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / "complexity_human_vs_llm_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("  Saved: complexity_human_vs_llm_boxplots.png")

    def create_quality_comparison_plots(self):
        """Create data quality-based Human vs LLM comparison box plots"""
        print("Creating data quality comparison plots...")

        if self.individual_data is None:
            raise ValueError("Data must be processed first")

        qualities = ['Q0', 'Q1', 'Q2', 'Q3']
        quality_labels = {
            'Q0': 'Normal Baseline',
            'Q1': 'Space Injection',
            'Q2': 'Character Missing',
            'Q3': 'Missing Records'
        }

        # Create figure with subplots for accuracy and time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        # Accuracy by data quality
        accuracy_data_by_quality = []
        labels = []
        colors = []

        for quality in qualities:
            # Human data: aggregate by participant for this quality condition
            human_quality = self.individual_data[
                (self.individual_data['source'] == 'Human') &
                (self.individual_data['quality_condition'] == quality)
            ]

            if len(human_quality) > 0:
                human_participant_avg = human_quality.groupby('participant_id')['accuracy'].mean().values
            else:
                human_participant_avg = np.array([])

            # LLM data: use task-level averages for this quality condition
            llm_quality = self.individual_data[
                (self.individual_data['source'] == 'LLM') &
                (self.individual_data['quality_condition'] == quality)
            ]
            llm_acc = llm_quality['accuracy'].values if len(llm_quality) > 0 else np.array([])

            accuracy_data_by_quality.extend([human_participant_avg, llm_acc])
            labels.extend([f'Human***REMOVED***n({quality_labels[quality]})', f'LLM***REMOVED***n({quality_labels[quality]})'])
            colors.extend([self.colors['human'], self.colors['llm']])

        bp1 = ax1.boxplot(accuracy_data_by_quality, labels=labels, patch_artist=True)
        for i, box in enumerate(bp1['boxes']):
            box.set_facecolor(colors[i])

        ax1.set_title('Accuracy by Data Quality Condition: Human vs LLM', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy Rate', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        ax1.tick_params(axis='x', rotation=45)

        # Add sample size annotations
        for i, data in enumerate(accuracy_data_by_quality):
            ax1.text(i+1, -0.02, f'n={len(data)}', ha='center', va='top', fontsize=9)

        # Completion time by data quality
        time_data_by_quality = []

        for quality in qualities:
            # Human data: aggregate by participant for this quality condition
            human_quality = self.individual_data[
                (self.individual_data['source'] == 'Human') &
                (self.individual_data['quality_condition'] == quality)
            ]

            if len(human_quality) > 0:
                human_participant_avg = human_quality.groupby('participant_id')['completion_time_sec'].mean().values
            else:
                human_participant_avg = np.array([])

            # LLM data: use task-level averages for this quality condition
            llm_quality = self.individual_data[
                (self.individual_data['source'] == 'LLM') &
                (self.individual_data['quality_condition'] == quality)
            ]
            llm_time = llm_quality['completion_time_sec'].values if len(llm_quality) > 0 else np.array([])

            time_data_by_quality.extend([human_participant_avg, llm_time])

        bp2 = ax2.boxplot(time_data_by_quality, labels=labels, patch_artist=True)
        for i, box in enumerate(bp2['boxes']):
            box.set_facecolor(colors[i])

        ax2.set_title('Completion Time by Data Quality Condition: Human vs LLM', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Completion Time (seconds)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # Add sample size annotations
        for i, data in enumerate(time_data_by_quality):
            ax2.text(i+1, ax2.get_ylim()[0], f'n={len(data)}', ha='center', va='top', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / "quality_human_vs_llm_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("  Saved: quality_human_vs_llm_boxplots.png")

    def generate_summary_statistics(self):
        """Generate and print summary statistics for the comparison"""
        print("***REMOVED***n" + "="*60)
        print("SUMMARY STATISTICS: HUMAN vs LLM COMPARISON")
        print("="*60)

        if self.processed_data is None:
            raise ValueError("Data must be processed first")

        human_data = self.processed_data[self.processed_data['source'] == 'Human']
        llm_data = self.processed_data[self.processed_data['source'] == 'LLM']

        print(f"***REMOVED***nOverall Performance:")
        print(f"  Human Accuracy: {human_data['accuracy'].mean():.3f} ± {human_data['accuracy'].std():.3f}")
        print(f"  LLM Accuracy:   {llm_data['accuracy'].mean():.3f} ± {llm_data['accuracy'].std():.3f}")
        print(f"  Human Time:     {human_data['completion_time_sec'].mean():.1f} ± {human_data['completion_time_sec'].std():.1f} sec")
        print(f"  LLM Time:       {llm_data['completion_time_sec'].mean():.1f} ± {llm_data['completion_time_sec'].std():.1f} sec")

        print(f"***REMOVED***nSample Sizes:")
        print(f"  Human tasks: {len(human_data)}")
        print(f"  LLM tasks:   {len(llm_data)}")

        # Performance by complexity
        print(f"***REMOVED***nPerformance by Complexity:")
        for complexity in ['easy', 'medium', 'hard']:
            human_comp = human_data[human_data['complexity'] == complexity]
            llm_comp = llm_data[llm_data['complexity'] == complexity]

            if len(human_comp) > 0 and len(llm_comp) > 0:
                print(f"  {complexity.title()}:")
                print(f"    Human Accuracy: {human_comp['accuracy'].mean():.3f} (n={len(human_comp)})")
                print(f"    LLM Accuracy:   {llm_comp['accuracy'].mean():.3f} (n={len(llm_comp)})")

        # Performance by data quality
        print(f"***REMOVED***nPerformance by Data Quality:")
        quality_labels = {'Q0': 'Normal Baseline', 'Q1': 'Space Injection', 'Q2': 'Character Missing', 'Q3': 'Missing Records'}
        for quality in ['Q0', 'Q1', 'Q2', 'Q3']:
            human_qual = human_data[human_data['quality_condition'] == quality]
            llm_qual = llm_data[llm_data['quality_condition'] == quality]

            if len(human_qual) > 0 and len(llm_qual) > 0:
                print(f"  {quality} ({quality_labels[quality]}):")
                print(f"    Human Accuracy: {human_qual['accuracy'].mean():.3f} (n={len(human_qual)})")
                print(f"    LLM Accuracy:   {llm_qual['accuracy'].mean():.3f} (n={len(llm_qual)})")

    def run_complete_analysis(self, human_csv_path: str, llm_results_dir: str):
        """Run the complete box and whisker plot analysis"""
        print("="*60)
        print("BOX AND WHISKER PLOT ANALYSIS: HUMAN vs LLM")
        print("="*60)

        # Load data
        self.load_human_data(human_csv_path)
        self.load_llm_data(llm_results_dir)

        # Process data for comparison
        self.process_data_for_comparison()

        # Generate all visualizations
        self.create_overall_comparison_plots()
        self.create_complexity_comparison_plots()
        self.create_quality_comparison_plots()

        # Generate summary statistics
        self.generate_summary_statistics()

        print(f"***REMOVED***nAll visualizations saved to: {self.output_dir}")
        print("Generated files:")
        print("  - overall_human_vs_llm_boxplots.png")
        print("  - complexity_human_vs_llm_boxplots.png")
        print("  - quality_human_vs_llm_boxplots.png")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate Human vs LLM Box and Whisker Plot Comparisons')
    parser.add_argument('--human-data',
                       default='experiments/human_study/Participants_results.csv',
                       help='Path to human study results CSV file')
    parser.add_argument('--llm-data',
                       default='experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark',
                       help='Path to directory containing LLM evaluation results')
    parser.add_argument('--output-dir',
                       default='experiments/phase5_analysis/individual_visualizations',
                       help='Output directory for box plot visualizations')

    args = parser.parse_args()

    # Create and run analysis
    analyzer = BoxWhiskerComparison(output_dir=args.output_dir)
    analyzer.run_complete_analysis(args.human_data, args.llm_data)


if __name__ == "__main__":
    main()
