#!/usr/bin/env python3
"""
Model Ranking Summary for Phase 5 Analysis

Quick script to generate model performance rankings and key insights.
"""

import pandas as pd
import sys
from pathlib import Path

def generate_model_ranking_summary(results_dir: str = "experiments/phase5_analysis/results"):
    """Generate a comprehensive model ranking summary"""
    
    results_path = Path(results_dir)
    model_file = results_path / "model_specific_comparison.csv"
    
    if not model_file.exists():
        print(f"Error: Model comparison file not found at {model_file}")
        print("Please run the Phase 5 analysis first: python run_phase5_analysis.py")
        return
    
    # Load model comparison data
    df = pd.read_csv(model_file)
    
    print("=" * 80)
    print("PHASE 5: MODEL-SPECIFIC PERFORMANCE RANKING")
    print("=" * 80)
    
    # Overall summary
    human_accuracy = df['human_accuracy'].iloc[0]
    print(f"***REMOVED***n📊 HUMAN BASELINE: {human_accuracy:.3f} ({human_accuracy*100:.1f}% accuracy)")
    print(f"📈 MODELS TESTED: {len(df)} LLM models")
    
    # Top performers
    print(f"***REMOVED***n🏆 TOP PERFORMERS:")
    df_sorted = df.sort_values('model_accuracy', ascending=False)
    
    for i, (_, row) in enumerate(df_sorted.head(3).iterrows(), 1):
        improvement = row['accuracy_difference']
        speed = row['time_speedup_factor']
        acc_sig = "✓" if row['accuracy_chi2_significant'] else "✗"
        time_sig = "✓" if row['time_mw_significant'] else "✗"
        
        print(f"  {i}. {row['model']}")
        print(f"     Accuracy: {row['model_accuracy']:.3f} ({improvement:+.3f} vs human)")
        print(f"     Speed: {speed:.1f}x faster")
        print(f"     Significance: Acc {acc_sig}, Time {time_sig}")
        print()
    
    # Detailed rankings
    print("***REMOVED***n📈 ACCURACY RANKING (vs Human Baseline)")
    print("-" * 50)
    df_sorted = df.sort_values('model_accuracy', ascending=False)
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        improvement = row['accuracy_difference']
        status = "🟢" if improvement > 0.1 else "🟡" if improvement > 0 else "🔴"
        sig = "✓" if row['accuracy_chi2_significant'] else "✗"
        
        print(f"{i}. {status} {row['model']:<30} {row['model_accuracy']:.3f} ({improvement:+.3f}) [Sig: {sig}]")
    
    print(f"***REMOVED***n⚡ SPEED RANKING (Speedup Factor)")
    print("-" * 50)
    df_sorted = df.sort_values('time_speedup_factor', ascending=False)
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        speed = row['time_speedup_factor']
        status = "🟢" if speed > 10 else "🟡" if speed > 2 else "🔴"
        sig = "✓" if row['time_mw_significant'] else "✗"
        
        print(f"{i}. {status} {row['model']:<30} {speed:.1f}x faster [Sig: {sig}]")
    
    # Statistical significance summary
    print(f"***REMOVED***n📊 STATISTICAL SIGNIFICANCE SUMMARY")
    print("-" * 50)
    
    both_sig = df[(df['accuracy_chi2_significant']) & (df['time_mw_significant'])]
    acc_only = df[(df['accuracy_chi2_significant']) & (~df['time_mw_significant'])]
    time_only = df[(~df['accuracy_chi2_significant']) & (df['time_mw_significant'])]
    neither = df[(~df['accuracy_chi2_significant']) & (~df['time_mw_significant'])]
    
    print(f"🟢 Both accuracy & time significant: {len(both_sig)} models")
    for _, row in both_sig.iterrows():
        print(f"   - {row['model']}")
    
    print(f"***REMOVED***n🟡 Accuracy significant only: {len(acc_only)} models")
    for _, row in acc_only.iterrows():
        print(f"   - {row['model']}")
    
    print(f"***REMOVED***n🟡 Time significant only: {len(time_only)} models")
    for _, row in time_only.iterrows():
        print(f"   - {row['model']}")
    
    print(f"***REMOVED***n🔴 Neither significant: {len(neither)} models")
    for _, row in neither.iterrows():
        print(f"   - {row['model']}")
    
    # Key insights
    print(f"***REMOVED***n💡 KEY INSIGHTS")
    print("-" * 50)
    
    best_accuracy = df_sorted.iloc[0]
    fastest_model = df.loc[df['time_speedup_factor'].idxmax()]
    worst_accuracy = df_sorted.iloc[-1]
    
    print(f"• Best accuracy: {best_accuracy['model']} ({best_accuracy['model_accuracy']:.3f})")
    print(f"• Fastest model: {fastest_model['model']} ({fastest_model['time_speedup_factor']:.1f}x)")
    print(f"• Only model worse than humans: {worst_accuracy['model']} ({worst_accuracy['accuracy_difference']:.3f})")
    
    models_better = len(df[df['accuracy_difference'] > 0])
    print(f"• Models outperforming humans: {models_better}/{len(df)} ({models_better/len(df)*100:.0f}%)")
    
    avg_improvement = df[df['accuracy_difference'] > 0]['accuracy_difference'].mean()
    print(f"• Average improvement (successful models): +{avg_improvement:.3f} ({avg_improvement*100:.1f}%)")
    
    print(f"***REMOVED***n🎯 RECOMMENDATIONS")
    print("-" * 50)
    
    # Find best overall model (balance of accuracy and speed)
    df['combined_score'] = df['model_accuracy'] * (df['time_speedup_factor'] / df['time_speedup_factor'].max())
    best_overall = df.loc[df['combined_score'].idxmax()]
    
    print(f"• Best overall model: {best_overall['model']}")
    print(f"  - High accuracy ({best_overall['model_accuracy']:.3f}) with good speed ({best_overall['time_speedup_factor']:.1f}x)")
    
    if len(both_sig) > 0:
        print(f"• Statistically robust choices: {', '.join(both_sig['model'].tolist())}")
    
    if len(df[df['accuracy_difference'] < 0]) > 0:
        avoid_models = df[df['accuracy_difference'] < 0]['model'].tolist()
        print(f"• Avoid: {', '.join(avoid_models)} (worse than human baseline)")
    
    # Complexity and Quality Analysis
    complexity_file = results_path / "model_complexity_analysis.csv"
    quality_file = results_path / "model_quality_analysis.csv"

    if complexity_file.exists():
        print(f"***REMOVED***n🎯 PERFORMANCE BY TASK COMPLEXITY")
        print("-" * 50)

        complexity_df = pd.read_csv(complexity_file)

        for complexity in ['easy', 'medium', 'hard']:
            comp_data = complexity_df[complexity_df['complexity'] == complexity]
            if len(comp_data) > 0:
                best_model = comp_data.loc[comp_data['model_accuracy'].idxmax()]
                print(f"***REMOVED***n{complexity.upper()} Tasks:")
                print(f"  🏆 Best: {best_model['model']} ({best_model['model_accuracy']:.3f})")
                print(f"  📈 Improvement: {best_model['accuracy_difference']:+.3f} vs human")
                print(f"  ⚡ Speed: {best_model['time_speedup_factor']:.1f}x faster")

    if quality_file.exists():
        print(f"***REMOVED***n🛡️ DATA QUALITY ROBUSTNESS")
        print("-" * 50)

        quality_df = pd.read_csv(quality_file)

        # Calculate robustness scores
        robustness_data = []
        for model in quality_df['model'].unique():
            model_data = quality_df[quality_df['model'] == model]
            q0_data = model_data[model_data['quality_condition'] == 'Q0']
            corrupted_data = model_data[model_data['quality_condition'].isin(['Q1', 'Q2', 'Q3'])]

            if len(q0_data) > 0 and len(corrupted_data) > 0:
                q0_acc = q0_data['model_accuracy'].iloc[0]
                corrupted_avg = corrupted_data['model_accuracy'].mean()
                robustness = corrupted_avg / q0_acc if q0_acc > 0 else 0
                robustness_data.append((model, robustness, q0_acc, corrupted_avg))

        # Sort by robustness
        robustness_data.sort(key=lambda x: x[1], reverse=True)

        for i, (model, robustness, q0_acc, corrupted_avg) in enumerate(robustness_data, 1):
            status = "🟢" if robustness > 0.8 else "🟡" if robustness > 0.6 else "🔴"
            print(f"{i}. {status} {model:<30} Robustness: {robustness:.3f}")
            print(f"   Perfect: {q0_acc:.3f} | Corrupted Avg: {corrupted_avg:.3f}")

        print(f"***REMOVED***n🔍 QUALITY CONDITION BREAKDOWN")
        print("-" * 50)

        for quality in ['Q0', 'Q1', 'Q2', 'Q3']:
            qual_data = quality_df[quality_df['quality_condition'] == quality]
            if len(qual_data) > 0:
                best_model = qual_data.loc[qual_data['model_accuracy'].idxmax()]
                worst_model = qual_data.loc[qual_data['model_accuracy'].idxmin()]

                quality_names = {
                    'Q0': 'Perfect Data',
                    'Q1': 'Extra Spaces',
                    'Q2': 'Missing Characters',
                    'Q3': 'Missing Records'
                }

                print(f"***REMOVED***n{quality} ({quality_names[quality]}):")
                print(f"  🏆 Best: {best_model['model']} ({best_model['model_accuracy']:.3f})")
                print(f"  🔻 Worst: {worst_model['model']} ({worst_model['model_accuracy']:.3f})")
                print(f"  📊 Range: {best_model['model_accuracy'] - worst_model['model_accuracy']:.3f}")

    print("***REMOVED***n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_model_ranking_summary(sys.argv[1])
    else:
        generate_model_ranking_summary()
