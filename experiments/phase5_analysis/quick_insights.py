#!/usr/bin/env python3
"""
Quick Insights Generator for Phase 5 Analysis

Provides rapid insights and recommendations based on model-specific 
complexity and quality analysis results.
"""

import pandas as pd
import sys
from pathlib import Path

def generate_quick_insights(results_dir: str = "experiments/phase5_analysis/results"):
    """Generate quick actionable insights from Phase 5 analysis"""
    
    results_path = Path(results_dir)
    
    # Load all analysis files
    files = {
        'overall': results_path / "overall_comparison.csv",
        'models': results_path / "model_specific_comparison.csv", 
        'complexity': results_path / "model_complexity_analysis.csv",
        'quality': results_path / "model_quality_analysis.csv"
    }
    
    # Check if files exist
    missing_files = [name for name, path in files.items() if not path.exists()]
    if missing_files:
        print(f"❌ Missing analysis files: {missing_files}")
        print("Please run Phase 5 analysis first: python run_phase5_analysis.py")
        return
    
    # Load data
    data = {}
    for name, path in files.items():
        data[name] = pd.read_csv(path)
    
    print("🚀 PHASE 5 QUICK INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    
    # Overall performance summary
    overall = data['overall'].iloc[0]
    print(f"***REMOVED***n📊 OVERALL PERFORMANCE")
    print(f"   Human Baseline: {overall['human_accuracy']:.1%}")
    print(f"   LLM Average: {overall['llm_accuracy']:.1%} ({overall['accuracy_difference']:+.1%})")
    print(f"   Speed Improvement: {overall['time_speedup_factor']:.1f}x faster")
    
    # Best model recommendations by use case
    models_df = data['models']
    complexity_df = data['complexity']
    quality_df = data['quality']
    
    print(f"***REMOVED***n🎯 BEST MODEL RECOMMENDATIONS BY USE CASE")
    print("-" * 50)
    
    # 1. Best overall model
    best_overall = models_df.loc[models_df['model_accuracy'].idxmax()]
    print(f"🏆 BEST OVERALL: {best_overall['model']}")
    print(f"   Accuracy: {best_overall['model_accuracy']:.1%} ({best_overall['accuracy_difference']:+.1%})")
    print(f"   Speed: {best_overall['time_speedup_factor']:.1f}x faster")
    print(f"   Use case: General-purpose manufacturing data analysis")
    
    # 2. Best for speed-critical applications
    fastest_accurate = models_df[models_df['model_accuracy'] > overall['human_accuracy']].loc[
        models_df[models_df['model_accuracy'] > overall['human_accuracy']]['time_speedup_factor'].idxmax()
    ]
    print(f"***REMOVED***n⚡ BEST FOR SPEED: {fastest_accurate['model']}")
    print(f"   Speed: {fastest_accurate['time_speedup_factor']:.1f}x faster")
    print(f"   Accuracy: {fastest_accurate['model_accuracy']:.1%}")
    print(f"   Use case: Real-time quality control, high-volume processing")
    
    # 3. Best for accuracy-critical applications
    most_accurate = models_df.loc[models_df['model_accuracy'].idxmax()]
    print(f"***REMOVED***n🎯 BEST FOR ACCURACY: {most_accurate['model']}")
    print(f"   Accuracy: {most_accurate['model_accuracy']:.1%}")
    print(f"   Improvement: {most_accurate['accuracy_difference']:+.1%} vs humans")
    print(f"   Use case: Critical safety inspections, regulatory compliance")
    
    # 4. Best for corrupted data
    # Calculate robustness scores
    robustness_scores = []
    for model in quality_df['model'].unique():
        model_data = quality_df[quality_df['model'] == model]
        q0_data = model_data[model_data['quality_condition'] == 'Q0']
        corrupted_data = model_data[model_data['quality_condition'].isin(['Q1', 'Q2', 'Q3'])]
        
        if len(q0_data) > 0 and len(corrupted_data) > 0:
            q0_acc = q0_data['model_accuracy'].iloc[0]
            corrupted_avg = corrupted_data['model_accuracy'].mean()
            robustness = corrupted_avg / q0_acc if q0_acc > 0 else 0
            robustness_scores.append((model, robustness, q0_acc))
    
    if robustness_scores:
        most_robust = max(robustness_scores, key=lambda x: x[1])
        print(f"***REMOVED***n🛡️ BEST FOR CORRUPTED DATA: {most_robust[0]}")
        print(f"   Robustness: {most_robust[1]:.1%} (corrupted vs perfect)")
        print(f"   Perfect data accuracy: {most_robust[2]:.1%}")
        print(f"   Use case: Legacy systems, poor data quality environments")
    
    # Task complexity insights
    print(f"***REMOVED***n📈 TASK COMPLEXITY INSIGHTS")
    print("-" * 50)
    
    for complexity in ['easy', 'medium', 'hard']:
        comp_data = complexity_df[complexity_df['complexity'] == complexity]
        if len(comp_data) > 0:
            best_model = comp_data.loc[comp_data['model_accuracy'].idxmax()]
            human_acc = best_model['human_accuracy']
            
            print(f"{complexity.upper()} Tasks:")
            print(f"   🏆 Best: {best_model['model']} ({best_model['model_accuracy']:.1%})")
            print(f"   📊 Human: {human_acc:.1%}")
            print(f"   📈 Improvement: {best_model['accuracy_difference']:+.1%}")
            
            # Find models that struggle with this complexity
            struggling = comp_data[comp_data['model_accuracy'] < human_acc]
            if len(struggling) > 0:
                worst = struggling.loc[struggling['model_accuracy'].idxmin()]
                print(f"   ⚠️  Avoid: {worst['model']} ({worst['model_accuracy']:.1%})")
            print()
    
    # Data quality insights
    print(f"🔍 DATA QUALITY INSIGHTS")
    print("-" * 50)
    
    quality_conditions = {
        'Q0': 'Perfect Data',
        'Q1': 'Extra Spaces', 
        'Q2': 'Missing Characters',
        'Q3': 'Missing Records'
    }
    
    for quality_code, quality_name in quality_conditions.items():
        qual_data = quality_df[quality_df['quality_condition'] == quality_code]
        if len(qual_data) > 0:
            best_model = qual_data.loc[qual_data['model_accuracy'].idxmax()]
            worst_model = qual_data.loc[qual_data['model_accuracy'].idxmin()]
            
            print(f"{quality_name} ({quality_code}):")
            print(f"   🏆 Best: {best_model['model']} ({best_model['model_accuracy']:.1%})")
            print(f"   🔻 Worst: {worst_model['model']} ({worst_model['model_accuracy']:.1%})")
            print(f"   📊 Range: {best_model['model_accuracy'] - worst_model['model_accuracy']:.1%}")
            print()
    
    # Avoid recommendations
    print(f"⚠️  MODELS TO AVOID")
    print("-" * 50)
    
    poor_performers = models_df[models_df['model_accuracy'] < overall['human_accuracy']]
    if len(poor_performers) > 0:
        for _, model in poor_performers.iterrows():
            print(f"❌ {model['model']}")
            print(f"   Accuracy: {model['model_accuracy']:.1%} ({model['accuracy_difference']:.1%} vs human)")
            print(f"   Reason: Performs worse than human baseline")
            print()
    
    # Statistical significance warnings
    unreliable_models = models_df[~models_df['accuracy_chi2_significant']]
    if len(unreliable_models) > 0:
        print(f"⚠️  STATISTICALLY UNRELIABLE (use with caution):")
        for _, model in unreliable_models.iterrows():
            print(f"   - {model['model']} (accuracy difference not statistically significant)")
        print()
    
    # Final recommendations
    print(f"💡 FINAL RECOMMENDATIONS")
    print("-" * 50)
    
    # Count statistically significant models
    reliable_models = models_df[
        (models_df['accuracy_chi2_significant']) & 
        (models_df['model_accuracy'] > overall['human_accuracy'])
    ]
    
    print(f"✅ Deploy-ready models: {len(reliable_models)}/{len(models_df)}")
    print(f"✅ Average improvement over humans: {models_df[models_df['model_accuracy'] > overall['human_accuracy']]['accuracy_difference'].mean():.1%}")
    print(f"✅ Average speed improvement: {models_df['time_speedup_factor'].mean():.1f}x")
    
    if len(reliable_models) > 0:
        print(f"***REMOVED***n🚀 RECOMMENDED DEPLOYMENT STRATEGY:")
        print(f"   1. Start with: {best_overall['model']} (best overall performance)")
        print(f"   2. For speed-critical: {fastest_accurate['model']}")
        print(f"   3. For accuracy-critical: {most_accurate['model']}")
        if robustness_scores:
            print(f"   4. For poor data quality: {most_robust[0]}")
        print(f"   5. Monitor performance and adjust based on specific use case")
    
    print("***REMOVED***n" + "=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_quick_insights(sys.argv[1])
    else:
        generate_quick_insights()
