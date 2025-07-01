#!/usr/bin/env python3
"""
Create Comprehensive Visualizations for Individual Model Analysis
================================================================

This script creates additional visualizations to support the comprehensive
individual model analysis findings.

Author: Manufacturing Data Assistant
Date: 2025-06-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_model_tier_visualization():
    """Create visualization showing model tiers and deployment readiness"""
    
    # Load model data
    results_dir = Path("results")
    model_data = pd.read_csv(results_dir / "model_specific_comparison.csv")
    
    # Define tiers based on analysis
    tier1_models = ['claude-sonnet-4-20250514', 'deepseek-reasoner', 'claude-3-5-haiku-latest', 'deepseek-chat']
    tier2_models = ['o4-mini-2025-04-16']
    tier3_models = ['gpt-4o-mini-2024-07-18']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Accuracy vs Speed with tier coloring
    colors = []
    for model in model_data['model']:
        if model in tier1_models:
            colors.append('green')
        elif model in tier2_models:
            colors.append('orange')
        else:
            colors.append('red')
    
    scatter = ax1.scatter(model_data['time_speedup_factor'], 
                         model_data['model_accuracy'],
                         c=colors, s=200, alpha=0.7, edgecolors='black')
    
    # Add model labels
    for i, model in enumerate(model_data['model']):
        short_name = model.split('-')[0]
        ax1.annotate(short_name, 
                    (model_data['time_speedup_factor'].iloc[i], 
                     model_data['model_accuracy'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Add human baseline
    human_acc = model_data['human_accuracy'].iloc[0]
    ax1.axhline(y=human_acc, color='black', linestyle='--', alpha=0.7, label='Human Baseline')
    
    ax1.set_xlabel('Speed Improvement Factor (x)')
    ax1.set_ylabel('Model Accuracy')
    ax1.set_title('Model Performance Tiers***REMOVED***n(Green=Tier 1, Orange=Tier 2, Red=Tier 3)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Deployment readiness radar
    models = model_data['model'].tolist()
    short_names = [m.split('-')[0] for m in models]
    
    # Create deployment readiness scores
    readiness_scores = []
    for _, row in model_data.iterrows():
        accuracy_score = min(row['model_accuracy'] / human_acc, 2.0)  # Cap at 2x human
        speed_score = min(row['time_speedup_factor'] / 20, 1.0)  # Normalize to max 20x
        significance_score = 1.0 if row['accuracy_chi2_significant'] else 0.5
        
        overall_score = (accuracy_score * 0.5 + speed_score * 0.3 + significance_score * 0.2)
        readiness_scores.append(overall_score)
    
    bars = ax2.bar(short_names, readiness_scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Deployment Readiness Score')
    ax2.set_title('Model Deployment Readiness')
    ax2.set_ylim(0, 2.0)
    
    # Add score labels on bars
    for bar, score in zip(bars, readiness_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/model_deployment_tiers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created model deployment tiers visualization")

def create_risk_assessment_matrix():
    """Create risk assessment matrix visualization"""
    
    # Load data
    results_dir = Path("results")
    model_data = pd.read_csv(results_dir / "model_specific_comparison.csv")
    complexity_data = pd.read_csv(results_dir / "model_complexity_analysis.csv")
    quality_data = pd.read_csv(results_dir / "model_quality_analysis.csv")
    
    # Calculate risk metrics
    models = model_data['model'].tolist()
    risk_matrix = []
    
    for model in models:
        model_row = model_data[model_data['model'] == model].iloc[0]
        
        # Accuracy risk (lower accuracy = higher risk)
        accuracy_risk = max(0, 1 - model_row['model_accuracy'])
        
        # Consistency risk (variance across complexity)
        model_complexity = complexity_data[complexity_data['model'] == model]
        if len(model_complexity) > 0:
            complexity_accuracies = model_complexity['model_accuracy'].values
            consistency_risk = np.std(complexity_accuracies)
        else:
            consistency_risk = 0.5
        
        # Robustness risk (performance degradation with poor data)
        model_quality = quality_data[quality_data['model'] == model]
        if len(model_quality) > 0:
            q0_acc = model_quality[model_quality['quality_condition'] == 'Q0']['model_accuracy'].iloc[0]
            corrupted_accs = model_quality[model_quality['quality_condition'].isin(['Q1', 'Q2', 'Q3'])]['model_accuracy'].values
            if len(corrupted_accs) > 0:
                robustness_risk = max(0, q0_acc - np.mean(corrupted_accs))
            else:
                robustness_risk = 0.3
        else:
            robustness_risk = 0.3
        
        # Statistical significance risk
        significance_risk = 0.1 if model_row['accuracy_chi2_significant'] else 0.5
        
        risk_matrix.append([accuracy_risk, consistency_risk, robustness_risk, significance_risk])
    
    # Create heatmap
    risk_df = pd.DataFrame(risk_matrix, 
                          columns=['Accuracy Risk', 'Consistency Risk', 'Robustness Risk', 'Significance Risk'],
                          index=[m.split('-')[0] for m in models])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(risk_df, annot=True, cmap='RdYlGn_r', center=0.3, 
                cbar_kws={'label': 'Risk Level'}, fmt='.3f')
    plt.title('Model Risk Assessment Matrix***REMOVED***n(Lower values = Lower risk)')
    plt.ylabel('Models')
    plt.xlabel('Risk Categories')
    plt.tight_layout()
    plt.savefig('results/risk_assessment_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created risk assessment matrix")

def create_use_case_recommendation_chart():
    """Create use case recommendation chart"""
    
    # Define use cases and recommended models
    use_cases = {
        'Critical Safety': 'deepseek-reasoner',
        'High Volume': 'claude-3-5-haiku-latest', 
        'Real-time QC': 'claude-3-5-haiku-latest',
        'Poor Data Quality': 'deepseek-reasoner',
        'Cost Sensitive': 'claude-3-5-haiku-latest',
        'Regulatory': 'claude-sonnet-4-20250514',
        'Pilot Testing': 'claude-sonnet-4-20250514',
        'Balanced Performance': 'claude-sonnet-4-20250514'
    }
    
    # Load model performance data
    results_dir = Path("results")
    model_data = pd.read_csv(results_dir / "model_specific_comparison.csv")
    
    # Create recommendation matrix
    models = model_data['model'].unique()
    model_short_names = [m.split('-')[0] for m in models]
    
    recommendation_matrix = np.zeros((len(use_cases), len(models)))
    
    for i, (use_case, recommended_model) in enumerate(use_cases.items()):
        for j, model in enumerate(models):
            if model == recommended_model:
                recommendation_matrix[i, j] = 1.0
            else:
                # Calculate suitability score based on model characteristics
                model_row = model_data[model_data['model'] == model].iloc[0]
                
                if use_case in ['Critical Safety', 'Regulatory']:
                    score = model_row['model_accuracy'] * 0.8 + (1 if model_row['accuracy_chi2_significant'] else 0) * 0.2
                elif use_case in ['High Volume', 'Real-time QC', 'Cost Sensitive']:
                    score = min(model_row['time_speedup_factor'] / 25, 1.0) * 0.6 + model_row['model_accuracy'] * 0.4
                else:
                    score = model_row['model_accuracy'] * 0.5 + min(model_row['time_speedup_factor'] / 25, 1.0) * 0.5
                
                recommendation_matrix[i, j] = score
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(recommendation_matrix, 
                xticklabels=model_short_names,
                yticklabels=list(use_cases.keys()),
                annot=True, cmap='RdYlGn', center=0.5,
                cbar_kws={'label': 'Suitability Score'}, fmt='.2f')
    plt.title('Model Suitability by Use Case***REMOVED***n(1.0 = Recommended, 0.5 = Neutral, 0.0 = Not Suitable)')
    plt.xlabel('Models')
    plt.ylabel('Use Cases')
    plt.tight_layout()
    plt.savefig('results/use_case_recommendations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created use case recommendation chart")

def create_confidence_interval_plot():
    """Create confidence interval visualization for model performance"""
    
    # Load data
    results_dir = Path("results")
    model_data = pd.read_csv(results_dir / "model_specific_comparison.csv")
    
    # Calculate confidence intervals (using sample sizes and assuming normal distribution)
    models = model_data['model'].tolist()
    short_names = [m.split('-')[0] for m in models]
    accuracies = model_data['model_accuracy'].tolist()
    sample_sizes = model_data['model_sample_size'].tolist()
    
    # Calculate 95% confidence intervals
    confidence_intervals = []
    for acc, n in zip(accuracies, sample_sizes):
        # Standard error for proportion
        se = np.sqrt(acc * (1 - acc) / n)
        # 95% CI
        margin = 1.96 * se
        ci_lower = max(0, acc - margin)
        ci_upper = min(1, acc + margin)
        confidence_intervals.append((ci_lower, ci_upper))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot confidence intervals
    for i, (model, acc, (ci_low, ci_high)) in enumerate(zip(short_names, accuracies, confidence_intervals)):
        ax.errorbar(i, acc, yerr=[[acc - ci_low], [ci_high - acc]], 
                   fmt='o', capsize=5, capthick=2, markersize=8)
        
        # Add sample size annotation
        ax.annotate(f'n={sample_sizes[i]}', (i, acc), 
                   xytext=(0, 15), textcoords='offset points', 
                   ha='center', fontsize=9, alpha=0.7)
    
    # Add human baseline
    human_acc = model_data['human_accuracy'].iloc[0]
    human_n = model_data['human_sample_size'].iloc[0]
    human_se = np.sqrt(human_acc * (1 - human_acc) / human_n)
    human_ci = 1.96 * human_se
    
    ax.axhline(y=human_acc, color='red', linestyle='--', alpha=0.7, label='Human Baseline')
    ax.fill_between(range(len(models)), human_acc - human_ci, human_acc + human_ci, 
                   color='red', alpha=0.2, label='Human 95% CI')
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=45)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy with 95% Confidence Intervals')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('results/confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created confidence interval plot")

if __name__ == "__main__":
    print("Creating comprehensive visualizations...")
    
    create_model_tier_visualization()
    create_risk_assessment_matrix()
    create_use_case_recommendation_chart()
    create_confidence_interval_plot()
    
    print("***REMOVED***n✅ All comprehensive visualizations created!")
    print("📁 Saved to results/ directory:")
    print("   - model_deployment_tiers.png")
    print("   - risk_assessment_matrix.png") 
    print("   - use_case_recommendations.png")
    print("   - confidence_intervals.png")
