#!/usr/bin/env python3
"""
Individual Visualization Methods for Phase 5 Analysis

This module contains individual chart methods that can be integrated into the main comparison framework.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# Provider-based color mapping for consistent visualization
PROVIDER_COLORS = {
    'claude': {
        'base': '#1f77b4',
        'variants': ['#1f77b4', '#17becf', '#aec7e8', '#c5dbf1']
    },
    'gpt': {
        'base': '#2ca02c', 
        'variants': ['#2ca02c', '#98df8a', '#d9f2d9', '#90ee90']
    },
    'openai': {
        'base': '#2ca02c',
        'variants': ['#2ca02c', '#98df8a', '#d9f2d9', '#90ee90'] 
    },
    'deepseek': {
        'base': '#ff7f0e',
        'variants': ['#ff7f0e', '#ffbb78', '#ffd4aa', '#ffe5cc']
    },
    'o4': {
        'base': '#9467bd',
        'variants': ['#9467bd', '#c5b0d5', '#d4c4e0', '#e0d0e8']
    }
}

def get_provider_color(model_name: str, model_list: list) -> str:
    """Get color for a model based on its provider family"""
    model_lower = model_name.lower()
    
    # Determine provider
    if 'claude' in model_lower:
        provider = 'claude'
    elif 'gpt' in model_lower or 'openai' in model_lower:
        provider = 'gpt'
    elif 'deepseek' in model_lower:
        provider = 'deepseek'
    elif 'o4' in model_lower:
        provider = 'o4'
    else:
        provider = 'deepseek'  # Default fallback
    
    # Get models from same provider
    provider_models = [m for m in model_list if 
                      (provider == 'claude' and 'claude' in m.lower()) or
                      (provider == 'gpt' and ('gpt' in m.lower() or 'openai' in m.lower())) or
                      (provider == 'deepseek' and 'deepseek' in m.lower()) or
                      (provider == 'o4' and 'o4' in m.lower())]
    
    # Get index within provider family
    try:
        provider_index = provider_models.index(model_name)
    except ValueError:
        provider_index = 0
    
    # Return appropriate color variant
    color_variants = PROVIDER_COLORS[provider]['variants']
    return color_variants[provider_index % len(color_variants)]

def plot_complexity_performance_comparison(complexity_data, models, complexities, output_path):
    """Create complexity performance comparison chart with grouped model families"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Extract model families (first part of model name)
    model_families = sorted(set([model.split('-')[0] for model in models]))
    family_colors = plt.cm.tab10(np.linspace(0, 1, len(model_families)))
    family_color_map = {family: color for family, color in zip(model_families, family_colors)}
    
    # Get human accuracy per complexity
    human_accs = []
    for complexity in complexities:
        for model in models:
            if complexity in complexity_data.get(model, {}):
                human_accs.append(complexity_data[model][complexity]['human_accuracy'])
                break
        else:
            human_accs.append(0)

    x = np.arange(len(complexities))
    total_models = len(models)
    bar_width = 0.8 / (total_models + 1)  # +1 for human baseline bar
    
    # Plot human baseline bars
    human_bars = ax.bar(x, human_accs, bar_width, color='red', alpha=0.7,
                        edgecolor='black', linewidth=1.5, label='Human Baseline')
    
    # Plot model bars grouped by family with similar hues
    for model_idx, model in enumerate(models):
        model_accs = []
        for complexity in complexities:
            if complexity in complexity_data.get(model, {}):
                model_accs.append(complexity_data[model][complexity]['model_accuracy'])
            else:
                model_accs.append(0)
        
        family = model.split('-')[0]
        base_color = family_color_map[family]
        # Create slightly different hue for each model in family
        hue_factor = 0.7 + 0.3 * (model_idx % 3) / 3
        model_color = tuple([c * hue_factor for c in base_color[:3]] + [base_color[3]])
        
        # Position model bars to the right of human baseline
        model_bars = ax.bar(x + (model_idx + 1) * bar_width, model_accs, bar_width,
                            color=model_color, alpha=0.8, edgecolor='black', linewidth=1.5,
                            label=model.replace('-', '***REMOVED***n'))

    ax.set_title('Model vs Human Performance by Task Complexity', fontsize=14, fontweight='bold')
    ax.set_xlabel('Task Complexity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(x + bar_width * (total_models + 1) / 2)
    ax.set_xticklabels([c.capitalize() for c in complexities])
    
    # Create custom legend with family colors
    legend_handles = [plt.Rectangle((0,0), 1, 1, color=family_color_map[family], label=family)
                     for family in model_families]
    legend_handles.append(plt.Rectangle((0,0), 1, 1, color='red', label='Human Baseline'))
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_complexity_improvement_heatmap(complexity_data, models, complexities, output_path):
    """Create complexity improvement heatmap"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    improvement_matrix = []
    for model in models:
        row = []
        for complexity in complexities:
            if complexity in complexity_data[model]:
                row.append(complexity_data[model][complexity]['accuracy_difference'])
            else:
                row.append(0)
        improvement_matrix.append(row)

    improvement_array = np.array(improvement_matrix)
    im = ax.imshow(improvement_array, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_title('Accuracy Improvement over Humans by Complexity', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(complexities)))
    ax.set_xticklabels([c.capitalize() for c in complexities])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=11)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(complexities)):
            if improvement_array[i, j] != 0:
                text = f'{improvement_array[i, j]:+.3f}'
                color = 'white' if abs(improvement_array[i, j]) > 0.25 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_complexity_speed_analysis(complexity_data, models, complexities, output_path):
    """Create complexity speed analysis chart showing time in seconds"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get human times per complexity with fallback to 0
    human_times = []
    for complexity in complexities:
        human_time = 0
        for model in models:
            if complexity in complexity_data.get(model, {}):
                human_time = complexity_data[model][complexity].get('human_time', 0)
                break
        human_times.append(human_time)
    
    # Calculate model times in seconds (time = human_time / speed_factor)
    time_matrix = []
    for model in models:
        row = []
        for i, complexity in enumerate(complexities):
            if complexity in complexity_data.get(model, {}):
                # Safely get values with defaults
                speed_factor = complexity_data[model][complexity].get('time_speedup_factor', 1)
                human_time = human_times[i]
                model_time = human_time / speed_factor if speed_factor != 0 else 0
                row.append(model_time)
            else:
                row.append(0)
        time_matrix.append(row)

    time_array = np.array(time_matrix)
    im = ax.imshow(time_array, cmap='YlOrRd_r', aspect='auto')  # _r for reversed colormap
    ax.set_title('Model Response Time by Task Complexity', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(complexities)))
    ax.set_xticklabels([c.capitalize() for c in complexities])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=11)
    
    # Add human baseline dashed lines
    for i, human_time in enumerate(human_times):
        ax.axhline(y=i, color='red', linestyle='--', alpha=0.7, xmin=0, xmax=1)
        ax.text(0, i, f'Human: {human_time:.1f}s',
                ha='right', va='center', color='red', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Add text annotations (model times)
    for i in range(len(models)):
        for j in range(len(complexities)):
            if time_array[i, j] > 0:
                text = f'{time_array[i, j]:.1f}s'
                color = 'black' if time_array[i, j] < human_times[j]/2 else 'white'
                ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

    # Add colorbar with seconds label
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Response Time (seconds)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_best_models_by_complexity(complexity_data, models, complexities, output_path):
    """Create best models by complexity chart"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    best_models_by_complexity = {}
    for complexity in complexities:
        best_acc = 0
        best_model = None
        for model in models:
            if complexity in complexity_data[model]:
                acc = complexity_data[model][complexity]['model_accuracy']
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
        best_models_by_complexity[complexity] = (best_model, best_acc)

    best_accs = [best_models_by_complexity[c][1] for c in complexities]
    best_names = [best_models_by_complexity[c][0].replace('-', '***REMOVED***n') if best_models_by_complexity[c][0] else 'None' for c in complexities]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(complexities, best_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_title('Best Performing Model by Task Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Task Complexity', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)

    # Add model names and accuracy values
    for i, (bar, name, acc) in enumerate(zip(bars, best_names, best_accs)):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02, 
               name, ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.text(bar.get_x() + bar.get_width()/2, acc/2, 
               f'{acc:.3f}', ha='center', va='center', fontweight='bold', color='white', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_quality_accuracy_heatmap(quality_data, models, qualities, output_path):
    """Create quality accuracy heatmap"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    accuracy_matrix = []
    for model in models:
        row = []
        for quality in qualities:
            if quality in quality_data[model]:
                row.append(quality_data[model][quality]['model_accuracy'])
            else:
                row.append(0)
        accuracy_matrix.append(row)

    accuracy_array = np.array(accuracy_matrix)
    im = ax.imshow(accuracy_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Model Accuracy by Data Quality Condition', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(qualities)))
    ax.set_xticklabels(['Normal***REMOVED***nBaseline', 'Spaces', 'Missing***REMOVED***nChars', 'Missing***REMOVED***nRecords'])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=11)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(qualities)):
            if accuracy_array[i, j] > 0:
                text = f'{accuracy_array[i, j]:.3f}'
                color = 'white' if accuracy_array[i, j] < 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_quality_robustness_ranking(quality_data, models, output_path):
    """Create quality robustness ranking chart"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    robustness_scores = []
    for model in models:
        if 'Q0' in quality_data[model]:
            q0_acc = quality_data[model]['Q0']['model_accuracy']
            corrupted_accs = []
            for q in ['Q1', 'Q2', 'Q3']:
                if q in quality_data[model]:
                    corrupted_accs.append(quality_data[model][q]['model_accuracy'])
            
            if corrupted_accs:
                avg_corrupted = np.mean(corrupted_accs)
                robustness = avg_corrupted / q0_acc if q0_acc > 0 else 0
                robustness_scores.append((model, robustness, q0_acc, avg_corrupted))

    # Sort by robustness score
    robustness_scores.sort(key=lambda x: x[1], reverse=True)

    if robustness_scores:
        rob_models = [item[0].replace('-', '***REMOVED***n') for item in robustness_scores]
        rob_scores = [item[1] for item in robustness_scores]
        
        colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' for score in rob_scores]
        bars = ax.barh(range(len(rob_models)), rob_scores, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(range(len(rob_models)))
        ax.set_yticklabels(rob_models, fontsize=11)
        ax.set_xlabel('Robustness Score (Corrupted/Perfect)', fontsize=12, fontweight='bold')
        ax.set_title('Data Quality Robustness Ranking', fontsize=14, fontweight='bold')
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Good (>0.8)')
        ax.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate (>0.6)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, rob_scores)):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_human_performance_distribution(human_data, output_path):
    """Create distribution graph of human participant performance"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Extract accuracy values
    accuracies = [result['accuracy'] for result in human_data]
    
    # Plot distribution
    sns.histplot(accuracies, kde=True, color='skyblue', bins=15, stat='density', alpha=0.5)
    
    # Calculate and plot mean and median
    mean_acc = np.mean(accuracies)
    median_acc = np.median(accuracies)
    ax.axvline(mean_acc, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_acc:.3f}')
    ax.axvline(median_acc, color='green', linestyle='-', linewidth=2, label=f'Median: {median_acc:.3f}')
    
    ax.set_title('Human Participant Performance Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_failure_classification_analysis(task_data, models, complexities, qualities, output_dir):
    """Create failure classification analysis charts and tables"""
    
    # Classify failures based on accuracy thresholds
    task_data['failure_type'] = 'success'
    task_data.loc[task_data['llm_accuracy'] < 0.3, 'failure_type'] = 'critical'
    task_data.loc[(task_data['llm_accuracy'] >= 0.3) & (task_data['llm_accuracy'] < 0.6), 'failure_type'] = 'major'
    task_data.loc[(task_data['llm_accuracy'] >= 0.6) & (task_data['llm_accuracy'] < 0.8), 'failure_type'] = 'minor'
    
    # 1. Failure distribution by model family
    model_families = sorted(set([model.split('-')[0] for model in models]))
    family_colors = plt.cm.tab10(np.linspace(0, 1, len(model_families)))
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    for i, family in enumerate(model_families):
        family_models = [m for m in models if m.startswith(family)]
        family_data = task_data[task_data['llm_model'].isin(family_models)]
        if not family_data.empty:
            sns.histplot(
                data=family_data, x='failure_type',
                hue='failure_type', palette=[family_colors[i]],
                stat='percent', discrete=True, ax=ax1,
                label=family, alpha=0.7
            )
    ax1.set_title('Failure Distribution by Model Family', fontsize=16)
    ax1.set_xlabel('Failure Severity', fontsize=14)
    ax1.set_ylabel('Percentage of Tasks', fontsize=14)
    ax1.legend(title='Model Family')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Failure patterns across complexity and quality
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(18, 8))
    
    # By complexity
    complexity_data = task_data.groupby(['complexity', 'failure_type']).size().unstack(fill_value=0)
    complexity_data = complexity_data.div(complexity_data.sum(axis=1), axis=0) * 100
    complexity_data.plot(kind='bar', stacked=True, ax=ax2, colormap='RdYlGn_r')
    ax2.set_title('Failure Distribution by Task Complexity', fontsize=16)
    ax2.set_xlabel('Complexity Level', fontsize=14)
    ax2.set_ylabel('Percentage', fontsize=14)
    ax2.legend(title='Failure Type')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # By quality condition
    quality_data = task_data.groupby(['quality_condition', 'failure_type']).size().unstack(fill_value=0)
    quality_data = quality_data.div(quality_data.sum(axis=1), axis=0) * 100
    quality_data.plot(kind='bar', stacked=True, ax=ax3, colormap='RdYlGn_r')
    ax3.set_title('Failure Distribution by Data Quality', fontsize=16)
    ax3.set_xlabel('Quality Condition', fontsize=14)
    ax3.set_ylabel('Percentage', fontsize=14)
    ax3.legend(title='Failure Type')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Detailed failure analysis by model
    fig3, ax4 = plt.subplots(1, 1, figsize=(16, 10))
    model_failure_data = task_data.groupby(['llm_model', 'failure_type']).size().unstack(fill_value=0)
    model_failure_data = model_failure_data.div(model_failure_data.sum(axis=1), axis=0) * 100
    model_failure_data = model_failure_data.sort_values(by='critical', ascending=False)
    
    model_failure_data.plot(kind='bar', stacked=True, ax=ax4, colormap='RdYlGn_r')
    ax4.set_title('Detailed Failure Analysis by Model', fontsize=18)
    ax4.set_xlabel('LLM Model', fontsize=14)
    ax4.set_ylabel('Failure Distribution (%)', fontsize=14)
    ax4.legend(title='Failure Type', loc='upper right')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    # Save charts
    fig1.savefig(f"{output_dir}/failure_by_model_family.png", dpi=300, bbox_inches='tight')
    fig2.savefig(f"{output_dir}/failure_patterns.png", dpi=300, bbox_inches='tight')
    fig3.savefig(f"{output_dir}/failure_by_model.png", dpi=300, bbox_inches='tight')
    
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    
    # Generate comprehensive failure report
    critical_failures = task_data[task_data['failure_type'] == 'critical']
    high_failure_models = critical_failures['llm_model'].value_counts().index[0]
    high_failure_complexity = critical_failures['complexity'].value_counts().index[0]
    high_failure_quality = critical_failures['quality_condition'].value_counts().index[0]
    
    report = f"# Failure Classification Report***REMOVED***n***REMOVED***n"
    report += f"## Key Findings***REMOVED***n"
    report += f"- **Critical failures** occurred in {len(critical_failures)} tasks ({len(critical_failures)/len(task_data)*100:.1f}% of all tasks)***REMOVED***n"
    report += f"- Models with highest critical failure rates: {high_failure_models}***REMOVED***n"
    report += f"- Most failure-prone complexity level: {high_failure_complexity}***REMOVED***n"
    report += f"- Most failure-prone quality condition: {high_failure_quality}***REMOVED***n***REMOVED***n"
    
    report += f"## Failure Pattern Analysis***REMOVED***n"
    report += f"1. **Model Family Patterns**:***REMOVED***n"
    for family in model_families:
        family_failures = critical_failures[critical_failures['llm_model'].str.startswith(family)]
        report += f"   - {family}: {len(family_failures)} critical failures***REMOVED***n"
    
    report += f"***REMOVED***n2. **Complexity Impact**:***REMOVED***n"
    for complexity in complexities:
        comp_failures = critical_failures[critical_failures['complexity'] == complexity]
        report += f"   - {complexity}: {len(comp_failures)} critical failures***REMOVED***n"
    
    report += f"***REMOVED***n3. **Quality Impact**:***REMOVED***n"
    for quality in qualities:
        qual_failures = critical_failures[critical_failures['quality_condition'] == quality]
        report += f"   - {quality}: {len(qual_failures)} critical failures***REMOVED***n"
    
    report += f"***REMOVED***n## Recommendations***REMOVED***n"
    report += f"1. Focus debugging efforts on {high_failure_models} for {high_failure_complexity} tasks with {high_failure_quality} data***REMOVED***n"
    report += f"2. Implement additional quality checks for {high_failure_quality} data conditions***REMOVED***n"
    report += f"3. Add specialized training for {high_failure_complexity} tasks in vulnerable models***REMOVED***n"
    report += f"4. Conduct root cause analysis on critical failure tasks: {', '.join(critical_failures['task_id'].sample(min(5, len(critical_failures)), random_state=42).tolist())}***REMOVED***n"
    
    # Save report
    with open(f"{output_dir}/failure_analysis_report.md", "w") as f:
        f.write(report)
