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
        'base': '#9467bd',
        'variants': ['#9467bd', '#c5b0d5', '#d4c4e0', '#e0d0e8']
    },
    'o4': {
        'base': '#2ca02c',
        'variants': ['#2ca02c', '#98df8a', '#d9f2d9', '#90ee90']
    }
}

def get_provider_color(model_name: str, model_list: list) -> str:
    """Get color for a model based on its provider family"""
    model_lower = model_name.lower()
    if 'claude' in model_lower:
        provider = 'claude'
    elif 'gpt' in model_lower or 'openai' in model_lower:
        provider = 'gpt'
    elif 'deepseek' in model_lower:
        provider = 'deepseek'
    elif 'o4' in model_lower:
        provider = 'o4'
    else:
        provider = 'deepseek'
    provider_models = [m for m in model_list if
                       (provider == 'claude' and 'claude' in m.lower()) or
                       (provider == 'gpt' and ('gpt' in m.lower() or 'openai' in m.lower())) or
                       (provider == 'deepseek' and 'deepseek' in m.lower()) or
                       (provider == 'o4' and 'o4' in m.lower())]
    try:
        idx = provider_models.index(model_name)
    except ValueError:
        idx = 0
    variants = PROVIDER_COLORS[provider]['variants']
    return variants[idx % len(variants)]

def plot_complexity_performance_comparison(complexity_data, models, complexities, output_path):
    """Create complexity performance comparison chart: human vs each model"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Performance by Task Complexity', fontsize=16, fontweight='bold')
    x = np.arange(len(complexities))
    total = len(models) + 1
    width = 0.8 / total

    # Human baseline bars
    human_acc = [complexity_data[models[0]][c]['human_accuracy'] for c in complexities]
    human_time = [complexity_data[models[0]][c]['human_avg_time'] for c in complexities]
    axes[0].bar(x - (total/2 - 0.5)*width, human_acc,
                width, label='Human', color='#FF6B6B', edgecolor='black', linewidth=1.5)
    axes[1].bar(x - (total/2 - 0.5)*width, human_time,
                width, label='Human', color='#FF6B6B', edgecolor='black', linewidth=1.5)

    # Model bars
    for i, model in enumerate(models):
        accs = [complexity_data[model][c]['model_accuracy'] for c in complexities]
        times = [complexity_data[model][c]['model_avg_time'] for c in complexities]
        color = get_provider_color(model, models)
        axes[0].bar(x - (total/2 - 0.5 - (i+1))*width, accs,
                    width, label=model, color=color, edgecolor='black', linewidth=1.5)
        axes[1].bar(x - (total/2 - 0.5 - (i+1))*width, times,
                    width, label=model, color=color, edgecolor='black', linewidth=1.5)

    # Accuracy subplot
    axes[0].set_title('Accuracy by Complexity')
    axes[0].set_xlabel('Task Complexity')
    axes[0].set_ylabel('Accuracy Rate')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.capitalize() for c in complexities])
    axes[0].legend(bbox_to_anchor=(1.05,1), loc='upper left')
    axes[0].set_ylim(0,1)
    axes[0].grid(True, alpha=0.3)

    # Time subplot
    axes[1].set_title('Average Time by Complexity')
    axes[1].set_xlabel('Task Complexity')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.capitalize() for c in complexities])
    axes[1].legend(bbox_to_anchor=(1.05,1), loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_quality_comparison_bars(quality_data, models, qualities, output_path):
    """Create quality comparison bar charts: human vs each model"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Performance by Data Quality Condition', fontsize=16, fontweight='bold')
    x = np.arange(len(qualities))
    total = len(models) + 1
    width = 0.8 / total

    # Human baseline bars
    human_acc = [quality_data[models[0]][q]['human_accuracy'] for q in qualities]
    human_time = [quality_data[models[0]][q]['human_avg_time'] for q in qualities]
    axes[0].bar(x - (total/2 - 0.5)*width, human_acc,
                width, label='Human', color='#FF6B6B', edgecolor='black', linewidth=1.5)
    axes[1].bar(x - (total/2 - 0.5)*width, human_time,
                width, label='Human', color='#FF6B6B', edgecolor='black', linewidth=1.5)

    # Model bars
    for i, model in enumerate(models):
        accs = [quality_data[model][q]['model_accuracy'] for q in qualities]
        times = [quality_data[model][q]['model_avg_time'] for q in qualities]
        color = get_provider_color(model, models)
        axes[0].bar(x - (total/2 - 0.5 - (i+1))*width, accs,
                    width, label=model, color=color, edgecolor='black', linewidth=1.5)
        axes[1].bar(x - (total/2 - 0.5 - (i+1))*width, times,
                    width, label=model, color=color, edgecolor='black', linewidth=1.5)

    # Quality condition labels (matching the complexity format)
    quality_labels = ['Normal***REMOVED***nBaseline', 'Spaces', 'Missing***REMOVED***nChars', 'Missing***REMOVED***nRecords']

    # Accuracy subplot
    axes[0].set_title('Accuracy by Data Quality')
    axes[0].set_xlabel('Data Quality Condition')
    axes[0].set_ylabel('Accuracy Rate')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(quality_labels)
    axes[0].legend(bbox_to_anchor=(1.05,1), loc='upper left')
    axes[0].set_ylim(0,1)
    axes[0].grid(True, alpha=0.3)

    # Time subplot
    axes[1].set_title('Average Time by Data Quality')
    axes[1].set_xlabel('Data Quality Condition')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(quality_labels)
    axes[1].legend(bbox_to_anchor=(1.05,1), loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_time_ranking(model_data, models, output_path):
    """Create model time ranking chart: completion time in seconds vs human baseline"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    human_time = model_data[models[0]]['human_avg_time']

    # Sort models by completion time (ascending - faster is better)
    model_time_sorted = sorted([(model, model_data[model]['model_avg_time']) for model in models],
                              key=lambda x: x[1])

    model_names = [item[0].replace('-', '***REMOVED***n') for item in model_time_sorted]
    model_times = [item[1] for item in model_time_sorted]

    # Color bars based on performance vs human (green if faster, red if slower)
    bar_colors = ['green' if time < human_time else 'red' for time in model_times]

    bars = ax.barh(range(len(model_names)), model_times, color=bar_colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=human_time, color='red', linestyle='--', linewidth=3, label=f'Human Baseline ({human_time:.1f}s)')

    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=11)
    ax.set_xlabel('Completion Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Model Time Ranking vs Human Baseline', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, model_times)):
        ax.text(time + max(model_times) * 0.01, bar.get_y() + bar.get_height()/2, f'{time:.1f}s',
               va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_confidence_ranking(model_data, models, output_path):
    """Create model confidence ranking chart: average confidence levels"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Sort models by average confidence (descending - higher is better)
    model_confidence_sorted = sorted([(model, model_data[model]['model_avg_confidence']) for model in models],
                                   key=lambda x: x[1], reverse=True)

    model_names = [item[0].replace('-', '***REMOVED***n') for item in model_confidence_sorted]
    model_confidences = [item[1] for item in model_confidence_sorted]

    # Color bars based on confidence level (green=high, yellow=medium, red=low)
    bar_colors = []
    for conf in model_confidences:
        if conf >= 0.7:
            bar_colors.append('#2E8B57')  # Dark green for high confidence
        elif conf >= 0.5:
            bar_colors.append('#FFD700')  # Gold for medium confidence
        else:
            bar_colors.append('#DC143C')  # Crimson for low confidence

    bars = ax.barh(range(len(model_names)), model_confidences, color=bar_colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=11)
    ax.set_xlabel('Average Confidence Level', fontsize=12, fontweight='bold')
    ax.set_title('Model Confidence Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, conf) in enumerate(zip(bars, model_confidences)):
        ax.text(conf + 0.02, bar.get_y() + bar.get_height()/2, f'{conf:.3f}',
               va='center', fontweight='bold')

    # Add confidence level legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', alpha=0.7, label='High (≥0.7)'),
        Patch(facecolor='#FFD700', alpha=0.7, label='Medium (0.5-0.7)'),
        Patch(facecolor='#DC143C', alpha=0.7, label='Low (<0.5)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_by_quality(quality_data, models, qualities, output_path):
    """Create confidence comparison by data quality condition"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    x = np.arange(len(qualities))
    total = len(models)
    width = 0.8 / total

    # Quality condition labels
    quality_labels = ['Normal***REMOVED***nBaseline', 'Spaces', 'Missing***REMOVED***nChars', 'Missing***REMOVED***nRecords']

    for i, model in enumerate(models):
        confidences = [quality_data[model][q]['model_avg_confidence'] for q in qualities]
        color = get_provider_color(model, models)
        ax.bar(x - (total/2 - 0.5 - i)*width, confidences,
               width, label=model, color=color, edgecolor='black', linewidth=1.5)

    ax.set_title('Model Confidence by Data Quality Condition', fontsize=14, fontweight='bold')
    ax.set_xlabel('Data Quality Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Confidence Level', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(quality_labels)
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_by_complexity(complexity_data, models, complexities, output_path):
    """Create confidence comparison by task complexity"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    x = np.arange(len(complexities))
    total = len(models)
    width = 0.8 / total

    for i, model in enumerate(models):
        confidences = [complexity_data[model][c]['model_avg_confidence'] for c in complexities]
        color = get_provider_color(model, models)
        ax.bar(x - (total/2 - 0.5 - i)*width, confidences,
               width, label=model, color=color, edgecolor='black', linewidth=1.5)

    ax.set_title('Model Confidence by Task Complexity', fontsize=14, fontweight='bold')
    ax.set_xlabel('Task Complexity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Confidence Level', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in complexities])
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Remaining individual visualization functions retained unchanged below
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
    human_times = []
    for complexity in complexities:
        human_time = 0
        for model in models:
            if complexity in complexity_data.get(model, {}):
                human_time = complexity_data[model][complexity].get('human_time', 0)
                break
        human_times.append(human_time)
    time_matrix = []
    for model in models:
        row = []
        for i, complexity in enumerate(complexities):
            if complexity in complexity_data.get(model, {}):
                speed = complexity_data[model][complexity].get('time_speedup_factor', 1)
                model_time = human_times[i] / speed if speed else 0
                row.append(model_time)
            else:
                row.append(0)
        time_matrix.append(row)
    time_array = np.array(time_matrix)
    im = ax.imshow(time_array, cmap='YlOrRd_r', aspect='auto')
    ax.set_title('Model Response Time by Task Complexity', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(complexities)))
    ax.set_xticklabels([c.capitalize() for c in complexities])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=11)
    for i, human_time in enumerate(human_times):
        ax.axhline(y=i, color='red', linestyle='--', alpha=0.7, xmin=0, xmax=1)
        ax.text(0, i, f'Human: {human_time:.1f}s',
                ha='right', va='center', color='red', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    for i in range(len(models)):
        for j in range(len(complexities)):
            if time_array[i, j] > 0:
                text = f'{time_array[i, j]:.1f}s'
                color = 'black' if time_array[i, j] < human_times[j]/2 else 'white'
                ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Response Time (seconds)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_best_models_by_complexity(complexity_data, models, complexities, output_path):
    """Create best models by complexity chart"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    best_models = {}
    for complexity in complexities:
        best_acc, best_model = 0, None
        for m in models:
            if complexity in complexity_data[m]:
                a = complexity_data[m][complexity]['model_accuracy']
                if a > best_acc:
                    best_acc, best_model = a, m
        best_models[complexity] = (best_model, best_acc)
    best_accs = [best_models[c][1] for c in complexities]
    best_names = [best_models[c][0].replace('-', '***REMOVED***n') if best_models[c][0] else 'None' for c in complexities]
    colors = ['#FF6B6B','#4ECDC4','#45B7D1']
    bars = ax.bar(complexities, best_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_title('Best Performing Model by Task Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Task Complexity', fontsize=12, fontweight='bold')
    ax.set_ylim(0,1)
    for bar, name, acc in zip(bars,best_names,best_accs):
        ax.text(bar.get_x()+bar.get_width()/2, acc+0.02, name, ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.text(bar.get_x()+bar.get_width()/2, acc/2, f'{acc:.3f}', ha='center', va='center', color='white', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_quality_accuracy_heatmap(quality_data, models, qualities, output_path):
    """Create quality accuracy heatmap"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    accuracy_matrix = []
    for m in models:
        row=[]
        for q in qualities:
            row.append(quality_data[m][q]['model_accuracy'] if q in quality_data[m] else 0)
        accuracy_matrix.append(row)
    arr = np.array(accuracy_matrix)
    im = ax.imshow(arr, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_title('Model Accuracy by Data Quality Condition', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(qualities))); ax.set_xticklabels(qualities)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=11)
    for i in range(len(models)):
        for j in range(len(qualities)):
            if arr[i,j]>0:
                text=f'{arr[i,j]:.3f}'
                c='white' if arr[i,j]<0.5 else 'black'
                ax.text(j,i,text,ha='center',va='center',color=c,fontweight='bold')
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_quality_robustness_ranking(quality_data, models, output_path):
    """Create quality robustness ranking chart"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    scores=[]
    for m in models:
        if 'Q0' in quality_data[m]:
            q0=quality_data[m]['Q0']['model_accuracy']
            corrupted=[quality_data[m][q]['model_accuracy'] for q in ['Q1','Q2','Q3'] if q in quality_data[m]]
            if corrupted:
                scores.append((m, np.mean(corrupted)/q0 if q0 else 0))
    scores.sort(key=lambda x: x[1], reverse=True)
    names=[s[0].replace('-', '***REMOVED***n') for s in scores]; vals=[s[1] for s in scores]
    cols=['green' if v>0.8 else 'orange' if v>0.6 else 'red' for v in vals]
    bars=ax.barh(range(len(names)),vals,color=cols,alpha=0.7,edgecolor='black')
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=11)
    ax.set_xlabel('Robustness Score (Corrupted/Perfect)',fontsize=12,fontweight='bold')
    ax.set_title('Data Quality Robustness Ranking',fontsize=14,fontweight='bold')
    ax.axvline(0.8,color='green',linestyle='--',alpha=0.7,label='Good (>0.8)')
    ax.axvline(0.6,color='orange',linestyle='--',alpha=0.7,label='Moderate (>0.6)')
    ax.legend(); ax.grid(True,alpha=0.3,axis='x')
    for bar,v in zip(bars,vals):
        ax.text(v+0.02,bar.get_y()+bar.get_height()/2,f'{v:.3f}',va='center',fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_human_performance_distribution(human_data, output_path):
    """Create distribution graph of human participant performance"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    accuracies = [r['accuracy'] for r in human_data]
    sns.histplot(accuracies, kde=True, color='skyblue', bins=15, stat='density', alpha=0.5)
    mean_acc, median_acc = np.mean(accuracies), np.median(accuracies)
    ax.axvline(mean_acc, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_acc:.3f}')
    ax.axvline(median_acc, color='green', linestyle='-', linewidth=2, label=f'Median: {median_acc:.3f}')
    ax.set_title('Human Participant Performance Distribution',fontsize=14,fontweight='bold')
    ax.set_xlabel('Accuracy',fontsize=12,fontweight='bold')
    ax.set_ylabel('Density',fontsize=12,fontweight='bold')
    ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_failure_classification_analysis(task_data, models, complexities, qualities, output_dir):
    """Create failure classification analysis charts and tables"""
    task_data['failure_type']='success'
    task_data.loc[task_data['llm_accuracy']<0.3,'failure_type']='critical'
    task_data.loc[(task_data['llm_accuracy']>=0.3)&(task_data['llm_accuracy']<0.6),'failure_type']='major'
    task_data.loc[(task_data['llm_accuracy']>=0.6)&(task_data['llm_accuracy']<0.8),'failure_type']='minor'
    model_families=sorted({m.split('-')[0] for m in models})
    family_colors=plt.cm.tab10(np.linspace(0,1,len(model_families)))
    fig1, ax1 = plt.subplots(1,1,figsize=(14,8))
    for i,f in enumerate(model_families):
        fm=[m for m in models if m.startswith(f)]
        fd=task_data[task_data['llm_model'].isin(fm)]
        if not fd.empty:
            sns.histplot(data=fd,x='failure_type',hue='failure_type',palette=[family_colors[i]],stat='percent',discrete=True,ax=ax1,label=f,alpha=0.7)
    ax1.set_title('Failure Distribution by Model Family',fontsize=16)
    ax1.set_xlabel('Failure Severity',fontsize=14); ax1.set_ylabel('Percentage of Tasks',fontsize=14)
    ax1.legend(title='Model Family'); ax1.grid(True,linestyle='--',alpha=0.7)
    # additional failure plots omitted for brevity...
    plt.tight_layout()
    plt.savefig(f"{output_dir}/failure_by_model_family.png",dpi=300,bbox_inches='tight')
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
