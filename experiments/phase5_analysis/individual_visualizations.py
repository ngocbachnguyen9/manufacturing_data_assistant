#!/usr/bin/env python3
"""
Individual Visualization Methods for Phase 5 Analysis

This module contains individual chart methods that can be integrated into the main comparison framework.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_complexity_performance_comparison(complexity_data, models, complexities, output_path):
    """Create complexity performance comparison chart"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    human_accs = []
    for complexity in complexities:
        if models and complexity in complexity_data[models[0]]:
            human_accs.append(complexity_data[models[0]][complexity]['human_accuracy'])
        else:
            human_accs.append(0)

    x = np.arange(len(complexities))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_accs = []
        for complexity in complexities:
            if complexity in complexity_data[model]:
                model_accs.append(complexity_data[model][complexity]['model_accuracy'])
            else:
                model_accs.append(0)

        ax.bar(x + i * width, model_accs, width, label=model.replace('-', '***REMOVED***n'), alpha=0.8)

    # Add human baseline
    ax.plot(x + width * (len(models)-1)/2, human_accs, 'ro-', linewidth=3, markersize=8, label='Human Baseline')

    ax.set_title('Model vs Human Performance by Task Complexity', fontsize=14, fontweight='bold')
    ax.set_xlabel('Task Complexity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * (len(models)-1)/2)
    ax.set_xticklabels([c.capitalize() for c in complexities])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
    """Create complexity speed analysis chart"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    speed_matrix = []
    for model in models:
        row = []
        for complexity in complexities:
            if complexity in complexity_data[model]:
                speed = complexity_data[model][complexity]['time_speedup_factor']
                row.append(min(speed, 25) if speed != float('inf') else 25)
            else:
                row.append(0)
        speed_matrix.append(row)

    speed_array = np.array(speed_matrix)
    im = ax.imshow(speed_array, cmap='YlOrRd', aspect='auto', vmin=0, vmax=25)
    ax.set_title('Speed Improvement by Task Complexity', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(complexities)))
    ax.set_xticklabels([c.capitalize() for c in complexities])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=11)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(complexities)):
            if speed_array[i, j] > 0:
                text = f'{speed_array[i, j]:.1f}x'
                color = 'white' if speed_array[i, j] > 12 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
