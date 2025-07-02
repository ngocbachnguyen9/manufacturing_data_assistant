#!/usr/bin/env python3
"""
Phase 5: Individual Visualizations Runner

This script generates all Phase 5 visualizations as separate, individual image files
rather than combined dashboards.
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add the experiments directory to the path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiments/phase5_analysis')))

from experiments.phase5_analysis.human_vs_llm_comparison import HumanVsLLMComparison, ComparisonConfig
from experiments.phase5_analysis.individual_visualizations import *

def generate_all_individual_visualizations(human_csv_path: str, llm_results_dir: str, 
                                         output_dir: str = "experiments/phase5_analysis/individual_visualizations"):
    """Generate all individual visualization charts"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🎨 GENERATING INDIVIDUAL VISUALIZATIONS")
    print("=" * 60)
    
    # Load and analyze data
    config = ComparisonConfig(generate_visualizations=False)
    comparison = HumanVsLLMComparison(config)
    comparison.load_human_data(human_csv_path)
    comparison.load_llm_data(llm_results_dir)
    results = comparison.perform_statistical_analysis()
    
    generated_files = []
    
    # 1. Model-Specific Individual Charts
    if 'model_specific_comparison' in results:
        model_data = results['model_specific_comparison']
        models = list(model_data.keys())
        
        print(f"***REMOVED***n📊 Generating Model Comparison Charts ({len(models)} models)...")
        
        # Accuracy vs Speed Scatter
        generate_accuracy_speed_scatter(model_data, models, output_path)
        generated_files.append("model_accuracy_speed_scatter.png")
        
        # Model Rankings
        generate_model_rankings(model_data, models, output_path)
        generated_files.extend(["model_accuracy_ranking.png", "model_speed_ranking.png"])

        # Model Time Ranking (new chart)
        plot_model_time_ranking(model_data, models, output_path / "model_time_ranking.png")
        generated_files.append("model_time_ranking.png")

        # Confidence Analysis Charts (use model_data from results)
        if 'model_data' in results:
            confidence_model_data = results['model_data']
            plot_model_confidence_ranking(confidence_model_data, models, output_path / "model_confidence_ranking.png")
            generated_files.append("model_confidence_ranking.png")

        # Statistical Significance
        generate_statistical_significance_chart(model_data, models, output_path)
        generated_files.append("statistical_significance_matrix.png")
        
        # Performance Radar
        generate_performance_radar(model_data, models, output_path)
        generated_files.append("performance_radar_chart.png")
        
        # Accuracy Improvements
        generate_accuracy_improvements(model_data, models, output_path)
        generated_files.append("accuracy_improvement_bars.png")
    
    # 2. Complexity Analysis Individual Charts
    if 'model_complexity_analysis' in results:
        complexity_data = results['model_complexity_analysis']
        models = list(complexity_data.keys())
        complexities = ['easy', 'medium', 'hard']
        
        print(f"***REMOVED***n🧩 Generating Complexity Analysis Charts...")
        
        # Complexity charts - generate them inline
        generate_complexity_accuracy_heatmap(complexity_data, models, complexities, output_path)
        generated_files.append("complexity_accuracy_heatmap.png")

        plot_complexity_performance_comparison(complexity_data, models, complexities,
                                             output_path / "complexity_performance_comparison.png")
        generated_files.append("complexity_performance_comparison.png")

        plot_complexity_improvement_heatmap(complexity_data, models, complexities,
                                          output_path / "complexity_improvement_heatmap.png")
        generated_files.append("complexity_improvement_heatmap.png")

        plot_complexity_speed_analysis(complexity_data, models, complexities,
                                     output_path / "complexity_speed_heatmap.png")
        generated_files.append("complexity_speed_heatmap.png")

        plot_best_models_by_complexity(complexity_data, models, complexities,
                                     output_path / "best_models_by_complexity.png")
        generated_files.append("best_models_by_complexity.png")

        # Confidence by Complexity
        plot_confidence_by_complexity(complexity_data, models, complexities,
                                    output_path / "confidence_by_complexity.png")
        generated_files.append("confidence_by_complexity.png")
    
    # 3. Quality Analysis Individual Charts
    if 'model_quality_analysis' in results:
        quality_data = results['model_quality_analysis']
        models = list(quality_data.keys())
        qualities = ['Q0', 'Q1', 'Q2', 'Q3']
        
        print(f"***REMOVED***n🛡️ Generating Quality Analysis Charts...")
        
        # Quality charts - generate them inline
        generate_quality_accuracy_heatmap(quality_data, models, qualities, output_path)
        generated_files.append("quality_accuracy_heatmap.png")

        generate_quality_robustness_ranking(quality_data, models, output_path)
        generated_files.append("quality_robustness_ranking.png")
        
        generate_quality_degradation_patterns(quality_data, models, qualities, output_path)
        generated_files.append("quality_degradation_patterns.png")
        
        plot_quality_comparison_bars(quality_data, models, qualities, output_path / "quality_comparison_bars.png")
        generated_files.append("quality_comparison_bars.png")

        # Confidence by Quality
        plot_confidence_by_quality(quality_data, models, qualities,
                                 output_path / "confidence_by_quality.png")
        generated_files.append("confidence_by_quality.png")
    
    # 4. Overall Comparison Charts (individual versions)
    print(f"***REMOVED***n📈 Generating Overall Comparison Charts...")
    
    generate_overall_individual_charts(results, output_path)
    generated_files.extend([
        "overall_accuracy_comparison.png",
        "overall_speed_comparison.png", 
        "overall_cost_comparison.png"
    ])
    
    print(f"***REMOVED***n✅ VISUALIZATION GENERATION COMPLETE")
    print("=" * 60)
    print(f"📁 Output directory: {output_path}")
    print(f"📊 Generated {len(generated_files)} individual charts:")
    
    for i, filename in enumerate(generated_files, 1):
        print(f"  {i:2d}. {filename}")
    
    print(f"***REMOVED***n💡 To view all charts:")
    print(f"   open {output_path}")
    
    return generated_files

def generate_accuracy_speed_scatter(model_data, models, output_path):
    """Generate accuracy vs speed scatter plot"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    accuracies = [model_data[model]['model_accuracy'] for model in models]
    speeds = [model_data[model]['time_speedup_factor'] for model in models]
    human_acc = model_data[models[0]]['human_accuracy']
    
    # Create bubble sizes based on statistical significance
    bubble_sizes = []
    colors = []
    for model in models:
        acc_sig = model_data[model]['statistical_tests']['accuracy_chi2']['significant']
        time_sig = model_data[model]['statistical_tests']['time_mannwhitney']['significant'] if model_data[model]['statistical_tests']['time_mannwhitney'] else False
        
        if acc_sig and time_sig:
            bubble_sizes.append(300)
            colors.append('#2E8B57')  # Dark green
        elif acc_sig or time_sig:
            bubble_sizes.append(200)
            colors.append('#FFD700')  # Gold
        else:
            bubble_sizes.append(100)
            colors.append('#FF6347')  # Red
    
    scatter = ax.scatter(speeds, accuracies, s=bubble_sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model.replace('-', '***REMOVED***n'), (speeds[i], accuracies[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Add human baseline
    ax.axhline(y=human_acc, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Human Baseline ({human_acc:.3f})')
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='No Speed Improvement')
    
    ax.set_xlabel('Speed Improvement Factor (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: Accuracy vs Speed***REMOVED***n(Bubble size indicates statistical significance)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add legend for bubble sizes
    legend_elements = [
        plt.scatter([], [], s=300, c='#2E8B57', alpha=0.7, edgecolors='black', label='Both Acc & Time Significant'),
        plt.scatter([], [], s=200, c='#FFD700', alpha=0.7, edgecolors='black', label='One Significant'),
        plt.scatter([], [], s=100, c='#FF6347', alpha=0.7, edgecolors='black', label='Neither Significant')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    plt.tight_layout()
    plt.savefig(output_path / "model_accuracy_speed_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_model_rankings(model_data, models, output_path):
    """Generate model ranking charts"""
    human_acc = model_data[models[0]]['human_accuracy']
    
    # Accuracy ranking
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    model_acc_sorted = sorted([(model, model_data[model]['model_accuracy']) for model in models], 
                             key=lambda x: x[1], reverse=True)
    
    model_names = [item[0].replace('-', '***REMOVED***n') for item in model_acc_sorted]
    model_accs = [item[1] for item in model_acc_sorted]
    
    bar_colors = ['green' if acc > human_acc else 'red' for acc in model_accs]
    bars = ax.barh(range(len(model_names)), model_accs, color=bar_colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=human_acc, color='red', linestyle='--', linewidth=3, label=f'Human Baseline ({human_acc:.3f})')
    
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=11)
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Ranking vs Human Baseline', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, acc) in enumerate(zip(bars, model_accs)):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.3f}', 
               va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "model_accuracy_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Speed ranking
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    model_speed_sorted = sorted([(model, model_data[model]['time_speedup_factor']) for model in models], 
                               key=lambda x: x[1], reverse=True)
    
    speed_names = [item[0].replace('-', '***REMOVED***n') for item in model_speed_sorted]
    speed_values = [min(item[1], 25) if item[1] != float('inf') else 25 for item in model_speed_sorted]
    
    bars = ax.barh(range(len(speed_names)), speed_values, color='lightblue', alpha=0.7, edgecolor='navy')
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=3, label='No improvement')
    
    ax.set_yticks(range(len(speed_names)))
    ax.set_yticklabels(speed_names, fontsize=11)
    ax.set_xlabel('Speed Improvement Factor', fontsize=12, fontweight='bold')
    ax.set_title('Model Speed Improvement Ranking', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, speed) in enumerate(zip(bars, speed_values)):
        ax.text(speed + 0.5, bar.get_y() + bar.get_height()/2, f'{speed:.1f}x', 
               va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "model_speed_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_statistical_significance_chart(model_data, models, output_path):
    """Generate statistical significance matrix"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    sig_matrix = []
    test_names = ['Accuracy***REMOVED***nSignificant', 'Time***REMOVED***nSignificant']

    for model in models:
        model_stats = model_data[model]['statistical_tests']
        row = []
        acc_sig = model_stats['accuracy_chi2']['significant'] if model_stats['accuracy_chi2'] else False
        time_sig = model_stats['time_mannwhitney']['significant'] if model_stats['time_mannwhitney'] else False
        row.extend([1 if acc_sig else 0, 1 if time_sig else 0])
        sig_matrix.append(row)

    sig_array = np.array(sig_matrix)
    im = ax.imshow(sig_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(test_names)))
    ax.set_xticklabels(test_names, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=11)
    ax.set_title('Statistical Significance Matrix***REMOVED***n(p < 0.05)', fontsize=14, fontweight='bold')

    for i in range(len(models)):
        for j in range(len(test_names)):
            text = '✓' if sig_array[i, j] == 1 else '✗'
            color = 'white' if sig_array[i, j] == 1 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=16, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path / "statistical_significance_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_radar(model_data, models, output_path):
    """Generate performance radar chart"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))

    top_3_models = sorted(models, key=lambda x: model_data[x]['model_accuracy'], reverse=True)[:3]

    categories = ['Accuracy***REMOVED***nImprovement', 'Speed***REMOVED***nImprovement', 'Cost***REMOVED***nEfficiency']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, model in enumerate(top_3_models):
        values = [
            max(0, model_data[model]['accuracy_difference']),
            min(model_data[model]['time_speedup_factor'] / 25, 1),
            min(model_data[model]['cost_efficiency_ratio'] / 100, 1) if model_data[model]['cost_efficiency_ratio'] != float('inf') else 1
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=3, label=model.replace('-', '***REMOVED***n'), color=colors_radar[i], markersize=8)
        ax.fill(angles, values, alpha=0.25, color=colors_radar[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Top 3 Models Performance Comparison***REMOVED***n(Normalized Metrics)', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "performance_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_accuracy_improvements(model_data, models, output_path):
    """Generate accuracy improvement bars"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    improvements = [(model, model_data[model]['accuracy_difference']) for model in models]
    improvements.sort(key=lambda x: x[1], reverse=True)

    model_names = [item[0].replace('-', '***REMOVED***n') for item in improvements]
    improvement_values = [item[1] for item in improvements]

    colors = ['green' if imp > 0.1 else 'orange' if imp > 0 else 'red' for imp in improvement_values]

    bars = ax.bar(range(len(model_names)), improvement_values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=11, rotation=45, ha='right')
    ax.set_ylabel('Accuracy Difference from Human Baseline', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Improvement over Human Baseline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, imp) in enumerate(zip(bars, improvement_values)):
        y_pos = imp + (0.01 if imp >= 0 else -0.01)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{imp:+.3f}',
               ha='center', va='bottom' if imp >= 0 else 'top', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / "accuracy_improvement_bars.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_quality_degradation_patterns(quality_data, models, qualities, output_path):
    """Generate quality degradation patterns chart"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    quality_labels = ['Normal Baseline***REMOVED***n(Q0)', 'Spaces***REMOVED***n(Q1)', 'Missing Chars***REMOVED***n(Q2)', 'Missing Records***REMOVED***n(Q3)']

    for i, model in enumerate(models):
        model_accs = []
        for quality in qualities:
            if quality in quality_data[model]:
                model_accs.append(quality_data[model][quality]['model_accuracy'])
            else:
                model_accs.append(0)

        ax.plot(range(len(qualities)), model_accs, marker='o', linewidth=2,
               label=model.replace('-', '***REMOVED***n'), alpha=0.8, markersize=8)

    ax.set_title('Model Performance Degradation by Data Quality', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Data Quality Condition', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(qualities)))
    ax.set_xticklabels(quality_labels, fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path / "quality_degradation_patterns.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_quality_comparison_bars(quality_data, models, qualities, output_path):
    """Generate quality comparison bar chart"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Get human baseline for each quality
    human_accs = []
    for quality in qualities:
        if models and quality in quality_data[models[0]]:
            human_accs.append(quality_data[models[0]][quality]['human_accuracy'])
        else:
            human_accs.append(0)

    x = np.arange(len(qualities))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_accs = []
        for quality in qualities:
            if quality in quality_data[model]:
                model_accs.append(quality_data[model][quality]['model_accuracy'])
            else:
                model_accs.append(0)

        ax.bar(x + i * width, model_accs, width, label=model.replace('-', '***REMOVED***n'), alpha=0.8)

    # Add human baseline
    ax.plot(x + width * (len(models)-1)/2, human_accs, 'ro-', linewidth=3, markersize=8, label='Human Baseline')

    ax.set_title('Model vs Human Performance by Data Quality', fontsize=14, fontweight='bold')
    ax.set_xlabel('Data Quality Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * (len(models)-1)/2)
    ax.set_xticklabels(['Normal***REMOVED***nBaseline', 'Spaces', 'Missing***REMOVED***nChars', 'Missing***REMOVED***nRecords'])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "quality_comparison_bars.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_overall_individual_charts(results, output_path):
    """Generate individual overall comparison charts"""
    overall = results['overall_comparison']

    # Overall accuracy comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    accuracy_data = [overall['human_accuracy'], overall['llm_accuracy']]
    accuracy_labels = ['Human', 'LLM (Avg)']
    colors = ['#FF6B6B', '#4ECDC4']

    bars = ax.bar(accuracy_labels, accuracy_data, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy Rate', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)

    for i, (bar, v) in enumerate(zip(bars, accuracy_data)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}',
               ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / "overall_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Overall speed comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    time_data = [overall['human_avg_time_sec'], overall['llm_avg_time_sec']]
    time_labels = ['Human', 'LLM (Avg)']

    bars = ax.bar(time_labels, time_data, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Average Completion Time Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')

    for i, (bar, v) in enumerate(zip(bars, time_data)):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(time_data) * 0.02, f'{v:.1f}s',
               ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / "overall_speed_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Overall cost comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cost_data = [overall['human_avg_cost_usd'], overall['llm_avg_cost_usd']]
    cost_labels = ['Human', 'LLM (Avg)']

    bars = ax.bar(cost_labels, cost_data, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Average Cost per Task Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cost (USD)', fontsize=12, fontweight='bold')

    for i, (bar, v) in enumerate(zip(bars, cost_data)):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(cost_data) * 0.02, f'${v:.3f}',
               ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / "overall_cost_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_complexity_accuracy_heatmap(complexity_data, models, complexities, output_path):
    """Generate complexity accuracy heatmap"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    accuracy_matrix = []
    for model in models:
        row = []
        for complexity in complexities:
            if complexity in complexity_data[model]:
                row.append(complexity_data[model][complexity]['model_accuracy'])
            else:
                row.append(0)
        accuracy_matrix.append(row)

    accuracy_array = np.array(accuracy_matrix)
    im = ax.imshow(accuracy_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Model Accuracy by Task Complexity', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(complexities)))
    ax.set_xticklabels([c.capitalize() for c in complexities], fontsize=12)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace('-', '***REMOVED***n') for m in models], fontsize=11)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(complexities)):
            if accuracy_array[i, j] > 0:
                text = f'{accuracy_array[i, j]:.3f}'
                color = 'white' if accuracy_array[i, j] < 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path / "complexity_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_quality_accuracy_heatmap(quality_data, models, qualities, output_path):
    """Generate quality accuracy heatmap"""
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
    plt.savefig(output_path / "quality_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_quality_robustness_ranking(quality_data, models, output_path):
    """Generate quality robustness ranking chart"""
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
        ax.set_xlabel('Robustness Score (Corrupted/Normal Baseline)', fontsize=12, fontweight='bold')
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
    plt.savefig(output_path / "quality_robustness_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate individual Phase 5 visualizations')
    parser.add_argument('--human-data', 
                       default='experiments/human_study/Participants_results.csv',
                       help='Path to human study results CSV file')
    parser.add_argument('--llm-data', 
                       default='experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark',
                       help='Path to directory containing LLM evaluation results')
    parser.add_argument('--output-dir', 
                       default='experiments/phase5_analysis/individual_visualizations',
                       help='Output directory for individual visualizations')
    
    args = parser.parse_args()
    
    # Validate input paths
    human_path = Path(args.human_data)
    llm_path = Path(args.llm_data)
    
    if not human_path.exists():
        print(f"Error: Human data file not found: {human_path}")
        sys.exit(1)
    
    if not llm_path.exists():
        print(f"Error: LLM data directory not found: {llm_path}")
        sys.exit(1)
    
    # Generate all individual visualizations
    generated_files = generate_all_individual_visualizations(
        str(human_path), str(llm_path), args.output_dir
    )
    
    print(f"***REMOVED***n🎉 Successfully generated {len(generated_files)} individual visualization files!")

if __name__ == "__main__":
    main()
