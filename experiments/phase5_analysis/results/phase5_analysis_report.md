# Phase 5: Human vs LLM Comparative Analysis Report
**Generated:** 2025-06-26 10:50:45

## Executive Summary

- **Human Accuracy:** 0.633 (63.3%)
- **LLM Accuracy:** 0.802 (80.2%)
- **Accuracy Difference:** +0.169 (+16.9%)

- **Human Avg Time:** 166.2 seconds
- **LLM Avg Time:** 51.7 seconds
- **Speed Improvement:** 3.2x faster

- **Human Avg Cost:** $1.154 per task
- **LLM Avg Cost:** $0.000 per task
- **Cost Efficiency:** infx more cost-effective

## Statistical Analysis

### Accuracy Comparison
- **Chi-square test:** χ² = 13.7411, p = 0.0002
- **Result:** Significant difference in accuracy

### Completion Time Comparison
- **Mann-Whitney U test:** U = 7468.0000, p = 0.0000
- **Result:** Significant difference in completion time

## Performance by Task Complexity

| Complexity | Human Accuracy | LLM Accuracy | Human Time (s) | LLM Time (s) |
|------------|----------------|--------------|----------------|--------------|
| Easy | 0.433 | 0.809 | 185.6 | 54.8 |
| Medium | 0.667 | 0.793 | 179.5 | 52.8 |
| Hard | 0.800 | 0.804 | 133.6 | 47.4 |

## Performance by Data Quality Condition

| Quality | Human Accuracy | LLM Accuracy | Human Time (s) | LLM Time (s) |
|---------|----------------|--------------|----------------|--------------|
| Q0 | 0.667 | 0.868 | 154.4 | 47.1 |
| Q1 | 0.733 | 0.800 | 176.4 | 56.9 |
| Q2 | 0.533 | 0.600 | 169.7 | 58.6 |
| Q3 | 0.533 | 0.807 | 187.9 | 53.0 |

## Model-Specific Performance vs Human Baseline

| Model | Accuracy | Accuracy Diff | Speed Factor | Cost Ratio | Acc. Significant | Time Significant |
|-------|----------|---------------|--------------|------------|------------------|------------------|
| claude-sonnet-4-20250514 | 0.941 | +0.307 | 11.8x | infx | ✓ | ✓ |
| gpt-4o-mini-2024-07-18 | 0.507 | -0.126 | 1.9x | infx | ✗ | ✓ |
| deepseek-reasoner | 0.974 | +0.341 | 1.0x | infx | ✓ | ✗ |
| o4-mini-2025-04-16 | 0.685 | +0.052 | 16.3x | infx | ✗ | ✓ |
| claude-3-5-haiku-latest | 0.930 | +0.296 | 21.2x | infx | ✓ | ✓ |
| deepseek-chat | 0.774 | +0.141 | 5.5x | infx | ✓ | ✓ |

*✓ = Statistically significant difference (p < 0.05), ✗ = Not significant*

## Model Performance by Task Complexity

### Best Models by Complexity Level

- **Easy**: deepseek-reasoner (0.933 accuracy, +0.500 vs human)
- **Medium**: claude-sonnet-4-20250514 (0.989 accuracy, +0.322 vs human)
- **Hard**: deepseek-reasoner (1.000 accuracy, +0.200 vs human)

## Model Performance by Data Quality Condition

### Data Quality Robustness Analysis

| Model | Robustness Score | Perfect Data (Q0) | Corrupted Avg (Q1-Q3) |
|-------|------------------|-------------------|------------------------|
| deepseek-reasoner 🟢 | 0.948 | 1.000 | 0.948 |
| claude-sonnet-4-20250514 🟢 | 0.910 | 0.985 | 0.896 |
| claude-3-5-haiku-latest 🟢 | 0.859 | 1.000 | 0.859 |
| gpt-4o-mini-2024-07-18 🟢 | 0.851 | 0.548 | 0.467 |
| o4-mini-2025-04-16 🟡 | 0.745 | 0.785 | 0.585 |
| deepseek-chat 🟡 | 0.742 | 0.889 | 0.659 |

*🟢 = Robust (>0.8), 🟡 = Moderate (0.6-0.8), 🔴 = Fragile (<0.6)*

## Cost-Effectiveness Analysis

## Data Summary

- **Total Common Tasks:** 90
- **Human Records:** 90
- **LLM Records:** 1620
- **LLM Models:** claude-3-5-haiku-latest, claude-sonnet-4-20250514, deepseek-chat, deepseek-reasoner, gpt-4o-mini-2024-07-18, o4-mini-2025-04-16

## Generated Files

### CSV Exports
- **Overall Comparison:** `experiments/phase5_analysis/results/overall_comparison.csv`
- **Statistical Tests:** `experiments/phase5_analysis/results/statistical_tests.csv`
- **Complexity Comparison:** `experiments/phase5_analysis/results/complexity_comparison.csv`
- **Model Specific Comparison:** `experiments/phase5_analysis/results/model_specific_comparison.csv`
- **Quality Comparison:** `experiments/phase5_analysis/results/quality_comparison.csv`
- **Model Complexity Analysis:** `experiments/phase5_analysis/results/model_complexity_analysis.csv`
- **Model Quality Analysis:** `experiments/phase5_analysis/results/model_quality_analysis.csv`
- **Task Level Comparison:** `experiments/phase5_analysis/results/task_level_comparison.csv`
- **Analysis Summary:** `experiments/phase5_analysis/results/analysis_summary.csv`

### Visualizations
- **Overall Performance:** `experiments/phase5_analysis/visualizations/overall_performance_comparison.png`
- **Performance By Complexity:** `experiments/phase5_analysis/visualizations/performance_by_complexity.png`
- **Performance By Quality:** `experiments/phase5_analysis/visualizations/performance_by_quality.png`
- **Time Vs Accuracy:** `experiments/phase5_analysis/visualizations/time_vs_accuracy_scatter.png`
- **Cost Analysis:** `experiments/phase5_analysis/visualizations/cost_analysis.png`
- **Model Specific Comparison:** `experiments/phase5_analysis/visualizations/model_specific_comparison.png`
- **Model Complexity Analysis:** `experiments/phase5_analysis/visualizations/model_complexity_analysis.png`
- **Model Quality Analysis:** `experiments/phase5_analysis/visualizations/model_quality_analysis.png`
