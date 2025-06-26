# Phase 5: Human vs LLM Comparative Analysis

This module provides comprehensive comparison capabilities between human participants and LLM models on manufacturing data analysis tasks.

## Features

- **Statistical Analysis**: Chi-square tests for accuracy, Mann-Whitney U tests for completion time
- **Cost-Effectiveness Analysis**: Detailed ROI calculations with multiple cost scenarios
- **Visualization Dashboard**: Comprehensive charts and plots for performance comparison
- **CSV Export**: All analysis results exported in structured CSV format
- **Comprehensive Reporting**: Markdown reports with key findings and recommendations

## Quick Start

### Basic Usage

```bash
# Run complete Phase 5 analysis with default settings
python run_phase5_analysis.py

# Specify custom data paths
python run_phase5_analysis.py ***REMOVED***
  --human-data experiments/human_study/Participants_results.csv ***REMOVED***
  --llm-data experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark

# Customize human hourly rate for cost analysis
python run_phase5_analysis.py --human-hourly-rate 35.0

# Skip certain analysis components
python run_phase5_analysis.py --no-viz --no-cost-analysis
```

### Advanced Usage

```python
from experiments.phase5_analysis.human_vs_llm_comparison import HumanVsLLMComparison, ComparisonConfig
from experiments.phase5_analysis.cost_analysis import CostAnalyzer

# Configure analysis
config = ComparisonConfig(
    human_hourly_rate=30.0,
    output_dir="custom_results",
    visualization_dir="custom_viz"
)

# Run comparison
comparison = HumanVsLLMComparison(config)
results = comparison.run_complete_analysis(
    "path/to/human_data.csv",
    "path/to/llm_results_dir"
)

# Detailed cost analysis
cost_analyzer = CostAnalyzer()
cost_report = cost_analyzer.generate_cost_comparison_report(
    human_performance, llm_performance
)
```

## Output Files

### CSV Exports
- `overall_comparison.csv` - High-level performance metrics
- `statistical_tests.csv` - Statistical significance test results
- `complexity_comparison.csv` - Performance breakdown by task complexity
- `quality_comparison.csv` - Performance breakdown by data quality condition
- `task_level_comparison.csv` - Detailed task-by-task comparison
- `analysis_summary.csv` - Metadata and summary statistics

### Visualizations
- `enhanced_model_comparison_dashboard.png` - Comprehensive model comparison dashboard
- `complexity_performance_matrix.png` - Detailed task complexity analysis matrix
- `quality_robustness_analysis.png` - Data quality robustness analysis
- `model_specific_comparison.png` - Individual model performance comparison
- `model_complexity_analysis.png` - Model performance by complexity breakdown
- `model_quality_analysis.png` - Model performance by quality breakdown
- `overall_performance_comparison.png` - Overall performance metrics
- `performance_by_complexity.png` - Performance across complexity levels
- `performance_by_quality.png` - Performance across data quality conditions
- `time_vs_accuracy_scatter.png` - Time vs accuracy trade-off analysis
- `cost_analysis.png` - Cost-effectiveness comparison
- `detailed_cost_analysis.png` - Advanced cost scenario analysis

### Reports
- `phase5_analysis_report.md` - Comprehensive markdown report
- `phase5_results.json` - Complete results in JSON format

## Configuration Options

### ComparisonConfig
- `human_hourly_rate`: Hourly rate for human labor cost calculation (default: $25.00)
- `output_dir`: Directory for CSV exports and reports
- `visualization_dir`: Directory for generated plots
- `export_csv`: Enable/disable CSV export
- `generate_visualizations`: Enable/disable visualization generation

### Cost Analysis Scenarios
The cost analyzer includes multiple predefined scenarios:
- Manufacturing Technician ($25/hr)
- Quality Assurance Specialist ($35/hr)
- Data Analyst ($45/hr)
- Senior Engineer ($65/hr)

Each scenario includes overhead multipliers, training costs, and setup costs.

## Statistical Methods

### Accuracy Comparison
- **Chi-square test**: Tests for significant differences in accuracy rates between human and LLM performance
- **Effect size**: Calculates practical significance of accuracy differences

### Time Comparison
- **Mann-Whitney U test**: Non-parametric test for completion time differences
- **Speedup factor**: Ratio of human to LLM completion times

### Cost Analysis
- **ROI calculation**: Return on investment considering accuracy and error costs
- **Break-even analysis**: Number of tasks needed to justify LLM implementation
- **Total cost of ownership**: Includes setup, training, and operational costs

## Requirements

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
```

## Data Format Requirements

### Human Data CSV
Required columns:
- `task_id`: Unique task identifier
- `completion_time_sec`: Task completion time in seconds
- `accuracy` or `is_correct`: Binary accuracy (0/1)
- `complexity`: Task complexity level (easy/medium/hard)
- `quality_condition` or `data quality`: Data quality condition (Q0/Q1/Q2/Q3)

### LLM Data CSV
Required columns:
- `task_id`: Unique task identifier matching human data
- `completion_time_sec`: Task completion time in seconds
- `is_correct`: Binary accuracy (True/False or 1/0)
- `complexity`: Task complexity level
- `quality_condition`: Data quality condition
- `total_cost_usd`: API cost per task
- `model`: LLM model identifier

## Troubleshooting

### Common Issues

1. **No common task_ids found**: Ensure task_id formats match between human and LLM datasets
2. **Encoding errors**: Human CSV may have BOM encoding - the system handles this automatically
3. **Missing columns**: Check that required columns exist and are properly named
4. **Empty results**: Verify that both datasets contain data for the same tasks

### Debug Mode

Add debug prints by modifying the comparison class:

```python
comparison = HumanVsLLMComparison(config)
comparison.load_human_data(human_path)
print(f"Human columns: {comparison.human_data.columns.tolist()}")
print(f"Human task_ids: {sorted(comparison.human_data['task_id'].unique())}")
```
