# Phase 5: Individual Visualizations

This directory contains **18 individual visualization charts** generated from the Phase 5 human vs LLM comparative analysis. Each chart is a separate, high-quality image file that can be used independently for presentations, reports, or publications.

## 📊 Model Comparison Charts (6 files)

### 1. `model_accuracy_speed_scatter.png`
**Accuracy vs Speed Performance Scatter Plot**
- Shows each model's position in accuracy-speed space
- Bubble size indicates statistical significance
- Human baseline and speed improvement thresholds marked
- **Use for**: Identifying optimal models balancing accuracy and speed

### 2. `model_accuracy_ranking.png`
**Model Accuracy Ranking Bar Chart**
- Horizontal bar chart ranking models by accuracy
- Color-coded: green (better than human), red (worse than human)
- Human baseline clearly marked
- **Use for**: Executive summaries, model selection decisions

### 3. `model_speed_ranking.png`
**Model Speed Improvement Ranking**
- Horizontal bar chart showing speed improvement factors
- All models compared to human baseline (1x = no improvement)
- **Use for**: Performance optimization discussions

### 4. `statistical_significance_matrix.png`
**Statistical Significance Validation Matrix**
- Heatmap showing which models have statistically significant improvements
- ✓ = significant (p < 0.05), ✗ = not significant
- Covers both accuracy and time improvements
- **Use for**: Research validation, methodology sections

### 5. `performance_radar_chart.png`
**Top 3 Models Performance Radar**
- Multi-dimensional comparison of top performers
- Normalized metrics: accuracy improvement, speed, cost efficiency
- **Use for**: Comprehensive model comparison presentations

### 6. `accuracy_improvement_bars.png`
**Accuracy Improvement over Human Baseline**
- Bar chart showing each model's accuracy difference from humans
- Color-coded: green (excellent), orange (good), red (poor)
- **Use for**: Highlighting model advantages/disadvantages

## 🧩 Task Complexity Analysis Charts (5 files)

### 7. `complexity_accuracy_heatmap.png`
**Model Accuracy by Task Complexity Heatmap**
- 2D heatmap: models vs complexity levels (easy/medium/hard)
- Color intensity shows accuracy levels
- Numerical values overlaid for precision
- **Use for**: Understanding model strengths across difficulty levels

### 8. `complexity_performance_comparison.png`
**Human vs Model Performance by Complexity**
- Grouped bar chart comparing all models to human baseline
- Separate bars for each complexity level
- **Use for**: Demonstrating complexity-specific advantages

### 9. `complexity_improvement_heatmap.png`
**Accuracy Improvement over Humans by Complexity**
- Heatmap showing improvement/degradation patterns
- Red/blue color scheme: red = worse, blue = better
- **Use for**: Identifying where models excel or struggle

### 10. `complexity_speed_heatmap.png`
**Speed Improvement by Task Complexity**
- Heatmap showing speed factors across complexity levels
- Helps identify if models maintain speed advantages on harder tasks
- **Use for**: Performance scaling analysis

### 11. `best_models_by_complexity.png`
**Best Performing Model by Complexity Level**
- Bar chart showing which model performs best for each complexity
- Model names and accuracy values clearly labeled
- **Use for**: Task-specific model recommendations

## 🛡️ Data Quality Analysis Charts (4 files)

### 12. `quality_accuracy_heatmap.png`
**Model Accuracy by Data Quality Condition**
- Heatmap across Q0 (normal baseline) to Q3 (corrupted) data
- Shows robustness to data quality degradation
- **Use for**: Evaluating model reliability in real-world conditions

### 13. `quality_robustness_ranking.png`
**Data Quality Robustness Ranking**
- Horizontal bar chart ranking models by robustness score
- Robustness = (corrupted data performance) / (normal baseline data performance)
- Color-coded thresholds: green (>0.8), orange (0.6-0.8), red (<0.6)
- **Use for**: Selecting models for poor data quality environments

### 14. `quality_degradation_patterns.png`
**Performance Degradation Patterns by Data Quality**
- Line chart showing how each model's accuracy changes with data quality
- Multiple lines for different models
- **Use for**: Understanding degradation patterns and model resilience

### 15. `quality_comparison_bars.png`
**Human vs Model Performance by Data Quality**
- Grouped bar chart across all quality conditions
- Human baseline line for reference
- **Use for**: Comprehensive quality condition analysis

## 📈 Overall Comparison Charts (3 files)

### 16. `overall_accuracy_comparison.png`
**Overall Accuracy Comparison**
- Simple bar chart: Human vs LLM average
- Clean, executive-friendly format
- **Use for**: High-level presentations, executive summaries

### 17. `overall_speed_comparison.png`
**Overall Speed Comparison**
- Bar chart showing average completion times
- Human vs LLM average with exact values
- **Use for**: Efficiency demonstrations

### 18. `overall_cost_comparison.png`
**Overall Cost Comparison**
- Cost per task comparison
- Human labor cost vs LLM API cost
- **Use for**: Economic justification, ROI discussions

## 🎯 Usage Recommendations

### For Executive Presentations
- `model_accuracy_ranking.png` - Clear model performance ranking
- `overall_accuracy_comparison.png` - High-level accuracy summary
- `overall_speed_comparison.png` - Efficiency demonstration
- `overall_cost_comparison.png` - Economic benefits

### For Technical Analysis
- `model_accuracy_speed_scatter.png` - Performance trade-offs
- `statistical_significance_matrix.png` - Validation of results
- `complexity_accuracy_heatmap.png` - Detailed complexity analysis
- `quality_robustness_ranking.png` - Reliability assessment

### For Research Publications
- `statistical_significance_matrix.png` - Methodology validation
- `complexity_improvement_heatmap.png` - Detailed performance analysis
- `quality_degradation_patterns.png` - Robustness evaluation
- `performance_radar_chart.png` - Multi-dimensional comparison

### For Specific Use Cases
- **Real-time applications**: `model_speed_ranking.png`
- **High-accuracy requirements**: `accuracy_improvement_bars.png`
- **Poor data quality**: `quality_robustness_ranking.png`
- **Task-specific deployment**: `best_models_by_complexity.png`

## 🔧 Technical Specifications

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG with transparency support
- **Size**: Optimized for both digital and print use
- **Color scheme**: Colorblind-friendly palettes
- **Fonts**: Clear, readable typography

## 📁 File Organization

All charts are saved as individual PNG files in this directory. Each file is self-contained and can be used independently. File names are descriptive and follow a consistent naming convention:

- `model_*` - Model-specific comparisons
- `complexity_*` - Task complexity analysis
- `quality_*` - Data quality analysis  
- `overall_*` - High-level comparisons

## 🚀 Generation

To regenerate these visualizations:

```bash
# Generate all individual charts
python run_phase5_individual_visualizations.py

# With custom paths
python run_phase5_individual_visualizations.py ***REMOVED***
  --human-data path/to/human_data.csv ***REMOVED***
  --llm-data path/to/llm_results/ ***REMOVED***
  --output-dir custom/output/directory
```

## 💡 Tips for Use

1. **Presentations**: Use high-contrast charts like rankings and comparisons
2. **Reports**: Include statistical validation charts for credibility
3. **Decision Making**: Focus on trade-off charts (accuracy vs speed)
4. **Publications**: Use heatmaps and detailed analysis charts
5. **Executive Briefings**: Stick to overall comparison charts

Each visualization tells a specific part of the human vs LLM performance story. Combine multiple charts to build comprehensive narratives for your specific audience and use case.
