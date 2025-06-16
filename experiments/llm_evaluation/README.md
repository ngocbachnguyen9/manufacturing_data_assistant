# LLM Evaluation and Benchmarking System

A comprehensive evaluation and benchmarking framework for Large Language Models (LLMs) on manufacturing data analysis tasks.

## 🎯 Overview

This system provides a complete evaluation framework to assess and benchmark LLM performance across different:
- **Models** (deepseek-chat, GPT-4, Claude, etc.)
- **Task Complexities** (easy, medium, hard)
- **Data Quality Conditions** (Q0-Q3: perfect to corrupted data)
- **Prompt Lengths** (short, normal, long)

## 🏗️ System Architecture

### Core Components

1. **Evaluation Framework** (`evaluation_framework.py`)
   - Performance metrics calculation
   - Benchmark scoring and ranking
   - Comprehensive reporting

2. **Model Comparison** (`model_comparison.py`)
   - Statistical significance testing
   - Multi-dimensional performance analysis
   - Visualization generation

3. **Prompt Effectiveness Analyzer** (`prompt_effectiveness_analyzer.py`)
   - Prompt length comparison (short/normal/long)
   - Token efficiency analysis
   - Time-accuracy trade-offs

4. **Comprehensive Benchmarking Suite** (`benchmarking_suite.py`)
   - Unified evaluation pipeline
   - Manufacturing-specific metrics
   - Executive summaries and recommendations

## 📊 Key Features

### Performance Metrics
- **Accuracy**: Task completion correctness
- **Speed**: Average completion time
- **Cost**: Token usage and API costs
- **Confidence**: Model confidence calibration
- **Error Detection**: Data quality issue identification
- **Token Efficiency**: Correct answers per token

### Manufacturing-Specific Analysis
- **Gear Identification Tasks**: Packing list to gear mapping
- **Printer Analysis Tasks**: Part-to-printer tracing
- **Compliance Verification**: Document date validation
- **Cross-System Integration**: Multi-source data handling

### Data Quality Conditions
- **Q0 (Baseline)**: Perfect data, 0% corruption
- **Q1 (Space Injection)**: Random spaces in barcodes (~15%)
- **Q2 (Character Missing)**: Missing characters in IDs (~12%)
- **Q3 (Missing Records)**: Strategic record removal (~7%)

## 🚀 Quick Start

### 1. Run the Demo
```bash
cd experiments/llm_evaluation
python run_evaluation_demo.py
```

### 2. Individual Component Usage

#### Basic Performance Evaluation
```python
from evaluation_framework import LLMEvaluationFramework

framework = LLMEvaluationFramework()
report = framework.generate_performance_report()
benchmark_results = framework.create_benchmark_suite()
```

#### Model Comparison
```python
from model_comparison import ModelComparison

comparison = ModelComparison()
stats_results = comparison.compare_models_statistical()
comparison.create_comparison_visualizations()
```

#### Prompt Effectiveness Analysis
```python
from prompt_effectiveness_analyzer import PromptEffectivenessAnalyzer

analyzer = PromptEffectivenessAnalyzer()
effectiveness = analyzer.analyze_prompt_effectiveness()
analyzer.create_prompt_comparison_visualizations()
```

#### Complete Benchmarking Suite
```python
from benchmarking_suite import ComprehensiveBenchmarkingSuite

suite = ComprehensiveBenchmarkingSuite()
results, json_path, summary_path = suite.run_and_export_complete_suite()
```

## 📁 File Structure

```
experiments/llm_evaluation/
├── evaluation_framework.py          # Core evaluation metrics
├── model_comparison.py              # Statistical model comparison
├── prompt_effectiveness_analyzer.py # Prompt length analysis
├── benchmarking_suite.py           # Unified benchmarking system
├── run_evaluation_demo.py          # Demo runner script
├── README.md                       # This file
├── performance_logs/
│   └── llm_performance_results.csv # Performance data
└── [Generated Output Directories]
    ├── visualizations/             # Comparison charts
    ├── prompt_analysis/           # Prompt effectiveness plots
    └── benchmark_suite_results/   # Complete benchmark outputs
```

## 📈 Generated Outputs

### Reports
- **Performance Report**: Comprehensive metrics analysis
- **Model Comparison Report**: Statistical comparison results
- **Prompt Effectiveness Report**: Prompt length analysis
- **Executive Summary**: High-level findings and recommendations

### Visualizations
- **Overall Performance Comparison**: Accuracy, time, cost, confidence
- **Performance by Complexity**: Easy/medium/hard task breakdown
- **Performance by Quality**: Q0-Q3 data condition analysis
- **Time vs Accuracy Scatter**: Trade-off analysis
- **Cost Analysis**: Cost efficiency and optimization
- **Error Detection Capabilities**: Data quality issue identification
- **Token Efficiency Analysis**: Input/output token usage
- **Prompt Comparison Charts**: Short/normal/long effectiveness

### Data Exports
- **Benchmark Results JSON**: Complete numerical results
- **Statistical Analysis**: Significance testing results
- **Manufacturing Metrics**: Domain-specific performance data

## 🎯 Task Prompt Variations

The system supports three prompt lengths for each task complexity:

### Short Prompts
- Minimal, concise instructions
- Fastest processing time
- Lower token usage
- Suitable for high-volume tasks

### Normal Prompts  
- Balanced detail and clarity
- Optimal accuracy-efficiency trade-off
- Recommended for most use cases

### Long Prompts
- Comprehensive, detailed instructions
- Highest accuracy potential
- Best for complex, critical tasks
- Current baseline format

## 📊 Benchmarking Methodology

### Scoring System
- **Accuracy Weight**: 40%
- **Speed Weight**: 20%
- **Cost Weight**: 15%
- **Confidence Weight**: 15%
- **Error Detection Weight**: 10%

### Statistical Tests
- **Chi-square test**: Accuracy comparison
- **Mann-Whitney U test**: Time comparison
- **Confidence intervals**: Performance ranges

### Manufacturing Metrics
- **Data Quality Handling**: Performance across Q0-Q3 conditions
- **Task Complexity Performance**: Easy/medium/hard breakdown
- **Cross-System Validation**: Multi-source integration success
- **Domain Accuracy**: Manufacturing-specific task performance

## 🔧 Configuration

### Task Prompt Configurations
Edit `config/task_prompts_variations.yaml` to modify:
- Prompt templates for each length
- Response format templates
- Task-specific instructions

### Evaluation Parameters
Modify evaluation settings in the respective Python files:
- Benchmark scoring weights
- Statistical test parameters
- Visualization styles

## 📋 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scipy (for statistical tests)
- PyYAML (for configuration files)

## 🎯 Use Cases

### Model Selection
- Compare multiple LLM providers
- Identify best model for specific task types
- Optimize cost vs performance trade-offs

### Prompt Engineering
- Evaluate prompt length effectiveness
- Optimize token usage
- Balance accuracy and efficiency

### Performance Monitoring
- Track model performance over time
- Identify degradation or improvements
- Benchmark against baselines

### Manufacturing Domain Analysis
- Assess manufacturing-specific capabilities
- Evaluate data quality handling
- Optimize for industrial use cases

## 🚀 Next Steps

1. **Run the demo** to see the system in action
2. **Review generated reports** for insights
3. **Customize prompts** based on effectiveness analysis
4. **Implement continuous evaluation** for new models
5. **Extend metrics** for specific use cases

## 📞 Support

For questions or issues with the evaluation system, please refer to the generated reports and visualizations for detailed analysis results.
