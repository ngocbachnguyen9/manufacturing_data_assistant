# Phase 5 Analysis System Architecture
**Generated:** 2025-06-27 17:45:00  
**Confidence Level:** 95%

## 🎯 System Overview

The Phase 5 Analysis System is a comprehensive framework for comparing human performance against Large Language Model (LLM) performance on manufacturing data analysis tasks. The system processes input data, performs statistical analysis, and generates detailed reports and visualizations.

## 📊 Input Data Sources

### Human Study Data
- **Location:** `experiments/human_study/Participants_results.csv`
- **Format:** CSV with UTF-8/UTF-8-BOM encoding
- **Structure:**
  ```
  task_id, participant_id, completion_time_sec, accuracy, comments, 
  complexity, data quality, notes
  ```
- **Content:** 90 task records from 9 participants (P1-P9)
- **Task Distribution:** 30 easy, 30 medium, 30 hard tasks
- **Quality Conditions:** Q0 (normal), Q1-Q3 (corrupted data)

### LLM Evaluation Results
- **Location:** `experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark/`
- **Format:** Multiple CSV files (one per model)
- **Naming Convention:** `{model_name}_short_all_{date}.csv`
- **Structure:**
  ```
  task_id, model, complexity, quality_condition, completion_time_sec,
  is_correct, total_cost_usd, input_tokens, output_tokens, final_confidence,
  reconciliation_issues, llm_final_report, ground_truth_answer,
  judge_consensus_score, judge_details, total_judges, agreement_level
  ```
- **Content:** 1,620 task records (270 per model × 6 models)

## 🚀 Main Entry Points

### 1. Primary Analysis Runner
**Script:** `run_phase5_analysis.py`
- **Purpose:** Main orchestrator for complete Phase 5 analysis
- **Usage:** 
  ```bash
  python run_phase5_analysis.py ***REMOVED***
    --human-data experiments/human_study/Participants_results.csv ***REMOVED***
    --llm-data experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark ***REMOVED***
    --output-dir experiments/phase5_analysis/results ***REMOVED***
    --viz-dir experiments/phase5_analysis/visualizations
  ```
- **Outputs:** Complete analysis with CSV exports, JSON results, and visualizations

### 2. Individual Visualizations Generator
**Script:** `run_phase5_individual_visualizations.py`
- **Purpose:** Generate specialized individual charts
- **Usage:**
  ```bash
  python run_phase5_individual_visualizations.py ***REMOVED***
    --output-dir experiments/phase5_analysis/individual_visualizations
  ```
- **Outputs:** 20+ specialized visualization charts

### 3. Comprehensive Model Analysis
**Script:** `comprehensive_individual_analysis.py`
- **Purpose:** Detailed individual model analysis with statistical insights
- **Usage:**
  ```bash
  python experiments/phase5_analysis/comprehensive_individual_analysis.py
  ```
- **Outputs:** Detailed markdown reports and advanced visualizations

## ⚙️ Core Analysis Engine

### Configuration Management
**Class:** `ComparisonConfig`
- **Purpose:** Centralized configuration for analysis parameters
- **Key Settings:**
  - Human hourly rate ($25/hour default)
  - Output directories
  - Visualization settings
  - Export options

### Main Analysis Class
**Class:** `HumanVsLLMComparison`
- **Purpose:** Core analysis engine orchestrating all comparisons
- **Key Methods:**
  - `load_human_data()`: CSV parsing with encoding handling
  - `load_llm_data()`: Multi-file aggregation and preprocessing
  - `align_data_by_task_id()`: Data alignment and validation
  - `perform_statistical_analysis()`: Statistical testing and analysis
  - `create_visualization_dashboard()`: Visualization generation

### Data Processing Pipeline

#### 1. Data Loading & Preprocessing
- **Human Data Processing:**
  - UTF-8/BOM encoding detection
  - Column name standardization
  - Cost calculation (time × hourly rate)
  - Data validation and cleaning

- **LLM Data Processing:**
  - Multi-file CSV aggregation
  - Boolean conversion (string → numeric)
  - Model identification and labeling
  - Data consistency validation

#### 2. Data Alignment
- **Task ID Matching:** Ensures human and LLM data cover same tasks
- **Quality Validation:** Verifies data completeness and consistency
- **Sample Size Calculation:** Determines statistical power for tests

#### 3. Statistical Analysis
- **Chi-square Tests:** Accuracy comparisons (categorical data)
- **Mann-Whitney U Tests:** Completion time comparisons (non-parametric)
- **Confidence Intervals:** Bootstrap and t-distribution methods
- **Effect Size Calculations:** Cohen's d, Glass's Delta, Hedges' g

## 🔬 Analysis Modules

### Performance Analysis
1. **Overall Performance:** Aggregate accuracy, speed, and cost metrics
2. **Complexity Analysis:** Performance breakdown by task difficulty (easy/medium/hard)
3. **Quality Analysis:** Robustness to data quality degradation (Q0-Q3)
4. **Model-Specific Analysis:** Individual model profiles and comparisons

### Advanced Analysis
1. **Individual Model Analysis:** Comprehensive statistical profiles
2. **Head-to-Head Comparisons:** Pairwise model comparisons
3. **Risk Assessment:** Deployment readiness evaluation
4. **Cost-Effectiveness Analysis:** ROI calculations and scenarios

### Cost Analysis Module
**Class:** `CostAnalyzer`
- **Purpose:** Economic impact assessment
- **Features:**
  - Multiple cost scenarios (technician, analyst, engineer)
  - ROI calculations with break-even analysis
  - Total cost of ownership modeling
  - Sensitivity analysis

## 📈 Visualization Generation

### Main Dashboard Generator
**Method:** `create_visualization_dashboard()`
- **Purpose:** Generate comprehensive visualization suite
- **Chart Types:**
  - Accuracy vs Speed scatter plots
  - Model ranking charts
  - Performance heatmaps
  - Statistical significance matrices
  - Performance radar charts

### Individual Chart Generators
**Module:** `individual_visualizations.py`
- **Specialized Charts:**
  - Complexity performance comparisons
  - Quality robustness rankings
  - Confidence analysis charts
  - Error pattern analysis
  - Provider-based comparisons

### Advanced Visualizations
**Module:** `create_comprehensive_visualizations.py`
- **Advanced Charts:**
  - Model deployment tiers
  - Risk assessment matrices
  - Use case recommendation heatmaps
  - Confidence interval plots

## 📄 Output Generation

### CSV Exports (Structured Data)
1. `model_specific_comparison.csv` - Individual model vs human metrics
2. `model_complexity_analysis.csv` - Performance by task complexity
3. `model_quality_analysis.csv` - Performance by data quality condition
4. `task_level_comparison.csv` - Task-by-task detailed comparison
5. `statistical_tests.csv` - Statistical test results and significance
6. `overall_comparison.csv` - Aggregate performance summary

### JSON Results (Complete Data)
- `phase5_results.json` - Complete analysis results in structured format
- Includes all statistical tests, model data, and generated plot paths
- Machine-readable format for further analysis

### Markdown Reports (Human-Readable)
1. `phase5_analysis_report.md` - Executive summary with key findings
2. `comprehensive_individual_model_analysis.md` - Detailed model analysis
3. `model_selection_framework.md` - Decision framework and recommendations
4. `executive_summary.md` - Strategic overview for stakeholders

### Visualizations (Three Directories)
1. `visualizations/` - Main dashboard charts
2. `individual_visualizations/` - Specialized analysis charts
3. `results/` - Analysis-specific advanced charts

## 🛠️ Supporting Tools

### Quick Analysis Tools
1. `quick_insights.py` - Rapid analysis summary and recommendations
2. `model_ranking_summary.py` - Model performance rankings
3. `visualization_guide.py` - Chart documentation and usage guide

## 🔄 System Workflow

### Standard Execution Flow
1. **Initialization:** Load configuration and validate input paths
2. **Data Loading:** Process human and LLM data files
3. **Data Alignment:** Match tasks and validate data consistency
4. **Statistical Analysis:** Perform comprehensive statistical testing
5. **Analysis Modules:** Execute performance, complexity, and quality analysis
6. **Visualization Generation:** Create charts and dashboards
7. **Output Generation:** Export CSV, JSON, and markdown results
8. **Report Generation:** Create human-readable analysis reports

### Parallel Execution Paths
- **Main Analysis:** Core statistical comparison (run_phase5_analysis.py)
- **Individual Charts:** Specialized visualizations (run_phase5_individual_visualizations.py)
- **Comprehensive Analysis:** Advanced model analysis (comprehensive_individual_analysis.py)

## 📋 Quality Assurance

### Data Validation
- Input file existence and format validation
- Data consistency checks across sources
- Sample size adequacy verification
- Statistical assumption validation

### Error Handling
- Graceful handling of missing files
- Encoding detection and conversion
- Data type validation and conversion
- Statistical test assumption checking

### Output Validation
- Chart generation verification
- CSV export completeness
- Report generation success
- File path and naming consistency

## 🔧 Configuration Options

### Command Line Arguments
- `--human-data`: Path to human study CSV file
- `--llm-data`: Path to LLM results directory
- `--output-dir`: Results output directory
- `--viz-dir`: Visualizations output directory
- `--human-hourly-rate`: Cost calculation parameter
- `--no-csv`: Skip CSV exports
- `--no-viz`: Skip visualization generation
- `--no-cost-analysis`: Skip cost analysis

### Environment Requirements
- Python 3.9+
- Required packages: pandas, numpy, scipy, matplotlib, seaborn
- Minimum 4GB RAM for full analysis
- 1GB disk space for complete output

---
*System architecture documented with 95% confidence based on comprehensive code analysis*
