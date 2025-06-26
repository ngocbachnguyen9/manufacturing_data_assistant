# Manufacturing Data Assistant - Complete Project Execution Guide

This guide provides step-by-step instructions for running the complete Manufacturing Data Assistant project from Phase 1 through Phase 5.

## Prerequisites

### System Requirements
- Python 3.8+
- Virtual environment (recommended)
- Sufficient disk space (~2GB for all datasets and results)

### Environment Setup

1. **Clone and Setup Environment**
```bash
cd manufacturing_data_assistant
python -m venv venv
source venv/bin/activate  # On Windows: venv***REMOVED***Scripts***REMOVED***activate
```

2. **Install Dependencies**
```bash
# Core dependencies
pip install -r requirements.txt

# Phase 5 additional dependencies
pip install pandas>=1.3.0 numpy>=1.20.0 matplotlib>=3.3.0 seaborn>=0.11.0 scipy>=1.7.0
```

3. **Configure API Keys** (for LLM evaluation)
```bash
# Create .env file with your API keys
echo "OPENAI_API_KEY=your_openai_key" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_key" >> .env
echo "DEEPSEEK_API_KEY=your_deepseek_key" >> .env
```

## Phase-by-Phase Execution

### Phase 1: Data Generation & Environment Setup

**Purpose**: Generate experimental datasets with controlled data quality conditions

**Required Files**:
- `data/manufacturing_base/*.csv` (base manufacturing data)
- `config/experiment_config.yaml`

**Execution**:
```bash
# Generate Q0 baseline and Q1/Q2/Q3 corrupted datasets
python run_phase1_generation.py
```

**Expected Outputs**:
- `data/experimental_datasets/Q0_baseline/` - Perfect baseline data
- `data/experimental_datasets/Q1_dataset/` - Space injection corruption (15%)
- `data/experimental_datasets/Q2_dataset/` - Character missing corruption (12%)
- `data/experimental_datasets/Q3_dataset/` - Missing records corruption (7%)
- `experiments/human_study/dirty_ids.json` - Corrupted entity tracking
- `data/ground_truth/baseline_answers.json` - Ground truth answers
- `data/generated_documents/` - FAA certificates and packing lists

**Validation**:
```bash
# Verify data generation
python scripts/validate_ground_truth_alignment.py
python scripts/test_corruption_recovery.py
```

### Phase 2: Multi-Agent LLM System (No separate execution)

**Purpose**: The LLM agent architecture is implemented and ready for Phase 4 evaluation

**Key Components**:
- `src/agents/` - Multi-agent architecture
- `src/tools/` - Manufacturing tools suite
- `config/agent_config.yaml` - Agent configuration
- `config/task_prompts.yaml` - Standard prompts
- `config/task_prompts_variations.yaml` - Prompt length variations

### Phase 3: Human Study Execution

**Purpose**: Collect human performance baseline data

**Prerequisites**: Phase 1 must be completed (requires `dirty_ids.json`)

**Execution**:
```bash
# Generate participant assignments and run study platform
python run_phase3_human_study.py
```

**Expected Outputs**:
- `experiments/human_study/participant_assignments.json` - Task assignments
- `experiments/human_study/Participants_results.csv` - Human performance data
- `session_logs/` - Session recordings

**Manual Process**: The study platform provides a CLI interface for human participants to complete tasks. Researcher manually records responses.

### Phase 4: LLM Evaluation

**Purpose**: Evaluate LLM performance on identical tasks to human study

**Prerequisites**: 
- Phase 1 completed (datasets available)
- Phase 3 completed (task assignments available)
- API keys configured

#### Option A: Standard LLM Evaluation
```bash
# Basic single-model evaluation using default prompts
python run_phase4_llm_evaluation.py
```

#### Option B: Comprehensive Multi-Model Evaluation
```bash
# Separated evaluation: 3 models × 3 prompt lengths = 9 runs
python run_separated_evaluation.py
```

#### Option C: Interactive Evaluation
```bash
# User-controlled model and prompt selection
python run_interactive_evaluation.py
```

#### Option D: Unbiased Evaluation
```bash
# Multiple judge models with consensus scoring
python run_unbiased_evaluation.py
```

**Expected Outputs**:
- `experiments/llm_evaluation/benchmark_*/` - Individual model results
- `experiments/llm_evaluation/performance_logs/` - Aggregated performance data
- `experiments/llm_evaluation/separated_runs/` - Separated evaluation results

### Phase 5: Comparative Analysis

**Purpose**: Statistical comparison of human vs LLM performance with cost analysis

**Prerequisites**: 
- Phase 3 completed (human data available)
- Phase 4 completed (LLM data available)

**Execution**:
```bash
# Complete comparative analysis with default settings
python run_phase5_analysis.py

# Custom configuration
python run_phase5_analysis.py ***REMOVED***
  --human-data experiments/human_study/Participants_results.csv ***REMOVED***
  --llm-data experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark ***REMOVED***
  --human-hourly-rate 30.0 ***REMOVED***
  --output-dir custom_results ***REMOVED***
  --viz-dir custom_visualizations
```

**Expected Outputs**:
- `experiments/phase5_analysis/results/` - CSV exports and analysis results
- `experiments/phase5_analysis/visualizations/` - Performance comparison charts
- `experiments/phase5_analysis/results/phase5_analysis_report.md` - Comprehensive report

#### Additional Phase 5 Analysis Tools
```bash
# Model ranking summary
python experiments/phase5_analysis/model_ranking_summary.py

# Quick insights
python experiments/phase5_analysis/quick_insights.py

# Visualization guide
python experiments/phase5_analysis/visualization_guide.py
```

## Complete End-to-End Execution

**For a complete project run from start to finish**:

```bash
# 1. Setup environment (one-time)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pandas numpy matplotlib seaborn scipy

# 2. Phase 1: Data Generation
python run_phase1_generation.py

# 3. Phase 3: Human Study (requires manual participation)
python run_phase3_human_study.py

# 4. Phase 4: LLM Evaluation (choose one option)
python run_separated_evaluation.py  # Recommended for comprehensive results

# 5. Phase 5: Comparative Analysis
python run_phase5_analysis.py
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure `.env` file contains valid API keys for LLM providers
2. **Import Errors**: Verify all dependencies are installed, especially Phase 5 requirements
3. **File Not Found**: Ensure previous phases completed successfully before proceeding
4. **Memory Issues**: Large datasets may require sufficient RAM (8GB+ recommended)

### Validation Scripts

```bash
# Test core functionality
python runtests.py

# Validate specific components
python tests/test_phase_1_runner.py
python tests/test_task_generator.py
python tests/test_llm_evaluation.py
```

### Data Verification

```bash
# Verify ground truth alignment
python scripts/validate_ground_truth_alignment.py

# Test corruption recovery
python scripts/test_corruption_recovery.py

# Test task queries
python scripts/test_task_queries.py
```

## Expected Timeline

- **Phase 1**: 5-10 minutes (data generation)
- **Phase 3**: 2-4 hours (human study with 9 participants)
- **Phase 4**: 30-60 minutes (LLM evaluation, depending on API speed)
- **Phase 5**: 5-10 minutes (analysis and visualization)

**Total Project Runtime**: ~3-5 hours (including human study time)

## Output Summary

Upon completion, you will have:
- Complete experimental datasets with controlled quality conditions
- Human performance baseline (9 participants, 90 tasks)
- LLM performance evaluation (multiple models and prompt variations)
- Statistical comparison analysis with visualizations
- Cost-effectiveness analysis and ROI calculations
- Comprehensive research findings and recommendations

For detailed analysis of results, refer to:
- `experiments/phase5_analysis/results/phase5_analysis_report.md`
- `experiments/phase5_analysis/visualizations/` directory
- Individual phase output directories for detailed data
