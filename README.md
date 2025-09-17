# Manufacturing Data Assistant

A comprehensive framework for evaluating and benchmarking Large Language Models (LLMs) against human performance on manufacturing data analysis tasks.

## 🎯 Project Overview

This project provides a complete pipeline for generating manufacturing datasets with controlled quality conditions, collecting human performance baselines, evaluating multiple LLM models, and performing comparative analysis. The system assesses performance across different:

- **Models** (DeepSeek, GPT-4, Claude, etc.)
- **Task Type** (E, M, H)
- **Data Quality Conditions** (Q0-Q3: perfect to corrupted data)
- **Prompt Lengths** (short, normal, long)

## 🏗️ Project Structure

The project is organized into five sequential phases:

### Phase 1: Data Generation & Environment Setup

Generates experimental datasets with controlled data quality conditions:
- Q0: Perfect baseline data (0% corruption)
- Q1: Space injection corruption (~10%)
- Q2: Character missing corruption (~10%)
- Q3: Missing records corruption (~10%)

**Key Components:**
- `run_phase1_generation.py`: Main data generation script
- `data/ground_truth/`: Ground truth answers for evaluation
- `data/experimental_datasets/`: Generated datasets with varying quality

### Phase 2: Multi-Agent LLM System

Implements the LLM agent architecture for manufacturing data analysis:
- Master Agent: Query orchestration with error handling
- Specialist Agents: Data retrieval, reconciliation, and synthesis
- Manufacturing Tools Suite: Domain-specific tools

**Key Components:**
- `src/agents/`: Multi-agent architecture implementation
- `src/tools/`: Manufacturing tools suite
- `config/`: Agent and prompt configurations

### Phase 3: Human Study Execution

Collects human performance baseline data on manufacturing tasks:
- Pattern-Pair design with 9 participants
- 90 total tasks across task types and data quality conditions
- Standardized task execution environment

**Key Components:**
- `run_phase3_human_study.py`: Human study platform
- `experiments/human_study/`: Study results and participant data

### Phase 4: LLM Evaluation

Evaluates LLM performance on identical tasks to human study:
- Multiple models (DeepSeek, Claude, GPT)
- Various prompt lengths (short, normal, long)
- Comprehensive performance metrics

**Key Components:**
- `run_phase4_llm_evaluation.py`: Standard evaluation
- `run_separated_evaluation.py`: Comprehensive multi-model evaluation
- `experiments/llm_evaluation/`: Performance logs and results

### Phase 5: Comparative Analysis

Performs statistical analysis comparing human vs LLM performance:
- Accuracy, speed, and cost comparisons
- Task type and data quality analysis
- Visualization generation and reporting

**Key Components:**
- `experiments/phase5_analysis/`: Analysis framework
- `individual_visualizations/`: 18 specialized analysis charts
- `results/`: Comprehensive reports and findings

## 📊 Key Findings

- **DeepSeek Reasoner**: 97.4% accuracy (1.54× human baseline)
- **Claude Haiku**: 21.2× faster than humans with 93.0% accuracy
- **Claude Sonnet**: 94.1% accuracy with balanced performance
- 5 out of 6 evaluated models outperformed human baseline
- LLMs showed greater resilience to data quality degradation
- Expected ROI ranges from 500-1,200% annually

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- API keys for LLM providers

### Quick Start

```bash
# 1. Setup environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Phase 1: Data Generation
python run_phase1_generation.py

# 3. Phase 3: Human Study (requires manual participation)
python run_phase3_human_study.py

# 4. Phase 4: LLM Evaluation
python run_separated_evaluation.py

# 5. Phase 5: Comparative Analysis
python run_phase5_analysis.py
```

## 📚 Documentation

- `docs/methodology.md`: Research methodology details
- `docs/deployment_guidelines.md`: Implementation guidelines
- `docs/ground_truth_validation.md`: Validation framework
- `docs/manufacturing_context.md`: Domain context

## 📈 Visualizations

The project generates 18 individual visualization charts for analysis:
- Model comparison charts (accuracy, speed, cost)
- Task type analysis charts
- Data quality analysis charts
- Overall performance comparison charts

## 🔧 Troubleshooting

See `PROJECT_EXECUTION_GUIDE.md` for detailed troubleshooting steps and validation scripts.
