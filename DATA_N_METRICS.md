# Data Sources and Metrics Documentation

This document provides detailed information about data sources, metrics calculation, and data types used throughout the Manufacturing Data Assistant project.

## Phase 1: Data Generation & Ingestion

### Data Sources

#### 1. Location Data System
- **Purpose**: Real-time asset tracking through manufacturing process
- **Format**: CSV with timestamp-based tracking events
- **Key Fields**: 
  - Entity ID (barcodes for tracked items)
  - Location identifier (process station)
  - Timestamp (entry/exit times)
  - Status (entered/exited/in-process)
- **Tracked Entities**:
  - Materials: ABSM0002, ABSM0003, etc.
  - Gears: 3DOR10001 to 3DOR100099
  - Orders: ORBOX0011, ORBOX00111, etc.
  - Workers: RFID card numbers
  - Equipment: Printer_1, Printer_2, etc.

#### 2. Machine Log System
- **Purpose**: Automated 3D printer operation recording
- **Format**: CSV with job execution details
- **Key Fields**:
  - Machine identifier
  - Job start timestamp
  - Job end timestamp
  - Total duration (seconds)
  - Material consumption
  - Part identification

#### 3. Relationship Data
- **Purpose**: Bidirectional parent-child mappings between entities
- **Format**: CSV with relationship pairs
- **Key Fields**:
  - Parent ID
  - Child ID
  - Relationship type
  - Creation timestamp

#### 4. Document System
- **Purpose**: FAA 8130-3 airworthiness certificates and Packing List documents
- **Format**: PDF documents with structured fields
- **Key Fields**:
  - Document ID
  - Issue date
  - Expiration date
  - Related entity IDs
  - Compliance status

### Data Quality Conditions

#### Q0: Perfect Baseline (0% corruption)
- Complete records with perfect data integrity
- Used as ground truth for evaluation

#### Q1: Space Injection (~15% corruption)
- Strategic spaces injected into entity IDs
- Example: "3DOR100081" → " 3DOR100081"
- Challenges string parsing and exact matching

#### Q2: Character Missing (~12% corruption)
- Strategic character removal from entity IDs
- Example: "3DOR100081" → "3DR100081"
- Tests fuzzy matching and error correction

#### Q3: Missing Records (~7% corruption)
- Strategic removal of relationship links
- Tracked in `*_Q3_errors.csv` files
- Tests alternative path finding and data reconciliation

## Phase 3: Human Study Metrics

### Performance Metrics

#### 1. Task Completion Accuracy
- **Calculation**: Binary correct/incorrect scoring
- **Data Type**: Boolean (0/1)
- **Purpose**: Baseline human performance benchmark

#### 2. Completion Time
- **Calculation**: Task start to submission time in seconds
- **Data Type**: Float (seconds)
- **Purpose**: Measure human processing speed

#### 3. Error Detection Rate
- **Calculation**: Percentage of data quality issues correctly identified
- **Data Type**: Qualitative Description
- **Purpose**: Assess human data quality awareness

#### 4. Data Source Utilization
- **Calculation**: Count of data sources accessed during task
- **Data Type**: Integer (1-5)
- **Purpose**: Measure search strategy efficiency

#### 5. Total Cost
- **Calculation**: Completion time × hourly rate ($25/hour)
- **Data Type**: Float (USD)
- **Purpose**: Establish cost baseline for ROI calculations

### Study Design Metrics

#### 1. Pattern-Pair Counterbalancing
- **Quality Patterns**:
  - PQ1: Q0×5, Q1×2, Q2×2, Q3×1
  - PQ2: Q0×5, Q1×2, Q2×1, Q3×2
  - PQ3: Q0×5, Q1×1, Q2×2, Q3×2
- **Prompt Patterns**:
  - PC1: E×4, M×3, H×3
  - PC2: E×3, M×4, H×3
  - PC3: E×3, M×3, H×4
- **Purpose**: Ensure balanced distribution of tasks

## Phase 4: LLM Evaluation Metrics

### Performance Metrics

#### 1. Accuracy
- **Calculation**: Number of correct answers / Total tasks
- **Data Type**: Float (0.0-1.0)
- **Purpose**: Primary performance indicator

#### 2. Completion Time
- **Calculation**: API request to response time in seconds
- **Data Type**: Float (seconds)
- **Purpose**: Measure processing efficiency

#### 3. API Cost
- **Calculation**: Input tokens × input rate + Output tokens × output rate
- **Data Type**: Float (USD)
- **Purpose**: Economic feasibility assessment

#### 4. Token Usage
- **Input Tokens**: Count of tokens in prompt
- **Output Tokens**: Count of tokens in response
- **Data Type**: Integer
- **Purpose**: Resource utilization measurement

#### 5. Error Detection Rate
- **Calculation**: Percentage of data quality issues correctly identified
- **Data Type**: Float (0.0-1.0)
- **Purpose**: Data quality awareness assessment

#### 6. Token Efficiency
- **Calculation**: Correct answers / (Input tokens + Output tokens) × 1000
- **Data Type**: Float (correct answers per 1000 tokens)
- **Purpose**: Efficiency optimization metric

#### 7. Final Confidence
- **Calculation**: Model's self-reported confidence (0.0-1.0)
- **Data Type**: Float
- **Purpose**: Calibration assessment

### Manufacturing-Specific Metrics

#### 1. Data Quality Handling
- **Calculation**: Performance across Q0-Q3 conditions
- **Data Type**: Dictionary of metrics by quality level
- **Purpose**: Robustness assessment

#### 2. Task Complexity Performance
- **Calculation**: Performance across easy/medium/hard tasks
- **Data Type**: Dictionary of metrics by complexity
- **Purpose**: Capability assessment

#### 3. Cross-System Validation
- **Calculation**: Success rate on multi-source data integration
- **Data Type**: Float (0.0-1.0)
- **Purpose**: Integration capability assessment

#### 4. Manufacturing Domain Accuracy
- **Calculation**: Performance on domain-specific tasks
- **Data Type**: Dictionary by task type
- **Purpose**: Domain expertise assessment

## Phase 5: Comparative Analysis Metrics

### Statistical Comparison Metrics

#### 1. Accuracy Difference
- **Calculation**: LLM accuracy - Human accuracy
- **Data Type**: Float (can be positive or negative)
- **Purpose**: Direct performance comparison

#### 2. Speed Improvement Factor
- **Calculation**: Human avg time / LLM avg time
- **Data Type**: Float (multiplier)
- **Purpose**: Efficiency comparison

#### 3. Cost Efficiency Ratio
- **Calculation**: Human avg cost / LLM avg cost
- **Data Type**: Float (multiplier)
- **Purpose**: Economic comparison



### Advanced Analysis Metrics

#### 1. Deployment Readiness Score
- **Calculation**: Weighted combination of:
  - Accuracy Score: min(LLM accuracy / Human accuracy, 2.0) × 0.5
  - Speed Score: min(Speed improvement factor / 20, 1.0) × 0.3
  - Significance Score: (1.0 if significant, 0.5 if not) × 0.2
- **Data Type**: Float (0.0-1.0)
- **Purpose**: Practical deployment decision support

#### 2. Risk Assessment Matrix
- **Components**:
  - Accuracy Risk: Inverse of accuracy
  - Consistency Risk: Standard deviation of performance
  - Robustness Risk: Performance degradation with poor data
  - Significance Risk: Statistical reliability factor
- **Data Type**: Array of floats (0.0-1.0)
- **Purpose**: Risk management and mitigation planning

#### 3. ROI Calculation
- **Calculation**: (Cost savings - Implementation costs) / Implementation costs
- **Data Type**: Float (percentage)
- **Purpose**: Business case justification

## Data Types and Structures

### Primary Data Structures

#### 1. Task Results DataFrame
- **Columns**:
  - task_id: Unique identifier
  - model/participant: Source of result
  - task_types: Task types (E/M/H)
  - quality_condition: Data quality level (Q0-Q3)
  - completion_time_sec: Time to complete
  - is_correct: Accuracy indicator
  - total_cost_usd: Cost calculation
  - input_tokens/output_tokens: Resource usage
  - final_confidence: Self-reported confidence
  - ground_truth_answer: Correct answer
- **Purpose**: Primary analysis dataset

#### 2. Performance Metrics Class
- **Attributes**:
  - accuracy: Overall correctness
  - avg_completion_time: Speed metric
  - avg_cost: Economic metric
  - avg_confidence: Calibration metric
  - error_detection_rate: Quality awareness
  - token_efficiency: Resource efficiency
- **Purpose**: Standardized metrics container

#### 3. Comparative Results Dictionary
- **Structure**: Nested dictionary with:
  - overall_comparison: Aggregate metrics
  - model_specific_comparison: Individual model metrics
  - complexity_analysis: Performance by difficulty
  - quality_analysis: Performance by data condition
  - statistical_tests: Significance testing results
- **Purpose**: Comprehensive analysis container

### Output Data Formats

#### 1. CSV Exports
- model_specific_comparison.csv
- model_complexity_analysis.csv
- model_quality_analysis.csv
- task_level_comparison.csv
- statistical_tests.csv
- overall_comparison.csv

#### 2. JSON Results
- phase5_results.json: Complete structured results

#### 3. Markdown Reports
- phase5_analysis_report.md: Human-readable summary
- comprehensive_individual_model_analysis.md: Detailed model analysis
- model_selection_framework.md: Decision support
- executive_summary.md: High-level findings