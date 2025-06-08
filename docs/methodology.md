## docs/methodology.md

````markdown
# Manufacturing Data Assistant Methodology

## Overview

This document details the comprehensive research methodology for evaluating LLM-based systems in manufacturing data retrieval and synthesis tasks, compared to manual human performance under varying data quality conditions.

## Research Question

**Primary RQ**: How effectively can LLM-based systems assist factory workers in retrieving and synthesising tracing and certification data from heterogeneous digital systems, under varying data quality conditions, compared to manual data retrieval methods?

**Secondary Questions**:
- What are the cost-effectiveness trade-offs between human effort and AI system deployment?
- How do different LLM models perform in manufacturing data analysis tasks?
- What data quality conditions most significantly impact task completion effectiveness?

## Methodology Overview

### Five-Phase Research Design

Phase 1: Data Generation & Quality Control
↓
Phase 2: LLM Agent System Implementation
↓
Phase 3: Human Study Execution (Pattern-Pair Design)
↓
Phase 4: Multi-Model LLM Evaluation
↓
Phase 5: Comparative Analysis & Results

## Phase 1: Scenario Definition and Data Corpus Generation

### 1.1 Manufacturing Environment

**Context**: 3D printing factory producing aerospace components with FAA certification requirements.

**Data Sources**:
- **Location Data**: Barcode scanning at 5 process stations
- **Machine Logs**: 10 industrial 3D printers with API logging
- **Relationship Data**: Bidirectional parent-child mappings
- **Worker Data**: RFID activity tracking for task accountability
- **Document System**: FAA 8130-3 airworthiness certificates

**Manufacturing Process Flow**:

Materials (Goods In) → Printer Setup → 3D Printing →
Job End Buffer → Parts Warehouse → Order Packing → Goods Out

### 1.2 Task Complexity Levels

**Easy (Direct Traversal)**:
- Query: "Find all gears for Order X"
- Data Sources: Relationship + Location data
- Expected Time: Human 3min, LLM 30sec

**Medium (Cross-System Validation)**:
- Query: "Determine printer for Part Y and count parts printed"
- Data Sources: Relationship + Machine logs + Location data
- Expected Time: Human 7min, LLM 1min

**Hard (Document Cross-Reference)**:
- Query: "For Product Z, verify ARC document dates vs warehouse arrival"
- Data Sources: Documents + Location + Relationship data
- Expected Time: Human 15min, LLM 2min

### 1.3 Data Quality Conditions

**Q0 (Baseline)**: Perfect data, 0% corruption
**Q1 (Spaces)**: Random space injection in barcodes (~15% of entries)
**Q2 (Character Missing)**: Random character removal from gear/order IDs
**Q3 (Missing Records)**: Strategic removal of relationship links (~5-8%)

## Phase 2: LLM Agent System Architecture

### 2.1 Performance-Optimized Model Selection

Based on intelligence/price/speed analysis:

ChatGPT -> Claude -> DeepSeek

### 2.2 Multi-Agent Architecture

**Master Agent**: Query orchestration with multi-tier error handling
**Specialist Agents**:
- Data Retrieval: Multi-source querying with fuzzy matching
- Reconciliation: Cross-system validation with missing data handling
- Synthesis: Manufacturing-context report generation

### 2.3 Manufacturing Tools Suite

- **Location Query Tool**: Barcode movement tracking
- **Machine Log Tool**: Printer operation analysis
- **Relationship Tool**: Parent-child traversal with alternative paths
- **Worker Data Tool**: RFID activity correlation
- **Document Parser Tool**: FAA 8130-3 certificate extraction
- **Barcode Validator Tool**: Format compliance verification

## Phase 3: Human Study Design

### 3.1 Pattern-Pair Counterbalancing (3×3 Design)

**Innovation**: Systematic counterbalancing using quality and prompt patterns rather than simple randomization.

**Quality Patterns**:
- PQ1: Q0×5, Q1×2, Q2×2, Q3×1
- PQ2: Q0×5, Q1×2, Q2×1, Q3×2
- PQ3: Q0×5, Q1×1, Q2×2, Q3×2

**Prompt Patterns**:
- PC1: E×4, M×3, H×3
- PC2: E×3, M×4, H×3
- PC3: E×3, M×3, H×4

**Participant Matrix**:
       PC1    PC2    PC3

PQ1    P1     P2     P3

PQ2    P4     P5     P6

PQ3    P7     P8     P9


**Results**: Perfect global balance (E/M/H = 30 each, Q0 = 45, Q1/Q2/Q3 = 15 each)

### 3.2 Data Collection Protocol

**Method**: Manual researcher recording (no audio/video)
**Participants**: 9 participants with manufacturing/data analysis familiarity
**Tasks per Participant**: 10 tasks with randomized order
**Response Format**: Structured forms (Google Forms compatible)

**Metrics Collected**:
- Task completion time (seconds precision)
- Response accuracy (binary scoring)
- Error detection success (Q1/Q2/Q3 identification)
- Response completeness
- Data sources utilized

## Phase 4: Multi-Model LLM Evaluation

### 4.1 Comparative Testing

**Models**: DeepSeek-R1, GPT-o4-mini, Claude-Sonnet-4
**Tasks**: Identical 90-task distribution as human study
**Metrics**: Accuracy, completion time, API cost, error detection, token usage

### 4.2 Performance Analysis

**Primary Metrics**:
- Manufacturing task accuracy
- Cost per correct answer
- Error detection capability
- Cross-system validation success

**Secondary Metrics**:
- Token efficiency
- Alternative reasoning success rates
- Confidence score accuracy

## Phase 5: Analysis Framework

### 5.1 Ground Truth Establishment

**Method**: Baseline data traversal from Q0 (perfect) dataset
**Validation**: Manual expert verification of answer paths
**Scoring**: Binary (correct/incorrect) with error detection requirements

**Success Criteria**:
- Easy: ±95% accuracy in gear identification
- Medium: Correct printer ID and part count
- Hard: Accurate date discrepancy detection + error flagging for Q1/Q2/Q3

### 5.2 Statistical Analysis

**Comparative Tests**:
- Human vs LLM performance across complexity levels
- Data quality condition impact analysis
- Cost-effectiveness ROI calculations

**Manufacturing Validity Assessment**:
- Operational applicability in real factory environments
- Cross-system integration effectiveness
- Regulatory compliance implications

## Expected Outcomes

### 5.3 Research Deliverables

1. **Performance Comparison**: Quantitative human vs LLM effectiveness
2. **Cost-Benefit Analysis**: Operational deployment ROI
3. **Data Quality Impact**: Sensitivity analysis across Q0-Q3 conditions
4. **Model Recommendations**: Optimal LLM selection for manufacturing
5. **Implementation Guidelines**: Practical deployment strategies

### 5.4 Industry Applications

- Manufacturing AI deployment frameworks
- Human-AI collaboration optimization
- Quality control system enhancement
- Regulatory compliance automation
- Supply chain traceability improvement

## Methodological Rigor

### Controls and Validation

- **Counterbalancing**: 3×3 pattern-pair design eliminates order effects
- **Ground Truth**: Systematic baseline answer generation
- **Error Tracking**: Dual output files for corruption validation
- **Replication**: Reproducible random seeds and documented procedures
- **Expert Validation**: Manufacturing domain expert review

### Limitations and Mitigations

- **Sample Size**: 9 participants balanced by sophisticated counterbalancing
- **Task Generalizability**: Three complexity levels covering operational range
- **Data Quality**: Controlled corruption types representing real manufacturing issues
- **Model Selection**: Performance-based selection considering cost and capability
