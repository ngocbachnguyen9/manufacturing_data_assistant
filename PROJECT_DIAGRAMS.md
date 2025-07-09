# Project Diagrams

## Project Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Manufacturing Data Assistant                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────────────┐
    │                           │                                   │
    ▼                           ▼                                   ▼
┌─────────────┐          ┌─────────────┐                    ┌─────────────┐
│ Data Sources │          │ Evaluation  │                    │  Analysis   │
└──────┬──────┘          └──────┬──────┘                    └──────┬──────┘
       │                        │                                  │
       ▼                        ▼                                  ▼
┌──────────────┐        ┌──────────────┐                  ┌──────────────┐
│ Location Data │        │ Human Study  │                  │ Performance  │
│ Machine Logs  │        │ LLM Testing  │                  │ Comparison   │
│ Relationship  │───────▶│ Multi-Model  │─────────────────▶│ Visualization│
│ Documents     │        │ Evaluation   │                  │ Reporting    │
└──────────────┘        └──────────────┘                  └──────────────┘
       │                        │                                  │
       ▼                        ▼                                  ▼
┌──────────────┐        ┌──────────────┐                  ┌──────────────┐
│ Data Quality  │        │ Performance  │                  │ Deployment   │
│ Q0: Perfect   │        │ - Accuracy   │                  │ Readiness    │
│ Q1: Spaces    │        │ - Speed      │                  │ Risk Analysis│
│ Q2: Missing   │        │ - Cost       │                  │ ROI          │
│ Q3: Records   │        │ - Confidence │                  │ Guidelines   │
└──────────────┘        └──────────────┘                  └──────────────┘
```

## Phase 1: Data Generation & Environment Setup

```
┌─────────────────────────────────────────────────────────┐
│           Phase 1: Data Generation & Setup              │
└───────────────────────────┬─────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌───────────────────┐      ┌───────────────────┐
    │   Data Sources    │      │  Quality Control  │
    └─────────┬─────────┘      └─────────┬─────────┘
              │                           │
    ┌─────────┴─────────┐      ┌─────────┴─────────┐
    ▼                   ▼      ▼                   ▼
┌─────────┐       ┌─────────┐ ┌─────────┐       ┌─────────┐
│Location │       │Document │ │   Q0    │       │   Q2    │
│  Data   │       │ System  │ │ Perfect │       │Character│
└─────────┘       └─────────┘ │Baseline │       │ Missing │
                              └─────────┘       └─────────┘
┌─────────┐       ┌─────────┐ ┌─────────┐       ┌─────────┐
│Machine  │       │Relation-│ │   Q1    │       │   Q3    │
│  Logs   │       │ship Data│ │ Space   │       │ Missing │
└─────────┘       └─────────┘ │Injection│       │ Records │
                              └─────────┘       └─────────┘
                                    │
                                    ▼
                            ┌───────────────────┐
                            │  Output Datasets  │
                            └─────────┬─────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      ▼               ▼               ▼
              ┌───────────────┐┌───────────────┐┌───────────────┐
              │ Ground Truth  ││ Experimental  ││ Validation    │
              │   Answers    ││   Datasets    ││    Scripts    │
              └───────────────┘└───────────────┘└───────────────┘
```

## Phase 2: Multi-Agent LLM System

```
┌─────────────────────────────────────────────────────────┐
│             Phase 2: Multi-Agent LLM System             │
└───────────────────────────┬─────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌───────────────────┐      ┌───────────────────┐
    │   Agent System    │      │    Tool Suite     │
    └─────────┬─────────┘      └─────────┬─────────┘
              │                           │
              ▼                           ▼
    ┌───────────────────┐      ┌───────────────────┐
    │   Master Agent    │      │ Manufacturing     │
    │                   │      │ Domain Tools      │
    │ - Orchestration   │      │                   │
    │ - Error Handling  │      │ - Data Retrieval  │
    │ - Task Management │      │ - Entity Lookup   │
    └─────────┬─────────┘      │ - Relationship    │
              │                │   Mapping         │
              │                └───────────────────┘
              │
              ▼
    ┌───────────────────┐
    │ Specialist Agents │
    │                   │
    │ - Data Retrieval  │
    │ - Reconciliation  │
    │ - Synthesis       │
    └───────────────────┘
              │
              ▼
    ┌───────────────────┐
    │  Configuration    │
    │                   │
    │ - Agent Config    │
    │ - Task Prompts    │
    │ - Prompt Variants │
    └───────────────────┘
```

## Phase 3: Human Study Execution

```
┌─────────────────────────────────────────────────────────┐
│             Phase 3: Human Study Execution              │
└───────────────────────────┬─────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌───────────────────┐      ┌───────────────────┐
    │   Study Design    │      │  Data Collection  │
    └─────────┬─────────┘      └─────────┬─────────┘
              │                           │
              ▼                           ▼
    ┌───────────────────┐      ┌───────────────────┐
    │  Pattern-Pair     │      │ Performance       │
    │  Design           │      │ Metrics           │
    │                   │      │                   │
    │ - 9 Participants  │      │ - Accuracy        │
    │ - 90 Total Tasks  │      │ - Completion Time │
    │ - Balanced        │      │ - Error Detection │
    │   Distribution    │      │ - Data Source Use │
    └───────────────────┘      │ - Total Cost      │
                               └───────────────────┘
                                         │
                                         ▼
                               ┌───────────────────┐
                               │  Study Platform   │
                               │                   │
                               │ - Task Interface  │
                               │ - Time Tracking   │
                               │ - Answer Logging  │
                               │ - User Experience │
                               └─────────┬─────────┘
                                         │
                                         ▼
                               ┌───────────────────┐
                               │  Study Results    │
                               │                   │
                               │ - Participant Data│
                               │ - Performance Logs│
                               │ - Session Records │
                               └───────────────────┘
```

## Phase 4: LLM Evaluation

```
┌─────────────────────────────────────────────────────────┐
│               Phase 4: LLM Evaluation                   │
└───────────────────────────┬─────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌───────────────────┐      ┌───────────────────┐
    │  Model Evaluation │      │ Prompt Variations │
    └─────────┬─────────┘      └─────────┬─────────┘
              │                           │
              ▼                           ▼
    ┌───────────────────┐      ┌───────────────────┐
    │  Models Tested    │      │  Prompt Types     │
    │                   │      │                   │
    │ - DeepSeek        │      │ - Short           │
    │ - Claude          │      │ - Normal          │
    │ - GPT-4           │      │ - Long            │
    │ - Others          │      │                   │
    └───────────────────┘      └───────────────────┘
              │                           │
              └───────────────┬───────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Evaluation Metrics │
                    │                   │
                    │ - Accuracy        │
                    │ - Completion Time │
                    │ - API Cost        │
                    │ - Token Usage     │
                    │ - Error Detection │
                    │ - Token Efficiency│
                    │ - Confidence      │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Evaluation Runners│
                    │                   │
                    │ - Standard        │
                    │ - Comprehensive   │
                    │ - Interactive     │
                    │ - Unbiased        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Performance Logs  │
                    │ & Results         │
                    └───────────────────┘
```

## Phase 5: Comparative Analysis

```
┌─────────────────────────────────────────────────────────┐
│             Phase 5: Comparative Analysis               │
└───────────────────────────┬─────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌───────────────────┐      ┌───────────────────┐
    │ Statistical       │      │ Visualization     │
    │ Analysis          │      │ Generation        │
    └─────────┬─────────┘      └─────────┬─────────┘
              │                           │
              ▼                           ▼
    ┌───────────────────┐      ┌───────────────────┐
    │ Comparison Metrics│      │ Chart Types       │
    │                   │      │                   │
    │ - Accuracy Diff   │      │ - Model Comparison│
    │ - Speed Factor    │      │ - Complexity      │
    │ - Cost Ratio      │      │   Analysis        │
    │ - Significance    │      │ - Quality Analysis│
    │   Testing         │      │ - Overall         │
    └───────────────────┘      │   Comparison      │
              │                └───────────────────┘
              │                           │
              └───────────────┬───────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Advanced Analysis │
                    │                   │
                    │ - Deployment      │
                    │   Readiness       │
                    │ - Risk Assessment │
                    │ - Use Case        │
                    │   Recommendations │
                    │ - ROI Calculation │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Output Generation │
                    │                   │
                    │ - CSV Exports     │
                    │ - JSON Results    │
                    │ - Markdown Reports│
                    │ - Visualizations  │
                    └───────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Key Findings      │
                    │                   │
                    │ - Model Rankings  │
                    │ - Performance     │
                    │   Insights        │
                    │ - Deployment      │
                    │   Guidelines      │
                    └───────────────────┘