# LLM Manufacturing Data Analysis - Executive Benchmark Summary
======================================================================

**Evaluation Date:** 2025-06-17T10:21:46.941207
**Models Evaluated:** deepseek-chat
**Total Tasks Analyzed:** 90

## 🏆 Top Performer
**deepseek-chat** - Overall Score: 0.645
- Accuracy: 67.8%
- Avg Time: 35.6s
- Avg Cost: $0.0000

## 🔍 Key Findings
### Manufacturing Task Performance:
- **Gear Identification**: 83.3% accuracy
- **Printer Analysis**: 76.7% accuracy
- **Compliance Verification**: 43.3% accuracy
### Data Quality Handling:
- **Perfect Data**: 55.6% accuracy
- **Space Errors**: 80.0% accuracy
- **Missing Chars**: 66.7% accuracy
- **Missing Records**: 93.3% accuracy

## 💡 Key Recommendations
### Data Quality
Weakest Performance on: Q0 conditions (55.6% accuracy)
            Recommendation: Implement additional data validation and error handling for Q0 scenarios
            Consider prompt engineering improvements for data quality issue detection

### Task Complexity
Most Challenging Tasks: hard (43.3% accuracy)
            Recommendation: Consider task decomposition or specialized prompting for hard tasks
            Average time for hard tasks: 35.9s

### Manufacturing Domain
Weakest Manufacturing Domain: compliance_verification (43.3% accuracy)
            Recommendation: Develop domain-specific training or fine-tuning for compliance_verification
            Consider adding more context or examples for this task type
