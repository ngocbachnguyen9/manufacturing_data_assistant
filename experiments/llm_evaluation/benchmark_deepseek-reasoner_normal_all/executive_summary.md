# LLM Manufacturing Data Analysis - Executive Benchmark Summary
======================================================================

**Evaluation Date:** 2025-06-18T09:55:20.151363
**Models Evaluated:** deepseek-reasoner
**Total Tasks Analyzed:** 90

## 🏆 Top Performer
**deepseek-reasoner** - Overall Score: 0.640
- Accuracy: 75.6%
- Avg Time: 200.8s
- Avg Cost: $0.0000

## 🔍 Key Findings
### Manufacturing Task Performance:
- **Gear Identification**: 73.3% accuracy
- **Printer Analysis**: 53.3% accuracy
- **Compliance Verification**: 100.0% accuracy
### Data Quality Handling:
- **Perfect Data**: 84.4% accuracy
- **Space Errors**: 80.0% accuracy
- **Missing Chars**: 60.0% accuracy
- **Missing Records**: 60.0% accuracy

## 💡 Key Recommendations
### Data Quality
Weakest Performance on: Q2 conditions (60.0% accuracy)
            Recommendation: Implement additional data validation and error handling for Q2 scenarios
            Consider prompt engineering improvements for data quality issue detection

### Task Complexity
Most Challenging Tasks: medium (53.3% accuracy)
            Recommendation: Consider task decomposition or specialized prompting for medium tasks
            Average time for medium tasks: 274.3s

### Manufacturing Domain
Weakest Manufacturing Domain: printer_analysis (53.3% accuracy)
            Recommendation: Develop domain-specific training or fine-tuning for printer_analysis
            Consider adding more context or examples for this task type
