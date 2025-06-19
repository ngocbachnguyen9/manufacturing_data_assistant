# LLM Manufacturing Data Analysis - Executive Benchmark Summary
======================================================================

**Evaluation Date:** 2025-06-18T17:25:53.896653
**Models Evaluated:** deepseek-reasoner
**Total Tasks Analyzed:** 90

## 🏆 Top Performer
**deepseek-reasoner** - Overall Score: 0.519
- Accuracy: 51.1%
- Avg Time: 294.4s
- Avg Cost: $0.0000

## 🔍 Key Findings
### Manufacturing Task Performance:
- **Gear Identification**: 70.0% accuracy
- **Printer Analysis**: 23.3% accuracy
- **Compliance Verification**: 60.0% accuracy
### Data Quality Handling:
- **Perfect Data**: 51.1% accuracy
- **Space Errors**: 40.0% accuracy
- **Missing Chars**: 40.0% accuracy
- **Missing Records**: 73.3% accuracy

## 💡 Key Recommendations
### Data Quality
Weakest Performance on: Q1 conditions (40.0% accuracy)
            Recommendation: Implement additional data validation and error handling for Q1 scenarios
            Consider prompt engineering improvements for data quality issue detection

### Task Complexity
Most Challenging Tasks: medium (23.3% accuracy)
            Recommendation: Consider task decomposition or specialized prompting for medium tasks
            Average time for medium tasks: 369.8s

### Manufacturing Domain
Weakest Manufacturing Domain: printer_analysis (23.3% accuracy)
            Recommendation: Develop domain-specific training or fine-tuning for printer_analysis
            Consider adding more context or examples for this task type
