# Prompt Effectiveness Analysis Report
==================================================

## Executive Summary

This report analyzes the effectiveness of different prompt lengths (short, normal, long)
on LLM performance across manufacturing data analysis tasks.

## Performance by Prompt Length

### Short Prompts
- **Accuracy**: 87.8%
- **Avg Completion Time**: 21.5s
- **Avg Cost**: $0.0000
- **Avg Confidence**: 0.72
- **Avg Input Tokens**: 1043
- **Token Efficiency**: 0.84 correct/1000 tokens
- **Task Count**: 90

### Normal Prompts
- **Accuracy**: 94.4%
- **Avg Completion Time**: 30.8s
- **Avg Cost**: $0.0000
- **Avg Confidence**: 0.76
- **Avg Input Tokens**: 1837
- **Token Efficiency**: 0.51 correct/1000 tokens
- **Task Count**: 90

### Long Prompts
- **Accuracy**: 94.4%
- **Avg Completion Time**: 36.7s
- **Avg Cost**: $0.0000
- **Avg Confidence**: 0.80
- **Avg Input Tokens**: 2622
- **Token Efficiency**: 0.36 correct/1000 tokens
- **Task Count**: 90

## Recommendations

- **Highest Accuracy**: Normal prompts (94.4%)
- **Fastest Completion**: Short prompts (21.5s avg)
- **Most Cost-Effective**: Short prompts ($0.0000 avg)

### Use Case Recommendations
- **Short Prompts**: Best for high-volume, time-sensitive tasks where speed is prioritized
- **Normal Prompts**: Optimal balance of accuracy and efficiency for most use cases
- **Long Prompts**: Best for complex tasks requiring high accuracy and detailed analysis