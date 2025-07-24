

import pandas as pd
import glob
from scipy.stats import ttest_ind
from pathlib import Path

def load_human_data(human_csv_path):
    """Load and preprocess human study results"""
    try:
        human_data = pd.read_csv(human_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        human_data = pd.read_csv(human_csv_path, encoding='utf-8-sig')
    
    human_data.columns = human_data.columns.str.strip()
    
    column_mapping = {
        'data quality': 'quality_condition',
        'completion_time_sec': 'completion_time_sec',
        'accuracy': 'is_correct'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in human_data.columns:
            human_data = human_data.rename(columns={old_col: new_col})
            
    if 'is_correct' not in human_data.columns:
        raise ValueError("Column 'is_correct' not found in human data.")

    return human_data

def load_llm_data_for_model(llm_results_dir, model_name_pattern):
    """Load and aggregate LLM results for a specific model pattern"""
    llm_files = glob.glob(f"{llm_results_dir}/{model_name_pattern}*.csv")
    if not llm_files:
        return None
    
    all_llm_data = []
    for file_path in llm_files:
        try:
            df = pd.read_csv(file_path)
            if 'is_correct' in df.columns:
                df['is_correct'] = df['is_correct'].map({
                    'True': 1, 'False': 0, True: 1, False: 0, 1: 1, 0: 0
                })
            all_llm_data.append(df)
        except Exception:
            continue
            
    if not all_llm_data:
        return None
        
    return pd.concat(all_llm_data, ignore_index=True)

def perform_ttest_analysis():
    human_data_path = 'experiments/human_study/Participants_results.csv'
    llm_data_dir = 'experiments/llm_evaluation/performance_logs/short_all_results_fair_benchmark'
    
    human_data = load_human_data(human_data_path)
    
    models = [
        "claude-sonnet-4-20250514",
        "gpt-4o-mini-2024-07-18",
        "deepseek-reasoner",
        "o4-mini-2025-04-16",
        "claude-3-5-haiku-latest",
        "deepseek-chat"
    ]
    
    results = []
    
    for model_name in models:
        model_data = load_llm_data_for_model(llm_data_dir, model_name)
        
        if model_data is None:
            print(f"No data found for model: {model_name}")
            continue

        # Align tasks
        common_tasks = set(human_data['task_id']).intersection(set(model_data['task_id']))
        human_aligned = human_data[human_data['task_id'].isin(common_tasks)]
        model_aligned = model_data[model_data['task_id'].isin(common_tasks)]

        if human_aligned.empty or model_aligned.empty:
            continue

        # T-test for accuracy
        acc_ttest = ttest_ind(model_aligned['is_correct'], human_aligned['is_correct'], equal_var=False)
        
        # T-test for completion time
        time_ttest = ttest_ind(model_aligned['completion_time_sec'], human_aligned['completion_time_sec'], equal_var=False)
        
        results.append({
            "Model": model_name,
            "Accuracy T-statistic": f"{acc_ttest.statistic:.4f}",
            "Accuracy p-value": f"{acc_ttest.pvalue:.4f}",
            "Time T-statistic": f"{time_ttest.statistic:.4f}",
            "Time p-value": f"{time_ttest.pvalue:.4f}"
        })
        
    return results

def format_results_to_markdown(results):
    if not results:
        return "No results to format."
        
    md = "## 8. T-Test Analysis***REMOVED***n***REMOVED***n"
    md += "This section provides the results of independent two-sample T-tests comparing each LLM's performance against the human baseline. The tests were conducted on both accuracy and completion time.***REMOVED***n***REMOVED***n"
    md += "| Model | Accuracy T-statistic | Accuracy p-value | Time T-statistic | Time p-value |***REMOVED***n"
    md += "|---|---|---|---|---|***REMOVED***n"
    
    for res in results:
        md += f"| {res['Model']} | {res['Accuracy T-statistic']} | {res['Accuracy p-value']} | {res['Time T-statistic']} | {res['Time p-value']} |***REMOVED***n"
        
    md += "***REMOVED***n**Interpretation:*****REMOVED***n"
    md += "- **T-statistic**: A positive value for accuracy indicates the model's mean accuracy is higher than the human mean. A negative value for time indicates the model is faster than humans.***REMOVED***n"
    md += "- **p-value**: A p-value less than 0.05 indicates that the observed difference is statistically significant.***REMOVED***n"
    
    return md

if __name__ == "__main__":
    analysis_results = perform_ttest_analysis()
    markdown_output = format_results_to_markdown(analysis_results)
    
    findings_path = Path('Findings.md')
    with open(findings_path, 'a') as f:
        f.write("***REMOVED***n" + markdown_output)
        
    print("T-Test analysis complete and appended to Findings.md")
    print(markdown_output)

