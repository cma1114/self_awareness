#!/usr/bin/env python3
"""
Script to analyze delegate game logs since April 24, 2025.
Extracts key data points and metrics, organizing results by model, dataset, and parameter combinations.
"""

import os
import re
import glob
import datetime
import time
from collections import defaultdict

# Constants
LOG_DIR = "./delegate_game_logs"
OUTPUT_FILE = "./delegate_analysis_results.txt"
CUTOFF_DATE = datetime.datetime(2025, 4, 24).timestamp()  # April 24, 2025

def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename."""
    # Filenames have format like: model_dataset_params_1234567890.log
    # The timestamp is the number at the end before .log
    match = re.search(r'_(\d+)\.log$', filename)
    if match:
        return int(match.group(1))
    return 0

def extract_model_name(filename):
    """Extract model name from the filename."""
    # Get the base filename without path
    base = os.path.basename(filename)
    
    # Extract the part before the dataset
    if "_GPQA_" in base:
        return base.split("_GPQA_")[0]
    elif "_MMLU_" in base:
        return base.split("_MMLU_")[0]
    else:
        # If neither dataset is found, take everything up to the last _ before .log
        match = re.match(r'^(.+)_\d+\.log$', base)
        if match:
            return match.group(1)
        return "unknown"

def extract_dataset(filename):
    """Extract dataset from the filename."""
    if "_GPQA_" in filename:
        return "GPQA"
    elif "_MMLU_" in filename:
        return "MMLU"
    return "unknown"

def extract_parameters(content):
    """Extract parameter combinations from log content."""
    params = {}
    
    # Extract feedback config in detail
    feedback_match = re.search(r'Feedback Config: ({.+?})', content, re.DOTALL)
    if feedback_match:
        feedback_str = feedback_match.group(1)
        # Parse the entire feedback config JSON-like string
        try:
            # Clean up the string to make it valid JSON
            feedback_str = feedback_str.replace("'", '"').replace("True", "true").replace("False", "false")
            import json
            feedback_config = json.loads(feedback_str)
            # Add all feedback params to the parameters dictionary
            for key, value in feedback_config.items():
                params[key] = value
        except:
            # Fallback to regex parsing if JSON parsing fails
            feedback_params = [
                'phase1_subject_feedback', 'phase1_teammate_feedback', 
                'phase2_subject_feedback', 'phase2_teammate_feedback',
                'show_answer_with_correctness'
            ]
            for param in feedback_params:
                if f"'{param}': True" in feedback_str:
                    params[param] = True
                elif f"'{param}': False" in feedback_str:
                    params[param] = False
    
    # Extract Skip Phase 1, Show Phase 1 Summary, and Show Full Phase 1 History
    skip_match = re.search(r'Skip Phase 1: (True|False)', content)
    if skip_match:
        params['Skip Phase 1'] = skip_match.group(1) == "True"
    else:
        # Default to False if not specified
        params['Skip Phase 1'] = False
    
    summary_match = re.search(r'Show Phase 1 Summary: (True|False)', content)
    if summary_match:
        params['Show Phase 1 Summary'] = summary_match.group(1) == "True"
    else:
        params['Show Phase 1 Summary'] = False
    
    history_match = re.search(r'Show Full Phase 1 History: (True|False)', content)
    if history_match:
        params['Show Full Phase 1 History'] = history_match.group(1) == "True"
    else:
        params['Show Full Phase 1 History'] = False
        
    return params

def extract_metrics(content, filename):
    """Extract key metrics from log content."""
    # Get file modification time instead of creation time
    file_mtime = os.path.getmtime(filename)
    file_date = datetime.datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M')
    
    metrics = {
        'filename': os.path.basename(filename),
        'file_date': file_date,
        'n_phase1': 0,
        'n_phase2': 0,
        'subject_p1_acc': 0.0,
        'subject_p2_acc': 0.0,
        'teammate_p1_acc': 0.0,
        'delegation_pct': 0.0,
        'stat_test_line': ''
    }
    
    # Extract N values
    n_match = re.search(r'Parameters: N_phase1=(\d+), N_phase2=(\d+)', content)
    if n_match:
        metrics['n_phase1'] = int(n_match.group(1))
        metrics['n_phase2'] = int(n_match.group(2))
    
    # Extract Phase 1 Accuracies
    p1_subj_match = re.search(r'Subject Phase 1 Accuracy \(SAFN\): ([\d.]+%)', content)
    if p1_subj_match:
        metrics['subject_p1_acc'] = p1_subj_match.group(1)
    
    # Extract Phase 2 Accuracy
    p2_acc_match = re.search(r'Phase 2 Accuracy: ([\d.]+%)', content)
    if p2_acc_match:
        metrics['subject_p2_acc'] = p2_acc_match.group(1)
    
    # Extract Teammate Phase 1 Accuracy
    teammate_acc_match = re.search(r'Teammate Accuracy Phase 1: ([\d.]+%)', content)
    if teammate_acc_match:
        metrics['teammate_p1_acc'] = teammate_acc_match.group(1)
    
    # Extract Delegation Percentage
    delegation_match = re.search(r'Delegation to teammate occurred in (\d+)/(\d+) trials \(([\d.]+)%\)', content)
    if delegation_match:
        metrics['delegation_pct'] = f"{delegation_match.group(3)}%"
    
    # Extract Statistical Test Line
    stat_test_match = re.search(r'(Statistical test \(P2 self vs P1\): z-score = [-\d.]+, p-value = [\d.]+)', content)
    if stat_test_match:
        metrics['stat_test_line'] = stat_test_match.group(1)
    
    return metrics

def param_to_string(params):
    """Convert parameter dictionary to a readable string."""
    param_strs = []
    
    # Feedback config
    feedback_params = [
        'phase1_subject_feedback', 'phase1_teammate_feedback', 
        'phase2_subject_feedback', 'phase2_teammate_feedback',
        'show_answer_with_correctness'
    ]
        
    # Phase settings
    phase_params = [
        'Skip Phase 1', 
        'Show Phase 1 Summary', 
        'Show Full Phase 1 History'
    ]
    for param in phase_params:
        if param in params:
            param_strs.append(f"{param}={params[param]}")
    
    return ", ".join(param_strs)

def main():
    # Get all log files in the directory
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    
    # Filter files by date
    recent_logs = []
    for file in log_files:
        timestamp = extract_timestamp_from_filename(file)
        if timestamp >= CUTOFF_DATE:
            recent_logs.append(file)
    
    print(f"Found {len(recent_logs)} log files since April 24, 2025")
    
    # Organize results by model, dataset, and parameters
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Process each log file
    for log_file in recent_logs:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check if file meets criteria
                if "Statistical test (P2 self vs P1)" not in content:
                    continue
                
                # Extract model, dataset, parameters and metrics
                model = extract_model_name(log_file)
                dataset = extract_dataset(log_file)
                
                # Skip if not GPQA or MMLU
                if dataset not in ["GPQA", "MMLU"]:
                    continue
                
                params = extract_parameters(content)
                metrics = extract_metrics(content, log_file)
                
                # Add parameters to metrics for later display
                metrics['params'] = params
                
                # Filter by N ≥ 50 for both phases
                if metrics['n_phase1'] < 50 or metrics['n_phase2'] < 50:
                    continue
                
                # Add to results - GROUP ONLY BY FEEDBACK CONFIG PARAMETERS
                # Extract only feedback config params for grouping
                feedback_params = {}
                for key in params:
                    if key.startswith('phase') or key == 'show_answer_with_correctness':
                        feedback_params[key] = params[key]
                
                # Create a string representation of just the feedback config for grouping
                feedback_config_str = ", ".join([f"{k}={v}" for k, v in sorted(feedback_params.items())])
                
                # Use ONLY feedback config for grouping
                results[model][dataset][feedback_config_str].append(metrics)
                
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    # Write results to output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write(f"Delegate Game Analysis Results (since April 24, 2025)\n")
        out.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Count total matching files
        total_files = sum(
            len(metrics_list)
            for model_dict in results.values()
            for dataset_dict in model_dict.values()
            for metrics_list in dataset_dict.values()
        )
        out.write(f"Total matching log files: {total_files}\n\n")
        
        # Iterate through results
        for model in sorted(results.keys()):
            out.write(f"=" * 80 + "\n")
            out.write(f"MODEL: {model}\n")
            out.write(f"=" * 80 + "\n\n")
            
            for dataset in sorted(results[model].keys()):
                out.write(f"{'-' * 40}\n")
                out.write(f"DATASET: {dataset}\n")
                out.write(f"{'-' * 40}\n\n")
                
                for param_str in sorted(results[model][dataset].keys()):
                    out.write(f"FEEDBACK CONFIG: {param_str}\n")
                    
                    out.write(f"{'-' * 60}\n")
                    
                    # Write metrics for each file
                    metrics_list = results[model][dataset][param_str]
                    for i, metrics in enumerate(metrics_list, 1):
                        out.write(f"File {i}: {metrics['filename']} (Modified: {metrics['file_date']})\n")
                        
                        # Extract and display the phase parameters for each file
                        phase_params = {k: v for k, v in metrics.get('params', {}).items() 
                                       if k in ['Skip Phase 1', 'Show Phase 1 Summary', 'Show Full Phase 1 History', 'Use Phase 2 Data']}
                        
                        if phase_params:
                            out.write(f"  PHASE PARAMETERS: {', '.join([f'{k}={v}' for k, v in sorted(phase_params.items())])}\n")
                        
                        out.write(f"  N Phase 1: {metrics['n_phase1']}, N Phase 2: {metrics['n_phase2']}\n")
                        out.write(f"  Subject P1 Accuracy: {metrics['subject_p1_acc']}, Teammate P1 Accuracy: {metrics['teammate_p1_acc']}\n")
                        out.write(f"  Subject P2 Accuracy: {metrics['subject_p2_acc']}, Delegation Rate: {metrics['delegation_pct']}\n")
                        if metrics['stat_test_line']:
                            out.write(f"  {metrics['stat_test_line']}\n")
                        out.write("\n")
                    
                    out.write("\n")
                
                out.write("\n")
    
    print(f"Analysis complete. Results written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()