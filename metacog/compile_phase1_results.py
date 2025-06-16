#!/usr/bin/env python3
"""
Script to compile phase1 results from each model on specified datasets.
Takes game_data.json files from April 24, 2025 and later, and aggregates results per model.
Processes files from:
1. delegate_game_logs - Files with _game_data.json
2. pass_game_logs - Files with _game_data.json or _phase1_data.json
3. capabilities_test_logs - Files with _test_data.json
"""

import os
import json
import glob
import datetime
import time
import re
import sys
from collections import defaultdict

# Constants
LOG_DIRS = {
    "delegate_game_logs": "./delegate_game_logs",
    "pass_game_logs": "./pass_game_logs",
    "capabilities_test_logs": "./capabilities_test_logs",
    "capabilities_results_sa": "./capabilities_results_sa",
}

CUTOFF_DATE = datetime.datetime(2025, 4, 24).timestamp()  # April 24, 2025

id_prefix_map = {
    "GPQA": "gpqa_",
    "MMLU": "mmlu_",
    "TruthfulQA": "tqa_",
    "SimpleQA": "sqa_",
    "GPSA": "gpsa_",
    "SimpleMC": "smc_"
}

def get_file_timestamp(file_path):
    """Get the file's modification timestamp from the OS."""
    return os.path.getmtime(file_path)

def extract_model_name_from_delegate_file(filename, dataset="GPQA"):
    """Extract model name from a delegate_game_logs filename."""
    base = os.path.basename(filename)
    
    # Look for the dataset name in the filename
    dataset_marker = f"_{dataset}_"
    if dataset_marker in base:
        return base.split(dataset_marker)[0]
    else:
        return "unknown"

def extract_model_name_from_pass_file(filename):
    """Extract model name from a pass_game_logs filename."""
    base = os.path.basename(filename)
    
    # Handle different filename patterns
    parts = base.split("_")
    if base.startswith("aop_game_"):
        return parts[2]
    elif base.startswith("aop_"):
        return parts[1]
    
    return "unknown"

def extract_model_name_from_capabilities_file(filename):
    """Extract model name from a capabilities_test_logs filename."""
    base = os.path.basename(filename)
    
    # Format: MODEL_DATASET_..._test_data.json
    parts = base.split("_")
    if len(parts) >= 2:
        if parts[1] in id_prefix_map.keys():
            return parts[0]
    
    return "unknown"

def is_dataset_file(file_path, dataset, file_content=None):
    """Determine if a file contains questions from the specified dataset."""
    
    # Check filename pattern
    dataset_marker = f"_{dataset}_"
    if dataset_marker in file_path:  # Primary check, esp. for delegate & capabilities
        return True

    if False:#file_content:
        # For delegate_game files
        if "phase1_results" in file_content and isinstance(file_content["phase1_results"], list):
            if file_content["phase1_results"]:  # Check if list is not empty
                sample_item_result = file_content["phase1_results"][0]
                if isinstance(sample_item_result, dict) and \
                   "question_id" in sample_item_result:
                    return True
        
        # For pass_game files 
        if "phase1_results" in file_content and isinstance(file_content["phase1_results"], dict):
            if file_content["phase1_results"]:  # Check if dict is not empty
                return True
        
        # For capabilities_test files 
        if "results" in file_content and isinstance(file_content["results"], dict):
            if file_content["results"]:  # Check if dict is not empty
                return True
    return False

def process_delegate_game_file(file_path, dataset="GPQA"):
    """Process a delegate_game_logs file and extract phase1 results."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Skip if not a file for the specified dataset
        if not is_dataset_file(file_path, dataset, data):
            return None, None, []
        
        # Extract model name
        model_name = extract_model_name_from_delegate_file(file_path, dataset)
        
        # Process phase1 results
        results = []
        
        # In delegate_game_logs, phase1_results is a list
        if "phase1_results" in data and isinstance(data["phase1_results"], list):
            for result in data["phase1_results"]:
                if "question_id" in result:
                    # Create result object
                    result_obj = {
                        "question_id": result["question_id"],
                        "question_text": result.get("question_text", ""),
                        "options": result.get("options", {}),
                        "correct_answer_label": result.get("correct_answer_label", ""),
                        "subject_answer": result.get("subject_answer", ""),
                        "subject_correct": result.get("subject_correct", None),
                        "probs": result.get("probs", None)
                    }
                    results.append(result_obj)
        
        return model_name, file_path, results
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, []

def process_pass_game_file(file_path, dataset="GPQA"):
    """Process a pass_game_logs file and extract phase1 results."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Skip if not a file for the specified dataset
        if not is_dataset_file(file_path, dataset, data):
            return None, None, []
        
        # Extract model name
        model_name = extract_model_name_from_pass_file(file_path)
        
        # Process phase1 results
        results = []
                
        # In pass_game_logs, phase1_results is a dictionary
        if "phase1_results" in data and isinstance(data["phase1_results"], dict):
            for question_id, result in data["phase1_results"].items():
                # Create result object
                result_obj = {
                    "question_id": question_id,
                    "question_text": result.get("question", {}).get("question", ""),
                    "options": result.get("question", {}).get("options", {}),
                    "correct_answer_label": result.get("question", {}).get("correct_answer", ""),
                    "subject_answer": result.get("subject_answer", ""),
                    "subject_correct": result.get("is_correct"),
                    "probs": result.get("probs", None)
                }
                results.append(result_obj)
        
        return model_name, file_path, results
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, []

def process_capabilities_test_file(file_path, dataset="GPQA"):
    """Process a capabilities_test_logs file and extract results."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Skip if not a file for the specified dataset
        if not is_dataset_file(file_path, dataset, data):
            return None, None, []
        
        # Extract model name
        model_name = extract_model_name_from_capabilities_file(file_path)
        
        # Process results
        results = []
        
        # In capabilities_test_logs, it's called "results" and is a dictionary
        if "results" in data and isinstance(data["results"], dict):
            for question_id, result in data["results"].items():
                # Create result object
                result_obj = {
                    "question_id": question_id,
                    "question_text": result.get("question", {}).get("question", ""),
                    "options": result.get("question", {}).get("options", {}),
                    "correct_answer_label": result.get("question", {}).get("correct_answer", ""),
                    "subject_answer": result.get("subject_answer", ""),
                    "subject_correct": result.get("is_correct"),
                    "probs": result.get("probs", None),
                    "judgments": result.get("judgments", None),
                    "evaluation_method": result.get("evaluation_method", None)
                }
                results.append(result_obj)
        
        return model_name, file_path, results
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, []

def process_all_files(dataset="GPQA", targ_model=None):
    """Process all game data files and compile results per model for the specified dataset."""
    # Create output directory if it doesn't exist
    output_dir = f"./compiled_results_{id_prefix_map[dataset].replace('_', '')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all relevant files
    delegate_files = glob.glob(os.path.join(LOG_DIRS["delegate_game_logs"], "*_game_data.json")) if "delegate_game_logs" in LOG_DIRS else []
    pass_files = glob.glob(os.path.join(LOG_DIRS["pass_game_logs"], "*_game_data.json")) + glob.glob(os.path.join(LOG_DIRS["pass_game_logs"], "*_phase1_data.json")) if "pass_game_logs" in LOG_DIRS else []
    capabilities_files = glob.glob(os.path.join(LOG_DIRS["capabilities_test_logs"], "*_test_data.json")) if "capabilities_test_logs" in LOG_DIRS else []
    if "capabilities_results_sa" in LOG_DIRS: capabilities_files += glob.glob(os.path.join(LOG_DIRS["capabilities_results_sa"], "*_test_data_evaluated.json"))
    
    # Filter files by date
    recent_delegate_files = [f for f in delegate_files if get_file_timestamp(f) >= CUTOFF_DATE]
    recent_pass_files = [f for f in pass_files if get_file_timestamp(f) >= CUTOFF_DATE]
    recent_capabilities_files = [f for f in capabilities_files if get_file_timestamp(f) >= CUTOFF_DATE]
    
    print(f"Found {len(recent_delegate_files)} delegate game files since April 24, 2025")
    print(f"Found {len(recent_pass_files)} pass game files since April 24, 2025")
    print(f"Found {len(recent_capabilities_files)} capabilities test files since April 24, 2025")
    print(f"Filtering for {dataset} dataset")
    
    # Group results by model
    model_results = defaultdict(lambda: {"files": [], "results": []})
    
    # Process delegate game files
    for file_path in recent_delegate_files:
        model_name, file_path, results = process_delegate_game_file(file_path, dataset)
        if targ_model and model_name != targ_model:
            continue
        if model_name and results:
            model_results[model_name]["files"].append(file_path)
            model_results[model_name]["results"].extend(results)
    
    # Process pass game files
    for file_path in recent_pass_files:
        model_name, file_path, results = process_pass_game_file(file_path, dataset)
        if targ_model and model_name != targ_model:
            continue
        if model_name and results:
            model_results[model_name]["files"].append(file_path)
            model_results[model_name]["results"].extend(results)
    
    # Process capabilities test files
    for file_path in recent_capabilities_files:
        model_name, file_path, results = process_capabilities_test_file(file_path, dataset)
        if targ_model and model_name != targ_model:
            continue
        if model_name and results:
            model_results[model_name]["files"].append(file_path)
            model_results[model_name]["results"].extend(results)
    
    print(f"Found data for {len(model_results)} models: {', '.join(model_results.keys())}")
    
    # Compile results for each model
    for model, data in model_results.items():
        compile_model_results(model, data["files"], data["results"], output_dir, dataset)

def compile_model_results(model, file_paths, all_results, output_dir, dataset="GPQA"):
    """Compile results for a specific model."""
    print(f"Processing {len(file_paths)} files for model {model} for {dataset} dataset...")
    
    # Group results by question ID
    question_responses = {}
    
    # Process each result
    for result in all_results:
        q_id = result["question_id"]
        
        # Skip if subject_correct is None
        if result["subject_answer"] is None or result["subject_answer"] == "": # or result["subject_correct"] is None or 
            continue
        if result["subject_correct"] is None: result["subject_correct"] = False
        
        # If this is the first time seeing this question, initialize its entry
        if q_id not in question_responses:
            question_responses[q_id] = {
                "responses": [],
                "fields": {
                    "question": result["question_text"],
                    "options": result["options"],
                    "correct_answer_label": result["correct_answer_label"],
                    "subject_answer": result["subject_answer"],
                    "is_correct": result["subject_correct"],
                    "probs": result["probs"],
                    "judgments": result.get("judgments", None),
                    "evaluation_method": result.get("evaluation_method", None)
                }
            }
        
        # Add this response
        question_responses[q_id]["responses"].append({
            "is_correct": result["subject_correct"],
            "probs": result["probs"]
        })
    
    # Filter questions that have consistent responses
    compiled_results = {}
    correct_count = 0
    total_count = 0
    
    for q_id, data in question_responses.items():
        responses = data["responses"]
        
        # Check if all responses have consistent correctness
        is_consistent = True
        for r in responses[1:]:
            if r["is_correct"] != responses[0]["is_correct"]:
                is_consistent = False
                break
        
        if is_consistent:
            # Use the first response's data, but store all probs
            compiled_results[q_id] = {
                "question": data["fields"]["question"],
                "options": data["fields"]["options"],
                "correct_answer_label": data["fields"]["correct_answer_label"],
                "subject_answer": data["fields"]["subject_answer"],
                "is_correct": responses[0]["is_correct"],
                "probs": data["fields"]["probs"],  # Store probs from first entry
                "judgments": data["fields"]["judgments"],
                "evaluation_method": data["fields"]["evaluation_method"]
            }
            
            # Count correct responses for accuracy calculation
            if responses[0]["is_correct"]:
                correct_count += 1
            total_count += 1
    
    # Calculate overall accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Create output data structure
    output_data = {
        "model": model,
        "dataset": dataset,
        "total_questions": total_count,
        "correct_questions": correct_count,
        "accuracy": accuracy,
        "compilation_timestamp": time.time(),
        "source_files": len(file_paths),
        "questions_analyzed": len(question_responses),
        "questions_included": total_count,
        "results": compiled_results
    }
    
    # Write to output file
    output_file = os.path.join(output_dir, f"{model}_phase1_compiled.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Compiled {total_count} consistent questions for {model} (accuracy: {accuracy:.2%})")
    print(f"Results written to {output_file}")

def main():
    """Main function to compile phase1 results."""
    print("Starting compilation of phase1 results...")
    start_time = time.time()
    
    # Hard-coded dataset - change this value to compile different datasets
    dataset = "GPSA"#"GPQA"#"SimpleQA"#"SimpleMC"# 
    
    process_all_files(dataset, targ_model="claude-3-sonnet-20240229")
    
    elapsed_time = time.time() - start_time
    print(f"Compilation completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()