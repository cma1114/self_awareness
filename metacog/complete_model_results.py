#!/usr/bin/env python3
"""
Script to complete a model's compiled results with missing dataset questions.

This script:
1. Takes a model name and dataset name as input
2. Loads the model's existing compiled results from ./compiled_results_{dataset}/
3. Loads the full dataset using load_and_format_dataset
4. Identifies questions in the dataset that are missing from the compiled results
5. Runs those questions through CapabilitiesTest (truncating if > MAX_QUESTIONS)
6. Extracts results and adds them to the compiled results file
7. Updates totals and accuracy statistics
8. Saves to ./completed_results_{dataset}/
"""

import os
import sys
import json
import time
from load_and_format_datasets import load_and_format_dataset
from capabilities_test import CapabilitiesTest

# Constants
MAX_QUESTIONS = 500  # Maximum number of questions to run through the capabilities test
OUTPUT_DIR = "./capabilities_test_logs"

def load_compiled_results(model_name, dataset="GPQA"):
    """Load the model's compiled results file."""
    compiled_dir = f"./compiled_results_{dataset.lower()}"
    result_file = os.path.join(compiled_dir, f"{model_name.replace("/","-")}_phase1_compiled.json")
    
    if not os.path.exists(result_file):
        print(f"No compiled results file found for {model_name} for dataset {dataset}. Creating a new one.")
        return {
            "model": model_name,
            "dataset": dataset,
            "total_questions": 0,
            "correct_questions": 0,
            "accuracy": 0,
            "compilation_timestamp": time.time(),
            "source_files": 0,
            "questions_analyzed": 0,
            "questions_included": 0,
            "results": {}
        }
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            compiled_data = json.load(f)
        print(f"Loaded compiled results for {model_name} with {compiled_data.get('total_questions', 0)} questions.")
        return compiled_data
    except Exception as e:
        print(f"Error loading compiled results file: {e}")
        sys.exit(1)

def identify_missing_questions(compiled_data, full_dataset):
    """Identify questions from the full dataset that are missing from compiled results."""
    # Get IDs of questions already in compiled results
    existing_question_ids = set(compiled_data["results"].keys())
    
    # Find questions in full dataset that aren't in compiled results
    missing_questions = []
    for question in full_dataset:
        q_id = question["id"]
        if q_id not in existing_question_ids:
            missing_questions.append(question)
    
    print(f"Found {len(missing_questions)} questions missing from compiled results.")
    return missing_questions

def run_capabilities_test(model_name, questions, dataset="GPQA"):
    """Run missing questions through CapabilitiesTest."""
    # Truncate questions if there are more than MAX_QUESTIONS
    if len(questions) > MAX_QUESTIONS:
        print(f"Truncating {len(questions)} questions to {MAX_QUESTIONS} to avoid excessive API usage.")
        questions = questions[:MAX_QUESTIONS]
    
    print(f"Running CapabilitiesTest for {model_name} with {len(questions)} {dataset} questions...")
    
    subject_id = f"{model_name.replace("/","-")}_{dataset}_completion"
    
    # Initialize and run CapabilitiesTest
    test = CapabilitiesTest(
        subject_id=subject_id,
        subject_name=model_name,
        questions=questions,
        n_questions=len(questions),
        is_human_player=False
    )
    
    success, results_file = test.run_capabilities_measurement()
    
    if not success:
        print(f"Error: CapabilitiesTest failed for {model_name}")
        return None
    
    print(f"CapabilitiesTest completed successfully. Results saved to {results_file}")
    return results_file

def extract_results(results_file):
    """Extract results from CapabilitiesTest output file."""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get("results", {})
        # Process the results to ensure we capture probability data
        processed_results = {}
        
        for q_id, result in results.items():
            processed_results[q_id] = {
                "question": result.get("question", {}),
                "subject_answer": result.get("subject_answer", ""),
                "is_correct": result.get("is_correct", None),
                "probs": result.get("probs", None)  # Extract probability data
            }
            
        print(f"Extracted {len(processed_results)} results from capabilities test output.")
        return processed_results
    except Exception as e:
        print(f"Error extracting results from {results_file}: {e}")
        return {}

def update_compiled_results(compiled_data, new_results):
    """Update compiled results with new results from CapabilitiesTest."""
    # Add new results
    new_count = 0
    new_correct = 0
    
    for q_id, result in new_results.items():
        if q_id not in compiled_data["results"]:
            # Extract required fields
            compiled_data["results"][q_id] = {
                "question": result.get("question", {}).get("question", ""),
                "options": result.get("question", {}).get("options", {}),
                "correct_answer_label": result.get("question", {}).get("correct_answer", ""),
                "subject_answer": result.get("subject_answer", ""),
                "is_correct": result.get("is_correct", None),
                "probs": result.get("probs", None)  # Store probability data if available
            }
            
            # Count only if is_correct is not None
            if result.get("is_correct") is not None:
                new_count += 1
                if result.get("is_correct"):
                    new_correct += 1
    
    # Recalculate statistics based on all results
    correct_count = 0
    total_count = 0
    
    # Count all results with valid is_correct values
    for q_id, result in compiled_data["results"].items():
        if result.get("is_correct") is not None:
            total_count += 1
            if result.get("is_correct"):
                correct_count += 1
    
    # Update the statistics
    compiled_data["total_questions"] = total_count
    compiled_data["correct_questions"] = correct_count
    
    if compiled_data["total_questions"] > 0:
        compiled_data["accuracy"] = compiled_data["correct_questions"] / compiled_data["total_questions"]
    
    compiled_data["compilation_timestamp"] = time.time()
    compiled_data["questions_included"] = len(compiled_data["results"])
    
    print(f"Added {new_count} new questions with {new_correct} correct answers.")
    print(f"Total results: {total_count} questions, {compiled_data['accuracy']:.2%} accuracy")
    
    return compiled_data

def save_compiled_results(compiled_data, dataset="GPQA"):
    """Save updated compiled results to file."""
    # Get the dataset from the compiled_data if available
    dataset = compiled_data.get("dataset", dataset)
    
    # Create output directory (compiled is replaced with completed)
    output_dir = f"./completed_results_{dataset.lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{compiled_data['model']}_phase1_completed.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(compiled_data, f, indent=2)
    
    print(f"Updated completed results saved to {output_file}")
    return output_file

def complete_model_results(model_name, dataset="GPQA"):
    """Complete a model's results by running missing dataset questions."""
    print(f"Starting completion process for model: {model_name} on dataset: {dataset}")
    
    # Step 1: Load compiled results
    compiled_data = load_compiled_results(model_name, dataset)
    
    # Step 2: Load full dataset
    print(f"Loading full {dataset} dataset...")
    full_dataset = load_and_format_dataset(dataset, None)
    print(f"Loaded {len(full_dataset)} questions from {dataset} dataset.")
    
    # Step 3: Identify missing questions
    missing_questions = identify_missing_questions(compiled_data, full_dataset)
    
    # If no missing questions, still save the complete file
    if not missing_questions:
        print("No missing questions found. Compiled results are already complete.")
        # Save the complete results file anyway
        save_compiled_results(compiled_data, dataset)
        print(f"Final statistics: {compiled_data['total_questions']} total questions, {compiled_data['accuracy']:.2%} accuracy")
        return
    
    # Step 4: Run missing questions through CapabilitiesTest (truncating if needed)
    results_file = run_capabilities_test(model_name, missing_questions, dataset)
    if not results_file:
        print("Failed to run capabilities test. Aborting.")
        return
    
    # Step 5: Extract results
    new_results = extract_results(results_file)
    
    # Step 6: Update compiled results
    updated_data = update_compiled_results(compiled_data, new_results)
    
    # Step 7: Save updated compiled results to the completed directory
    save_compiled_results(updated_data, dataset)
    
    print(f"Completion process finished for {model_name} on {dataset}.")
    print(f"Final statistics: {updated_data['total_questions']} total questions, {updated_data['accuracy']:.2%} accuracy")

def main():
    # Hard-coded model and dataset names
    model_name = "claude-3-sonnet-20240229"#"claude-3-haiku-20240307"#'gpt-4.1-2025-04-14'#'claude-sonnet-4-20250514'#"deepseek-chat"#"gemini-2.5-flash-preview-04-17"#"gpt-4o-2024-08-06"#"meta-llama/Meta-Llama-3.1-405B-Instruct"#"grok-3-latest"#"gemini-1.5-pro"#"gpt-3.5-turbo-0125"#"gpt-4-turbo-2024-04-09"#"claude-3-opus-20240229"#"claude-3-5-sonnet-20241022"#"claude-3-7-sonnet-20250219"
    dataset = "GPQA"  # Change this to run on different datasets: GPQA, SimpleQA, MMLU, TruthfulQA
    
    start_time = time.time()
    complete_model_results(model_name, dataset)
    elapsed_time = time.time() - start_time
    
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()