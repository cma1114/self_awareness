import json
import os
import numpy as np
from collections import defaultdict

# Assuming load_and_format_datasets.py and its load_and_format_dataset function are available
from load_and_format_datasets import load_and_format_dataset

def analyze_model_performance_with_features(gpqa_questions_processed, results_dir):
    """
    Analyzes model correctness against question difficulty, domain, and length.
    Assumes gpqa_questions_processed items have 'id', 'difficulty_score', 
    'high_level_domain', and 'Question' (for text).
    """
    feature_lookup = {
        item['id']: {
            'difficulty': item['difficulty_score'],
            'domain': item['high_level_domain'],
            'question_text': item['question'] 
        } 
        for item in gpqa_questions_processed if item.get('id') # Ensure ID exists
    }

    print(f"Analyzing files in directory: {results_dir}\n")

    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith(".json"):
            file_path = os.path.join(results_dir, filename)
            print(f"--- Processing File: {filename} ---")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Error reading/parsing {filename}: {e}. Skipping.", file=sys.stderr)
                continue
            
            model_results = data.get("results", {})
            if not isinstance(model_results, dict):
                print(f"  Warning: 'results' field not a dict in {filename}. Skipping.", file=sys.stderr)
                continue

            difficulties_correct, difficulties_incorrect = [], []
            lengths_correct, lengths_incorrect = [], []
            domain_correct_counts = defaultdict(int)
            domain_total_counts = defaultdict(int)
            
            for q_id, result_info in model_results.items():
                features = feature_lookup.get(q_id)
                if not features: continue # Skip if question not in our processed GPQA list
                
                question_text = features['question_text']
                question_length = len(question_text) if isinstance(question_text, str) else 0
                is_correct = result_info.get("is_correct")
                domain = features['domain']
                
                if domain:
                    domain_total_counts[domain] += 1
                    if is_correct is True:
                        domain_correct_counts[domain] += 1
                
                if features.get('difficulty') is not None: # Only use if difficulty score exists
                    if is_correct is True:
                        difficulties_correct.append(features['difficulty'])
                        lengths_correct.append(question_length)
                    elif is_correct is False:
                        difficulties_incorrect.append(features['difficulty'])
                        lengths_incorrect.append(question_length)
            
            avg_diff_correct = np.mean(difficulties_correct) if difficulties_correct else np.nan
            avg_diff_incorrect = np.mean(difficulties_incorrect) if difficulties_incorrect else np.nan
            print("  Difficulty Analysis:")
            print(f"    Avg difficulty for CORRECT answers: {avg_diff_correct:.2f} (N={len(difficulties_correct)})")
            print(f"    Avg difficulty for INCORRECT answers: {avg_diff_incorrect:.2f} (N={len(difficulties_incorrect)})")

            avg_len_correct = np.mean(lengths_correct) if lengths_correct else np.nan
            avg_len_incorrect = np.mean(lengths_incorrect) if lengths_incorrect else np.nan
            print("  Question Length Analysis:")
            print(f"    Avg length for CORRECT answers: {avg_len_correct:.1f} (N={len(lengths_correct)})")
            print(f"    Avg length for INCORRECT answers: {avg_len_incorrect:.1f} (N={len(lengths_incorrect)})")

            print("  Correctness % per High-level Domain:")
            for domain in sorted(domain_total_counts.keys()):
                total, correct = domain_total_counts[domain], domain_correct_counts[domain]
                correct_percent = (correct / total) * 100 if total > 0 else 0
                print(f"    {domain}: {correct_percent:.1f}% correct (N_correct={correct}, N_total={total})")
            
            print("-" * 30 + "\n")

if __name__ == "__main__":
    print("Loading GPQA dataset...")
    # load_and_format_dataset("GPQA") should return items with:
    # "id", "difficulty_score", "high_level_domain", "Question" (raw text)
    gpqa_data = load_and_format_dataset("GPQA") 

    if not gpqa_data:
        print("Failed to load GPQA data. Exiting.", file=sys.stderr)
        exit()
    
    print(f"Successfully loaded {len(gpqa_data)} GPQA questions.")

    completed_results_directory = "./completed_results_gpqa/" 
    if not os.path.isdir(completed_results_directory):
        print(f"Error: Directory not found: {completed_results_directory}", file=sys.stderr)
        exit()

    analyze_model_performance_with_features(gpqa_data, completed_results_directory)