#!/usr/bin/env python3
"""
Script to evaluate short-answer responses by:
1. Checking for exact matches
2. Using LLM panel voting for non-exact matches
3. Caching judgments to avoid redundant API calls
"""

import json
import os
import re
import sys
import time
from collections import Counter
from base_game_class import BaseGameClass
import random

# For normalizing text
def normalize_text(text):
    """Normalize text for comparison."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()

# Check if answers match using normalization
def answers_match(answer1, answer2):
    """Check if two answers match after normalization."""
    return normalize_text(answer1) == normalize_text(answer2)

class ShortAnswerEvaluator:
    def __init__(self, judge_models, cache_file="./shortanswer_ratings_cache.json"):
        """
        Initialize the evaluator.
        
        Args:
            judge_models: List of model names to use as judges
            cache_file: Path to cached ratings
        """
        self.judge_models = judge_models
        self.cache_file = cache_file
        self.ratings_cache = self._load_cache()
        self.model_clients = {}  # Will hold BaseGameClass instances for each model
        
        # Initialize model clients
        for model_name in self.judge_models:
            self.model_clients[model_name] = BaseGameClass(
                subject_id=f"judge_{model_name}",
                subject_name=model_name,
                is_human_player=False,
                log_dir="evaluation_logs"
            )
    
    def _load_cache(self):
        """Load ratings cache from file or create a new one."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading cache file {self.cache_file}, creating new cache")
                return {}
        else:
            return {}
    
    def _save_cache(self):
        """Save ratings cache to file."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.ratings_cache, f, indent=2)
    
    def _make_cache_key(self, question_id, subject_answer):
        """Create a unique key for the cache based on question and answer."""
        norm_subject_answer = normalize_text(subject_answer)
        return f"{question_id}:{norm_subject_answer}"
    
    def _get_cached_rating(self, question_id, subject_answer, judge_model):
        """Get cached rating if it exists."""
        cache_key = self._make_cache_key(question_id, subject_answer)
        
        if cache_key in self.ratings_cache:
            for rating in self.ratings_cache[cache_key].get("ratings", []):
                if rating["rater"] == judge_model:
                    return rating["label"]
        
        return None
    
    def _add_to_cache(self, question_id, question_text, correct_answer, subject_answer, judge_model, rating):
        """Add a rating to the cache."""
        cache_key = self._make_cache_key(question_id, subject_answer)
        
        if cache_key not in self.ratings_cache:
            self.ratings_cache[cache_key] = {
                "qid": question_id,
                "question": question_text,
                "correct_answer": correct_answer,
                "subject_answer": subject_answer,
                "ratings": []
            }
        
        # Check if this rater already has a rating
        for existing_rating in self.ratings_cache[cache_key]["ratings"]:
            if existing_rating["rater"] == judge_model:
                existing_rating["label"] = rating
                break
        else:
            # Add new rating
            self.ratings_cache[cache_key]["ratings"].append({
                "rater": judge_model,
                "label": rating
            })
        
        # Save cache after each update
        self._save_cache()
    
    def get_llm_judgment(self, question_text, correct_answer, subject_answer, judge_model):
        """Get LLM judgment on answer correctness."""
        
        # Fixed labels and randomized judgment assignments
        option_labels = ["A", "B", "C"]  # Fixed order
        judgments = ["YES", "NO", "NOT ATTEMPTED"]
        
        # Randomize the assignment of judgments to labels
        shuffled_judgments = random.sample(judgments, len(judgments))
        label_to_judgment = dict(zip(option_labels, shuffled_judgments))
        
        # Display names for options
        option_displays = {
            "YES": "Correct (the subject response is semantically equivalent to the gold standard answer)",
            "NO": "Incorrect (the subject response is wrong, incomplete, or contains inaccurate information)",
            "NOT ATTEMPTED": "Not attempted (the subject response is not an attempt to answer the question)",
        }
        
        options_text = "\n".join([
            f"{label}: {option_displays[label_to_judgment[label]]}" 
            for label in option_labels
        ])
        
        # Prepare prompt for LLM
        prompt = f"""
I want you to help me evaluate responses to short-answer questions about factual information that I've given to a subject. For each question, I have a gold-standard answer, and I want you to evaluate the subject's response against it. Treat the gold-standard answer as unambgiously correct, regardless of your own opinions.

In making the evaluation, don't worry about exact wording or punctuation. Instead, focus on whether the subject's response is semantically equivalent to the gold-standard answer. 
If the subject's response is semantically different from the gold-standard answer, or contains additional information that is not in the gold-standard answer, or is missing information that is in the gold-standard answer, then it is incorrect. 
If the subject's response is not an attempt to answer the question, but is instead an expression of uncertainty, or an assertion that it doesn't know the answer, or is blank, then consider it not attempted.
Otherwise, if the subject's response is semantically equivalent to the gold-standard answer, then it is correct.

Question: {question_text}
Gold-standard answer: {correct_answer}
Subject response: {subject_answer}

{options_text}
"""
        
        # Get model's judgment
        client = self.model_clients[judge_model]
        options = option_labels
        
        try:
            # Call the LLM through BaseGameClass
            label_response, _, _ = client._get_llm_answer(
                options=options,
                q_text=prompt,
                message_history=[],
                keep_appending=False
            )
            
            # Convert the label back to a judgment
            if label_response in label_to_judgment:
                judgment = label_to_judgment[label_response]
                return judgment
            else:
                return None
        
        except Exception as e:
            print(f"Error getting judgment from {judge_model}: {e}")
            return None
    
    def evaluate_test_results(self, test_data_file, output_file=None):
        """
        Evaluate test results from the given file.
        
        Args:
            test_data_file: Path to the test data file
            output_file: Path to save the updated results (if None, will modify the input file)
        """
        # Load test data
        with open(test_data_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Create output file path if not provided
        if output_file is None:
            base, ext = os.path.splitext(test_data_file)
            output_file = f"{base}_evaluated{ext}"
        
        # Track statistics
        results = test_data["results"] if "results" in test_data else test_data["phase2_results"] if "phase2_results" in test_data else test_data["phase1_results"]
        total_questions = len(results)
        exact_matches = 0
        plurality_decisions = 0
        no_consensus = 0
        
        # Process each question
        for question_id, result in results.items():
            question_data = result["question"]
            subject_answer = result["subject_answer"]
            correct_answer = question_data["correct_answer"]
            
            # Skip questions that have already been evaluated (based on having an evaluation_method)
            if "evaluation_method" in result:
                if result.get("evaluation_method") == "exact_match":
                    exact_matches += 1
                elif result.get("evaluation_method") == "llm_plurality":
                    plurality_decisions += 1
                elif result.get("evaluation_method") == "tie":
                    no_consensus += 1
                continue
            
            # Step 1: Check for exact match
            if answers_match(subject_answer, correct_answer):
                result["is_correct"] = True
                result["evaluation_method"] = "exact_match"
                exact_matches += 1
                print(f"Question {question_id}: Exact match")
                continue
            
            # Step 2: Get judgments from all models (or use cached ones)
            print(f"Evaluating question {question_id} using LLM panel...")
            model_judgments = {}  # Dictionary to store model:judgment pairs
            
            # Get the subject_id from the top-level of the test data
            file_subject_id = test_data.get("subject_id", "")
            
            # Check which models should be excluded to avoid self-judging
            self_judging_models = []
            for model in self.judge_models:
                # Check if model name appears in the file's subject_id
                if model.lower() in file_subject_id.lower():
                    self_judging_models.append(model)
                    print(f"Excluding {model} from judging because it's part of subject_id: {file_subject_id}")
            
            # Filter to only valid judge models (those not in self_judging_models)
            valid_judge_models = [m for m in self.judge_models if m not in self_judging_models]
            
            if not valid_judge_models:
                print(f"Warning: No valid judges available after excluding self-judging models from {file_subject_id}")
                continue
            
            # First check if we already have judgments in the cache for this question
            cache_key = self._make_cache_key(question_id, subject_answer)
            if cache_key in self.ratings_cache:
                # Extract existing ratings from cache
                for rating in self.ratings_cache[cache_key].get("ratings", []):
                    if rating["rater"] in valid_judge_models:
                        model_judgments[rating["rater"]] = rating["label"]
                        print(f"Using cached rating from {rating['rater']}: {rating['label']}")
            
            # Get judgments from models not found in cache
            missing_models = [model for model in valid_judge_models if model not in model_judgments]
            for judge_model in missing_models:
                judgment = self.get_llm_judgment(
                    question_data["question"], 
                    correct_answer, 
                    subject_answer, 
                    judge_model
                )
                
                if judgment:
                    model_judgments[judge_model] = judgment
                    # Add to cache
                    self._add_to_cache(
                        question_id, question_data["question"], correct_answer, subject_answer, judge_model, judgment
                    )
            
            # Step 3: Determine plurality decision
            if model_judgments:
                judgments = list(model_judgments.values())
                judgment_counts = Counter(judgments)
                most_common_items = judgment_counts.most_common()
                most_common_judgment, count = most_common_items[0]
                
                # Check for ties
                is_tie = len(most_common_items) > 1 and most_common_items[0][1] == most_common_items[1][1]
                
                # Store judgments in the result
                result["judgments"] = model_judgments
                
                if is_tie:
                    # If there's a tie for most common judgment, we don't have consensus
                    result["is_correct"] = None
                    result["evaluation_method"] = "tie"
                    no_consensus += 1
                    print(f"Question {question_id}: Tie in judgments: {dict(judgment_counts)}")
                else:
                    # Use the plurality judgment (most common)
                    if most_common_judgment == "YES":
                        result["is_correct"] = True
                    elif most_common_judgment == "NO":
                        result["is_correct"] = False
                    else:  # NOT ATTEMPTED
                        result["is_correct"] = None
                    
                    result["evaluation_method"] = "llm_plurality"
                    plurality_decisions += 1
                    print(f"Question {question_id}: Plurality vote: {most_common_judgment} ({count}/{len(judgments)})")
            else:
                # No judgments received
                print(f"Question {question_id}: No judgments received")
        
        # Calculate overall accuracy
        correct_count = sum(1 for result in test_data["results"].values() 
                            if result.get("is_correct") is True)
        total_evaluated = sum(1 for result in test_data["results"].values() 
                            #if result.get("is_correct") is not None)
                            if result.get("is_correct") is True or result.get("is_correct") is False)
        
        if total_evaluated > 0:
            test_data["accuracy"] = correct_count / total_evaluated
        
        # Save updated results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total questions: {total_questions}")
        print(f"Exact matches: {exact_matches}")
        print(f"Plurality decisions: {plurality_decisions}")
        print(f"No consensus: {no_consensus}")
        print(f"Accuracy: {test_data['accuracy'] if 'accuracy' in test_data else 'N/A'}")
        print(f"Results saved to: {output_file}")
        
        return output_file

def main():
    
    test_data_file = "./capabilities_test_logs/claude-sonnet-4-20250514_SimpleQA_500_1748311017_test_data.json"#"./pass_game_logs/aop_gpt-4o-2024-08-06_1746999717_game_data.json"#
    judge_models = ["gpt-4o-2024-08-06", "deepseek-chat", "gemini-2.0-flash-001"]#["grok-3-latest", "gemini-2.0-flash-001", "gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022"]
    
    print(f"Evaluating {test_data_file} using models: {judge_models}")
    
    evaluator = ShortAnswerEvaluator(judge_models)
    evaluator.evaluate_test_results(test_data_file)

if __name__ == "__main__":
    main()