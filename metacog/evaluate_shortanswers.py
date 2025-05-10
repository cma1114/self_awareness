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
    
    def _get_cache_key(self, question_id, question_text, correct_answer, subject_answer):
        """Create a unique key for the cache based on question and answer."""
        norm_subject_answer = normalize_text(subject_answer)
        return f"{question_id}:{norm_subject_answer}"
    
    def _get_cached_rating(self, question_id, question_text, correct_answer, subject_answer, judge_model):
        """Get cached rating if it exists."""
        cache_key = self._get_cache_key(question_id, question_text, correct_answer, subject_answer)
        
        if cache_key in self.ratings_cache:
            for rating in self.ratings_cache[cache_key].get("ratings", []):
                if rating["rater"] == judge_model:
                    return rating["label"]
        
        return None
    
    def _add_to_cache(self, question_id, question_text, correct_answer, subject_answer, judge_model, rating):
        """Add a rating to the cache."""
        cache_key = self._get_cache_key(question_id, question_text, correct_answer, subject_answer)
        
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
    
    def get_llm_judgment(self, question_id, question_text, correct_answer, subject_answer, judge_model):
        """Get LLM judgment on answer correctness."""
        # Check cache first
        cached_rating = self._get_cached_rating(
            question_id, question_text, correct_answer, subject_answer, judge_model
        )
        
        if cached_rating is not None:
            print(f"Using cached rating from {judge_model}: {cached_rating}")
            return cached_rating
        
        # Prepare prompt for LLM
        prompt = f"""
You are evaluating whether a given answer is correct for a question.

Question: {question_text}
Correct answer: {correct_answer}
User answer: {subject_answer}

Is the user answer correct? Consider semantic equivalence, not just exact matches.
Respond with EXACTLY ONE of these options:
- YES (if the answer is correct or equivalent to the correct answer)
- NO (if the answer is incorrect)
- UNCERTAIN (if you cannot confidently determine correctness)

Your judgment:
"""
        
        # Get model's judgment
        client = self.model_clients[judge_model]
        options = ["YES", "NO", "UNCERTAIN"]
        
        try:
            # Call the LLM through BaseGameClass
            judgment, _, _ = client._get_llm_answer(
                options=options,
                q_text=prompt,
                message_history=[],
                keep_appending=False,
                setup_text="Evaluate the answer's correctness based solely on factual accuracy."
            )
            
            # Add to cache
            self._add_to_cache(
                question_id, question_text, correct_answer, subject_answer, judge_model, judgment
            )
            
            return judgment
        
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
        total_questions = len(test_data["results"])
        exact_matches = 0
        majority_votes = 0
        no_consensus = 0
        
        # Process each question
        for question_id, result in test_data["results"].items():
            question_data = result["question"]
            subject_answer = result["subject_answer"]
            correct_answer = question_data["correct_answer"]
            
            # Step 1: Check for exact match
            if answers_match(subject_answer, correct_answer):
                result["is_correct"] = True
                result["evaluation_method"] = "exact_match"
                exact_matches += 1
                print(f"Question {question_id}: Exact match")
                continue
            
            # Step 2: Get judgments from all models
            print(f"Evaluating question {question_id} using LLM panel...")
            judgments = []
            
            for judge_model in self.judge_models:
                judgment = self.get_llm_judgment(
                    question_id, 
                    question_data["question"], 
                    correct_answer, 
                    subject_answer, 
                    judge_model
                )
                
                if judgment:
                    judgments.append(judgment)
            
            # Step 3: Determine majority vote
            if judgments:
                judgment_counts = Counter(judgments)
                most_common = judgment_counts.most_common(1)[0]
                most_common_judgment, count = most_common
                
                # Check if there's a clear majority
                if count > len(judgments) / 2:  # More than 50%
                    if most_common_judgment == "YES":
                        result["is_correct"] = True
                    elif most_common_judgment == "NO":
                        result["is_correct"] = False
                    else:  # UNCERTAIN
                        result["is_correct"] = None
                    
                    result["evaluation_method"] = "llm_majority"
                    result["judgments"] = {model: self.get_llm_judgment(
                        question_id, question_data["question"], correct_answer, 
                        subject_answer, model
                    ) for model in self.judge_models}
                    
                    majority_votes += 1
                    print(f"Question {question_id}: Majority vote: {most_common_judgment} ({count}/{len(judgments)})")
                else:
                    # No clear majority
                    result["is_correct"] = None
                    result["evaluation_method"] = "no_consensus"
                    result["judgments"] = {model: self.get_llm_judgment(
                        question_id, question_data["question"], correct_answer, 
                        subject_answer, model
                    ) for model in self.judge_models}
                    
                    no_consensus += 1
                    print(f"Question {question_id}: No consensus: {dict(judgment_counts)}")
            else:
                # No judgments received
                print(f"Question {question_id}: No judgments received")
                result["evaluation_method"] = "no_judgments"
        
        # Calculate overall accuracy
        correct_count = sum(1 for result in test_data["results"].values() 
                            if result.get("is_correct") is True)
        total_evaluated = sum(1 for result in test_data["results"].values() 
                            if result.get("is_correct") is not None)
        
        if total_evaluated > 0:
            test_data["accuracy"] = correct_count / total_evaluated
        
        # Save updated results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total questions: {total_questions}")
        print(f"Exact matches: {exact_matches}")
        print(f"Majority votes: {majority_votes}")
        print(f"No consensus: {no_consensus}")
        print(f"Accuracy: {test_data['accuracy'] if 'accuracy' in test_data else 'N/A'}")
        print(f"Results saved to: {output_file}")
        
        return output_file

def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_shortanswers.py <test_data_file> <judge_model1,judge_model2,...> [output_file]")
        sys.exit(1)
    
    test_data_file = sys.argv[1]
    judge_models = sys.argv[2].split(',')
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"Evaluating {test_data_file} using models: {judge_models}")
    
    evaluator = ShortAnswerEvaluator(judge_models)
    evaluator.evaluate_test_results(test_data_file, output_file)

if __name__ == "__main__":
    main()