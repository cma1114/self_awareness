#!/usr/bin/env python3
"""
Script to evaluate short-answer responses from delegate_game_logs format by:
1. Checking for exact matches for "Self" chosen answers.
2. Using LLM panel voting for non-exact "Self" chosen matches.
3. Caching judgments to avoid redundant API calls.
4. Updating the 'subject_correct' field in the original data structure.
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
    text = str(text).lower() # Ensure text is string
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Check if answers match using normalization
def answers_match(answer1, answer2):
    """Check if two answers match after normalization."""
    return normalize_text(answer1) == normalize_text(answer2)

class DelegateShortAnswerEvaluator: # Renamed class for clarity
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
        self.model_clients = {}
        
        for model_name in self.judge_models:
            self.model_clients[model_name] = BaseGameClass(
                subject_id=f"judge_{model_name}_{int(time.time())}", # Ensure unique judge ID
                subject_name=model_name,
                is_human_player=False,
                log_dir="evaluation_logs"
            )
    
    def _load_cache(self):
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
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.ratings_cache, f, indent=2)
    
    def _make_cache_key(self, question_id, subject_answer):
        norm_subject_answer = normalize_text(subject_answer)
        return f"{question_id}:{norm_subject_answer}"
    
    def _get_cached_rating(self, question_id, subject_answer, judge_model):
        cache_key = self._make_cache_key(question_id, subject_answer)
        if cache_key in self.ratings_cache:
            for rating in self.ratings_cache[cache_key].get("ratings", []):
                if rating["rater"] == judge_model:
                    return rating["label"]
        return None
    
    def _add_to_cache(self, question_id, question_text, correct_answer, subject_answer, judge_model, rating):
        cache_key = self._make_cache_key(question_id, subject_answer)
        if cache_key not in self.ratings_cache:
            self.ratings_cache[cache_key] = {
                "qid": question_id,
                "question": question_text,
                "correct_answer": correct_answer,
                "subject_answer": subject_answer,
                "ratings": []
            }
        
        for existing_rating in self.ratings_cache[cache_key]["ratings"]:
            if existing_rating["rater"] == judge_model:
                existing_rating["label"] = rating
                break
        else:
            self.ratings_cache[cache_key]["ratings"].append({
                "rater": judge_model,
                "label": rating
            })
        self._save_cache()
    
    def get_llm_judgment(self, question_text, correct_answer, subject_answer, judge_model):
        option_labels = ["A", "B", "C"]
        judgments = ["YES", "NO", "NOT ATTEMPTED"]
        shuffled_judgments = random.sample(judgments, len(judgments))
        label_to_judgment = dict(zip(option_labels, shuffled_judgments))
        
        option_displays = {
            "YES": "Correct (the subject response is semantically equivalent to the gold standard answer)",
            "NO": "Incorrect (the subject response is wrong, incomplete, or contains inaccurate information)",
            "NOT ATTEMPTED": "Not attempted (the subject response is not an attempt to answer the question)",
        }
        
        options_text = "\n".join([
            f"{label}: {option_displays[label_to_judgment[label]]}" 
            for label in option_labels
        ])
        
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
        client = self.model_clients[judge_model]
        options = option_labels
        
        try:
            label_response, _, _ = client._get_llm_answer(
                options=options,
                q_text=prompt,
                message_history=[],
                keep_appending=False
            )
            if label_response in label_to_judgment:
                return label_to_judgment[label_response]
            else:
                print(f"Warning: Unexpected label response '{label_response}' from {judge_model} for Q: {question_text[:50]}... A: {subject_answer[:50]}...")
                return None
        except Exception as e:
            print(f"Error getting judgment from {judge_model} for Q: {question_text[:50]}... A: {subject_answer[:50]}... Error: {e}")
            return None
    
    def evaluate_delegate_game_file(self, game_data_file, output_file=None):
        """
        Evaluate short answers from a delegate game log file.
        Updates 'subject_correct' for trials where delegation_choice was 'Self'.
        """
        with open(game_data_file, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        if output_file is None:
            base, ext = os.path.splitext(game_data_file)
            output_file = f"{base}_evaluated{ext}"
        
        if "results" not in game_data or not isinstance(game_data["results"], list):
            print(f"Error: 'results' key not found or is not a list in {game_data_file}")
            return None

        trials_to_evaluate = 0
        exact_matches = 0
        llm_evaluated_count = 0
        llm_correct_plurality = 0
        llm_incorrect_plurality = 0
        llm_ties = 0
        
        file_subject_id = game_data.get("subject_id", "") # Used for self-judging exclusion

        # Determine valid judge models for the file
        self_judging_models_for_file = [model for model in self.judge_models if model.lower() in file_subject_id.lower()]
        valid_judge_models_for_file = [m for m in self.judge_models if m not in self_judging_models_for_file]

        if not valid_judge_models_for_file:
            print(f"Warning: No valid judges available for the entire file {game_data_file} (subject_id: {file_subject_id}) after excluding: {self_judging_models_for_file}.")
            # LLM evaluation will be skipped for all trials in this file if this list is empty.
        else:
            print(f"Valid judges for file {game_data_file} (subject_id: {file_subject_id}): {valid_judge_models_for_file}")


        for trial_data in game_data["results"]:
            if trial_data.get("delegation_choice") == "Self" and trial_data.get("subject_answer"):
                trials_to_evaluate += 1
                question_id = trial_data.get("question_id", f"q_{trials_to_evaluate}") # Fallback q_id
                question_text = trial_data.get("question_text")
                subject_answer = trial_data.get("subject_answer")
                correct_answer = trial_data.get("correct_answer")

                if not all([question_id, question_text, subject_answer, correct_answer]):
                    print(f"Skipping trial due to missing data: QID {question_id}")
                    trial_data["subject_correct"] = None # Mark as unevaluated
                    trial_data["team_correct"] = None # Mirror subject_correct
                    trial_data["evaluation_method"] = "skipped_missing_data"
                    continue

                # Initialize evaluation fields
                trial_data["subject_correct"] = None 
                trial_data["evaluation_method"] = "pending"
                trial_data["judgments"] = {}

                # Step 1: Check for exact match
                if answers_match(subject_answer, correct_answer):
                    trial_data["subject_correct"] = True
                    trial_data["team_correct"] = True # Mirror subject_correct
                    trial_data["evaluation_method"] = "exact_match_delegate"
                    exact_matches += 1
                    print(f"QID {question_id}: Exact match (Correct)")
                    continue
                
                # Step 2: LLM panel evaluation
                print(f"Evaluating QID {question_id} using LLM panel for subject answer: '{subject_answer[:50]}...'")
                model_judgments_dict = {}

                # Check if there are any valid judges for this file before proceeding with LLM eval for this trial
                if not valid_judge_models_for_file: # Use the file-level list
                    print(f"QID {question_id}: Skipping LLM evaluation as no valid judges were identified for this file ({file_subject_id}).")
                    trial_data["subject_correct"] = None # No LLM eval possible
                    trial_data["team_correct"] = None # Mirror subject_correct
                    trial_data["evaluation_method"] = "no_valid_judges_for_file"
                    continue
                
                # print(f"Evaluating QID {question_id} using LLM panel (judges: {valid_judge_models_for_file}) for subject answer: '{subject_answer[:50]}...'") # Already printed before loop
                # Check cache first
                cache_key = self._make_cache_key(question_id, subject_answer)
                if cache_key in self.ratings_cache:
                    for rating_entry in self.ratings_cache[cache_key].get("ratings", []):
                        if rating_entry["rater"] in valid_judge_models_for_file: # Use file-level list
                            model_judgments_dict[rating_entry["rater"]] = rating_entry["label"]
                            print(f"QID {question_id}: Using cached rating from {rating_entry['rater']}: {rating_entry['label']}")
                
                # Get judgments from models not found in cache
                missing_models = [model for model in valid_judge_models_for_file if model not in model_judgments_dict] # Use file-level list
                for judge_model in missing_models:
                    print(f"QID {question_id}: Querying {judge_model}...")
                    judgment = self.get_llm_judgment(question_text, correct_answer, subject_answer, judge_model)
                    if judgment:
                        model_judgments_dict[judge_model] = judgment
                        self._add_to_cache(question_id, question_text, correct_answer, subject_answer, judge_model, judgment)
                    else:
                        print(f"QID {question_id}: No judgment received from {judge_model}")
                
                trial_data["judgments"] = model_judgments_dict
                
                # Step 3: Determine plurality decision
                if model_judgments_dict:
                    llm_evaluated_count +=1
                    judgments_list = list(model_judgments_dict.values())
                    judgment_counts = Counter(judgments_list)
                    most_common_items = judgment_counts.most_common()
                    
                    if not most_common_items: # Should not happen if model_judgments_dict is not empty
                        trial_data["evaluation_method"] = "llm_no_judgments_recorded"
                        print(f"QID {question_id}: No judgments recorded despite attempting LLM eval.")
                        continue

                    most_common_judgment, count = most_common_items[0]
                    is_tie = len(most_common_items) > 1 and most_common_items[0][1] == most_common_items[1][1]
                    
                    if is_tie:
                        trial_data["subject_correct"] = None # Undecided
                        trial_data["team_correct"] = None # Mirror subject_correct
                        trial_data["evaluation_method"] = "tie_delegate"
                        llm_ties += 1
                        print(f"QID {question_id}: Tie in judgments: {dict(judgment_counts)}")
                    else:
                        if most_common_judgment == "YES":
                            trial_data["subject_correct"] = True
                            llm_correct_plurality +=1
                        elif most_common_judgment == "NO":
                            trial_data["subject_correct"] = False
                            llm_incorrect_plurality +=1
                        else: # NOT ATTEMPTED or other
                            trial_data["subject_correct"] = None
                        trial_data["team_correct"] = trial_data["subject_correct"] # Mirror subject_correct
                        trial_data["evaluation_method"] = "llm_plurality_delegate"
                        print(f"QID {question_id}: Plurality vote: {most_common_judgment} ({count}/{len(judgments_list)}) -> Correct: {trial_data['subject_correct']}")
                else:
                    trial_data["subject_correct"] = None # No LLM judgments
                    trial_data["team_correct"] = None # Mirror subject_correct
                    trial_data["evaluation_method"] = "llm_no_judgments_received"
                    print(f"QID {question_id}: No LLM judgments received for evaluation.")
        
        # Calculate overall accuracy for "Self" trials
        self_answered_correctly = 0
        self_answered_total_evaluated = 0
        for trial_data in game_data["results"]:
            if trial_data.get("delegation_choice") == "Self" and trial_data.get("subject_answer"):
                if trial_data.get("subject_correct") is True:
                    self_answered_correctly += 1
                if trial_data.get("subject_correct") is True or trial_data.get("subject_correct") is False: # Count only if evaluated to True/False
                    self_answered_total_evaluated +=1
        
        overall_subject_accuracy_on_self_trials = (self_answered_correctly / self_answered_total_evaluated) if self_answered_total_evaluated > 0 else None
        game_data["overall_subject_accuracy_on_self_trials"] = overall_subject_accuracy_on_self_trials

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2)
        
        print("\n--- Evaluation Summary for Delegate Game File ---")
        print(f"Input file: {game_data_file}")
        print(f"Total trials in results: {len(game_data['results'])}")
        print(f"Trials with 'delegation_choice' == 'Self' and subject_answer: {trials_to_evaluate}")
        print(f"  Exact matches (Correct): {exact_matches}")
        print(f"  Evaluated by LLM panel: {llm_evaluated_count}")
        print(f"    LLM Plurality Correct: {llm_correct_plurality}")
        print(f"    LLM Plurality Incorrect: {llm_incorrect_plurality}")
        print(f"    LLM Ties (Undecided): {llm_ties}")
        print(f"Overall subject accuracy on 'Self' trials (where evaluated): {overall_subject_accuracy_on_self_trials:.2%}" if overall_subject_accuracy_on_self_trials is not None else "N/A")
        print(f"Results saved to: {output_file}")
        
        return output_file

def main():
    delegate_game_file = "./delegate_game_logs/claude-3-5-sonnet-20241022_SimpleQA_50_100_subj0.7_team0.5_temp0.0_1747970647_game_data.json"
    
    judge_models = ["gemini-2.0-flash-001", "deepseek-chat", "gpt-4o-2024-08-06"]#["grok-3-latest", "gemini-2.0-flash-001", "gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022"]
    
    print(f"Evaluating delegate game file: {delegate_game_file} using judges: {judge_models}")
    
    evaluator = DelegateShortAnswerEvaluator(judge_models)
    evaluator.evaluate_delegate_game_file(delegate_game_file)

if __name__ == "__main__":
    main()