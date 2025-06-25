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
    text = str(text).lower() # Ensure text is string
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Check if answers match using normalization
def answers_match(answer1, answer2):
    """Check if two answers match after normalization."""
    return normalize_text(answer1) == normalize_text(answer2)

class ShortAnswerEvaluator:
    def __init__(self, judge_models, cache_file="./shortanswer_ratings_cache.json"):
        """
        Initialize the base evaluator.
        
        Args:
            judge_models: List of model names to use as judges
            cache_file: Path to cached ratings
        """
        self.judge_models = judge_models
        self.cache_file = cache_file
        self.ratings_cache = self._load_cache()
        self.model_clients = {}

        for model_name in self.judge_models:
            # Ensure unique judge ID for all instances by appending timestamp
            judge_subject_id = f"judge_{model_name}_{int(time.time())}"
            self.model_clients[model_name] = BaseGameClass(
                subject_id=judge_subject_id,
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
                print(f"Warning: Unexpected label response '{label_response}' from {judge_model} for Q: {question_text[:50]}... A: {subject_answer[:50]}...")
                return None
        
        except Exception as e:
            print(f"Error getting judgment from {judge_model} for Q: {question_text[:50]}... A: {subject_answer[:50]}... Error: {e}")
            return None

    def _perform_evaluation_for_item(self, question_id, question_text, correct_answer, subject_answer, file_subject_id_for_exclusion):
        """
        Performs evaluation for a single item (question/trial).
        Handles exact match, LLM panel evaluation (with caching and self-judging exclusion),
        and plurality decision.

        Returns:
            dict: {
                "is_correct": True/False/None,
                "evaluation_method": str,
                "judgments": dict
            }
        """
        # Step 1: Check for exact match
        if answers_match(subject_answer, correct_answer):
            print(f"QID {question_id}: Exact match (Correct)")
            return {
                "is_correct": True,
                "evaluation_method": "exact_match",
                "judgments": {}
            }

        # Step 2: LLM panel evaluation
        print(f"Evaluating QID {question_id} using LLM panel for subject answer: '{str(subject_answer)[:50]}...'")
        model_judgments_dict = {}

        # Determine valid judge models
        self_judging_models = [model for model in self.judge_models if model.lower() in file_subject_id_for_exclusion.lower()]
        valid_judge_models = [m for m in self.judge_models if m not in self_judging_models]

        if not valid_judge_models:
            print(f"QID {question_id}: Skipping LLM evaluation as no valid judges were identified for this item (file_subject_id: {file_subject_id_for_exclusion}).")
            return {
                "is_correct": None,
                "evaluation_method": "no_valid_judges",
                "judgments": {}
            }
        
        print(f"Valid judges for QID {question_id} (file_subject_id: {file_subject_id_for_exclusion}): {valid_judge_models}")

        # Check cache first
        cache_key = self._make_cache_key(question_id, subject_answer)
        if cache_key in self.ratings_cache:
            for rating_entry in self.ratings_cache[cache_key].get("ratings", []):
                if rating_entry["rater"] in valid_judge_models:
                    model_judgments_dict[rating_entry["rater"]] = rating_entry["label"]
                    print(f"QID {question_id}: Using cached rating from {rating_entry['rater']}: {rating_entry['label']}")
        
        # Get judgments from models not found in cache
        missing_models = [model for model in valid_judge_models if model not in model_judgments_dict]
        for judge_model in missing_models:
            print(f"QID {question_id}: Querying {judge_model}...")
            judgment = self.get_llm_judgment(question_text, correct_answer, subject_answer, judge_model)
            if judgment:
                model_judgments_dict[judge_model] = judgment
                self._add_to_cache(question_id, question_text, correct_answer, subject_answer, judge_model, judgment)
            else:
                print(f"QID {question_id}: No judgment received from {judge_model}")
        
        # Step 3: Determine plurality decision
        if not model_judgments_dict:
            print(f"QID {question_id}: No LLM judgments received for evaluation.")
            return {
                "is_correct": None,
                "evaluation_method": "llm_no_judgments_received",
                "judgments": {}
            }

        judgments_list = list(model_judgments_dict.values())
        judgment_counts = Counter(judgments_list)
        most_common_items = judgment_counts.most_common()
        
        if not most_common_items: # Should not happen if model_judgments_dict is not empty
            print(f"QID {question_id}: No judgments recorded despite attempting LLM eval.")
            return {
                "is_correct": None,
                "evaluation_method": "llm_no_judgments_recorded",
                "judgments": model_judgments_dict
            }

        most_common_judgment, count = most_common_items[0]
        is_tie = len(most_common_items) > 1 and most_common_items[0][1] == most_common_items[1][1]
        
        if is_tie:
            print(f"QID {question_id}: Tie in judgments: {dict(judgment_counts)}")
            return {
                "is_correct": None,
                "evaluation_method": "tie",
                "judgments": model_judgments_dict
            }
        else:
            final_correctness = None
            if most_common_judgment == "YES":
                final_correctness = True
            elif most_common_judgment == "NO":
                final_correctness = False
            # else NOT ATTEMPTED or other, final_correctness remains None
            
            print(f"QID {question_id}: Plurality vote: {most_common_judgment} ({count}/{len(judgments_list)}) -> Correct: {final_correctness}")
            return {
                "is_correct": final_correctness,
                "evaluation_method": "llm_plurality",
                "judgments": model_judgments_dict
            }

    def evaluate_test_results(self, test_data_file, output_file=None):
        """
        Evaluate test results from the given file (standard format).
        
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
        file_subject_id_for_exclusion = test_data.get("subject_id", "") # Get once for the file

        for ctr, (question_id, result) in enumerate(results.items()):
            # Skip questions that have already been evaluated
            if "evaluation_method" in result and result["evaluation_method"] not in ["pending", None, ""]:
                # Recalculate stats for already evaluated items
                if result.get("evaluation_method") == "exact_match":
                    exact_matches += 1
                elif result.get("evaluation_method") == "llm_plurality":
                    plurality_decisions += 1
                elif result.get("evaluation_method") == "tie": # or "no_valid_judges" or "llm_no_judgments_received"
                    no_consensus +=1 # Grouping undecided/unevaluated by LLM here
                elif result.get("evaluation_method") in ["no_valid_judges", "llm_no_judgments_received", "llm_no_judgments_recorded"]:
                    no_consensus +=1 # Explicitly count these as needing attention or unable to evaluate
                continue

            question_data = result["question"]
            subject_answer = result.get("subject_answer")
            correct_answer = question_data.get("correct_answer")
            question_text = question_data.get("question")

            if not all([subject_answer, correct_answer, question_text]):
                print(f"Skipping QID {question_id} due to missing critical data (answer, correct_answer, or question_text).")
                result["is_correct"] = None
                result["evaluation_method"] = "skipped_missing_data"
                result["judgments"] = {}
                no_consensus +=1 # Count as unevaluated
                continue
            
            evaluation_outcome = self._perform_evaluation_for_item(
                question_id,
                question_text,
                correct_answer,
                subject_answer,
                file_subject_id_for_exclusion
            )

            result["is_correct"] = evaluation_outcome["is_correct"]
            result["evaluation_method"] = evaluation_outcome["evaluation_method"]
            result["judgments"] = evaluation_outcome["judgments"]

            # Update statistics based on the new evaluation outcome
            if evaluation_outcome["evaluation_method"] == "exact_match":
                exact_matches += 1
            elif evaluation_outcome["evaluation_method"] == "llm_plurality":
                plurality_decisions += 1
            elif evaluation_outcome["evaluation_method"] in ["tie", "no_valid_judges", "llm_no_judgments_received", "llm_no_judgments_recorded"]:
                no_consensus += 1

            print(f"Finished evaluating {ctr + 1}/{total_questions}")
        
        # Calculate overall accuracy
        # Ensure 'results' is the correct key, it might be test_data["results"] or similar
        # This part needs to be careful about the structure of test_data
        target_results_dict = test_data.get("results", test_data.get("phase1_results", test_data.get("phase2_results")))
        if target_results_dict is None:
            print("Error: Could not find results dictionary in test_data.")
            test_data["accuracy"] = None
        else:
            correct_count = sum(1 for res_item in target_results_dict.values()
                                if res_item.get("is_correct") is True)
            total_evaluated = sum(1 for res_item in target_results_dict.values()
                                  if res_item.get("is_correct") is True or res_item.get("is_correct") is False)
            
            if total_evaluated > 0:
                test_data["accuracy"] = correct_count / total_evaluated
            else:
                test_data["accuracy"] = None # Avoid division by zero, explicitly set to None
        
        # Save updated results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total questions: {total_questions}")
        print(f"Exact matches (Correct): {exact_matches}")
        print(f"LLM Plurality Decisions: {plurality_decisions}")
        # Correctly count LLM correct/incorrect based on 'is_correct' after LLM plurality
        llm_correct_plurality = sum(1 for res_item in target_results_dict.values() if res_item.get("evaluation_method") == "llm_plurality" and res_item.get("is_correct") is True)
        llm_incorrect_plurality = sum(1 for res_item in target_results_dict.values() if res_item.get("evaluation_method") == "llm_plurality" and res_item.get("is_correct") is False)
        print(f"  LLM Plurality Correct: {llm_correct_plurality}")
        print(f"  LLM Plurality Incorrect: {llm_incorrect_plurality}")
        print(f"Ties / No Consensus / Not Evaluated by LLM: {no_consensus}")
        print(f"Accuracy (based on True/False evaluations): {test_data.get('accuracy', 'N/A'):.2%}" if test_data.get('accuracy') is not None else "N/A")
        print(f"Results saved to: {output_file}")
        
        return output_file

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
        llm_evaluated_count = 0 # Tracks items that went through LLM path and got judgments
        llm_correct_plurality = 0
        llm_incorrect_plurality = 0
        llm_ties = 0
        
        file_subject_id_for_exclusion = game_data.get("subject_id", "")

        for ctr, trial_data in enumerate(game_data["results"]):
            if trial_data.get("delegation_choice") == "Self" and trial_data.get("subject_answer"):
                trials_to_evaluate += 1
                # Ensure unique fallback q_id, incorporating original counter and time for robustness
                question_id = trial_data.get("question_id", f"q_{ctr}_{int(time.time())}")
                question_text = trial_data.get("question_text")
                subject_answer = trial_data.get("subject_answer")
                correct_answer = trial_data.get("correct_answer")

                if not all([question_id, question_text, subject_answer, correct_answer]):
                    print(f"Skipping trial QID {question_id} due to missing critical data.")
                    trial_data["subject_correct"] = None
                    trial_data["team_correct"] = None # Mirror subject_correct
                    trial_data["evaluation_method"] = "skipped_missing_data"
                    trial_data["judgments"] = {}
                    continue

                evaluation_outcome = self._perform_evaluation_for_item(
                    question_id,
                    question_text,
                    correct_answer,
                    subject_answer,
                    file_subject_id_for_exclusion
                )

                trial_data["subject_correct"] = evaluation_outcome["is_correct"]
                trial_data["evaluation_method"] = evaluation_outcome["evaluation_method"]
                trial_data["judgments"] = evaluation_outcome["judgments"]
                trial_data["team_correct"] = trial_data["subject_correct"] # Mirror for "Self" trials

                # Update statistics based on the evaluation outcome
                if evaluation_outcome["evaluation_method"] == "exact_match":
                    exact_matches += 1
                elif evaluation_outcome["evaluation_method"] == "llm_plurality":
                    llm_evaluated_count += 1
                    if evaluation_outcome["is_correct"] is True:
                        llm_correct_plurality += 1
                    elif evaluation_outcome["is_correct"] is False:
                        llm_incorrect_plurality += 1
                    # If is_correct is None (e.g. "NOT ATTEMPTED" plurality), it's not counted here
                elif evaluation_outcome["evaluation_method"] == "tie":
                    llm_evaluated_count += 1
                    llm_ties += 1
                # Other outcomes like "no_valid_judges", "llm_no_judgments_received" are handled by _perform_evaluation_for_item
                # and their 'is_correct' will be None. They don't increment llm_evaluated_count here
                # as they didn't result in a plurality decision or tie from received judgments.
                print(f"Finished evaluating {ctr + 1}")
            
        # Calculate overall accuracy for "Self" trials
        self_answered_correctly = 0
        self_answered_total_evaluated = 0 # Only True/False evaluations
        for trial_data in game_data["results"]:
            if trial_data.get("delegation_choice") == "Self" and trial_data.get("subject_answer"):
                if trial_data.get("subject_correct") is True:
                    self_answered_correctly += 1
                if trial_data.get("subject_correct") is True or trial_data.get("subject_correct") is False:
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
        print(f"  Evaluated by LLM panel (resulting in plurality/tie): {llm_evaluated_count}")
        print(f"    LLM Plurality Correct: {llm_correct_plurality}")
        print(f"    LLM Plurality Incorrect: {llm_incorrect_plurality}")
        print(f"    LLM Ties (Undecided): {llm_ties}")
        # Add count of trials that couldn't be fully evaluated by LLM if needed
        other_llm_outcomes = trials_to_evaluate - exact_matches - llm_evaluated_count
        if other_llm_outcomes > 0:
            print(f"    LLM Other (e.g. no valid judges, no judgments received): {other_llm_outcomes}")
        print(f"Overall subject accuracy on 'Self' trials (where evaluated True/False): {overall_subject_accuracy_on_self_trials:.2%}" if overall_subject_accuracy_on_self_trials is not None else "N/A")
        print(f"Results saved to: {output_file}")
        
        return output_file

def main():
    test_data_file = "./capabilities_test_logs/gemini-2.0-flash-001_SimpleQA_500_1750864388_test_data.json"
    #test_data_file = "./delegate_game_logs/claude-3-sonnet-20240229_GPSA_50_500_team0.2_temp0.0_1750854313_game_data.json"
    #test_data_file = "./pass_game_logs/claude-3-5-sonnet-20241022_SimpleQA_noqcnt_nopcnt_noscnt_temp0.0_1750721190_game_data.json"
    
    judge_models = ["deepseek-chat", "gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022"]#["grok-3-latest", "gemini-2.0-flash-001", "gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "deepseek-chat"]


    print(f"Evaluating {test_data_file} using models: {judge_models}")
    
    evaluator = ShortAnswerEvaluator(judge_models)

    if "_game_logs" in test_data_file:
        print(f"Detected delegate game log file: {test_data_file}")
        evaluator.evaluate_delegate_game_file(test_data_file)
    else:
        print(f"Detected standard test results file: {test_data_file}")
        evaluator.evaluate_test_results(test_data_file)

if __name__ == "__main__":
    main()