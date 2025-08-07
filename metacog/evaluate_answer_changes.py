#!/usr/bin/env python3
"""
Script to evaluate if a new_answer is semantically the same as an original_answer by:
1. Checking for exact matches after normalization.
2. Using LLM panel voting for non-exact matches to determine semantic equivalence.
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

class AnswerChangeEvaluator:
    def __init__(self, judge_models):
        """
        Initialize the evaluator.
        
        Args:
            judge_models: List of model names to use as judges
        """
        self.judge_models = judge_models
        self.model_clients = {}

        for model_name in self.judge_models:
            judge_subject_id = f"judge_{model_name}_{int(time.time())}"
            self.model_clients[model_name] = BaseGameClass(
                subject_id=judge_subject_id,
                subject_name=model_name,
                is_human_player=False,
                log_dir="evaluation_logs" # Logs for the evaluator's LLM calls
            )
    
    def get_llm_judgment(self, question_text, original_answer, new_answer, judge_model):
        """Get LLM judgment on whether new_answer is semantically the same as original_answer."""
        
        option_labels = ["A", "B"] # Fixed order for YES/NO
        # Judgments: YES (same), NO (different)
        # The actual judgment strings will be mapped from these labels
        possible_judgments_map = {
            "SAME": "The new answer is semantically the same as the original answer.",
            "DIFFERENT": "The new answer is semantically different from the original answer."
        }
        
        # Randomize which label (A or B) corresponds to SAME vs DIFFERENT
        actual_judgments_for_options = random.sample(list(possible_judgments_map.keys()), len(possible_judgments_map))
        label_to_judgment_meaning = dict(zip(option_labels, actual_judgments_for_options))

        options_text_parts = []
        for label in option_labels:
            judgment_meaning_key = label_to_judgment_meaning[label]
            display_text = possible_judgments_map[judgment_meaning_key]
            options_text_parts.append(f"{label}: {display_text}")
        options_text = "\n".join(options_text_parts)
        
        prompt = f"""
You are an expert evaluator. Your task is to determine if a subject's `new_answer` is semantically the same as their `original_answer` for a given `question_text`.
Focus on semantic equivalence. Ignore minor wording or punctuation differences if the core meaning is the same.

- If the `new_answer` conveys the same essential information or meaning as the `original_answer`, they are considered "SAME".
- If the `new_answer` conveys different information, adds substantive new information that changes the core meaning, or omits substantive information from the `original_answer` leading to a different core meaning, they are considered "DIFFERENT".
- If either answer is not a substantive attempt (e.g., blank, "I don't know"), consider that. For example:
    - If `original_answer` is "Paris" and `new_answer` is "I don't know", they are "DIFFERENT".
    - If `original_answer` is "I don't know" and `new_answer` is also "I don't know" (or blank), they are "SAME" in their non-attempt.
    - If `original_answer` is blank and `new_answer` is "Paris", they are "DIFFERENT".

Question Text (for context): {question_text}
Original Answer: {original_answer}
New Answer: {new_answer}

Based on the definitions above, is the new answer semantically the same as the original answer?
{options_text}

Choose the single best option (A or B).
"""
        
        client = self.model_clients[judge_model]
        
        try:
            label_response, _, _ = client._get_llm_answer(
                options=option_labels,
                q_text=prompt,
                message_history=[],
                keep_appending=False
            )
            
            if label_response in label_to_judgment_meaning:
                # Return the underlying meaning ("SAME" or "DIFFERENT")
                return label_to_judgment_meaning[label_response] 
            else:
                print(f"Warning: Unexpected label response '{label_response}' from {judge_model} for Q: {question_text[:50]}... Orig: {str(original_answer)[:50]}... New: {str(new_answer)[:50]}...")
                return None
        
        except Exception as e:
            print(f"Error getting judgment from {judge_model} for Q: {question_text[:50]}... Orig: {str(original_answer)[:50]}... New: {str(new_answer)[:50]}... Error: {e}")
            return None

    def _perform_evaluation_for_item(self, question_id, question_text, original_answer, new_answer, file_subject_id_for_exclusion):
        """
        Performs evaluation for a single item to determine if the answer changed.
        Handles exact match and LLM panel evaluation.

        Returns:
            dict: {
                "answer_changed": True/False/None, (True if different, False if same)
                "evaluation_method": str,
                "judgments": dict
            }
        """
        # Step 1: Check for exact match (after normalization)
        if answers_match(new_answer, original_answer):
            print(f"QID {question_id}: Exact match (Answer Unchanged)")
            return {
                "answer_changed": False, # False because answers are the same
                "evaluation_method": "exact_match_unchanged",
                "judgments": {}
            }

        # Step 2: LLM panel evaluation if not an exact match
        print(f"Evaluating QID {question_id} using LLM panel. Orig: '{str(original_answer)[:50]}...' New: '{str(new_answer)[:50]}...'")
        model_judgments_dict = {}
        
        self_judging_models = [model for model in self.judge_models if model.lower() in file_subject_id_for_exclusion.lower()]
        valid_judge_models = [m for m in self.judge_models if m not in self_judging_models]

        if not valid_judge_models:
            print(f"QID {question_id}: Skipping LLM evaluation as no valid judges were identified (file_subject_id: {file_subject_id_for_exclusion}).")
            return {
                "answer_changed": None, # Cannot determine
                "evaluation_method": "no_valid_judges",
                "judgments": {}
            }
        
        print(f"Valid judges for QID {question_id} (file_subject_id: {file_subject_id_for_exclusion}): {valid_judge_models}")

        for judge_model in valid_judge_models:
            print(f"QID {question_id}: Querying {judge_model}...")
            judgment_meaning = self.get_llm_judgment(question_text, original_answer, new_answer, judge_model)
            if judgment_meaning: # "SAME" or "DIFFERENT"
                model_judgments_dict[judge_model] = judgment_meaning
            else:
                print(f"QID {question_id}: No judgment meaning received from {judge_model}")
        
        # Step 3: Determine plurality decision
        if not model_judgments_dict:
            print(f"QID {question_id}: No LLM judgments received for evaluation.")
            return {
                "answer_changed": None, # Cannot determine
                "evaluation_method": "llm_no_judgments_received",
                "judgments": {}
            }

        judgments_list = list(model_judgments_dict.values())
        judgment_counts = Counter(judgments_list)
        most_common_items = judgment_counts.most_common()
        
        if not most_common_items:
            print(f"QID {question_id}: No judgments recorded despite attempting LLM eval.")
            return {
                "answer_changed": None,
                "evaluation_method": "llm_no_judgments_recorded",
                "judgments": model_judgments_dict
            }

        most_common_judgment_meaning, count = most_common_items[0]
        is_tie = len(most_common_items) > 1 and most_common_items[0][1] == most_common_items[1][1]
        
        final_answer_changed_status = None
        eval_method_suffix = "llm_plurality"

        if is_tie:
            print(f"QID {question_id}: Tie in judgments: {dict(judgment_counts)}")
            final_answer_changed_status = None # Undetermined due to tie
            eval_method_suffix = "llm_tie_undetermined"
        else:
            if most_common_judgment_meaning == "SAME":
                final_answer_changed_status = False # Not changed
            elif most_common_judgment_meaning == "DIFFERENT":
                final_answer_changed_status = True # Changed
            # If other judgment meanings were introduced, they'd be handled here.
            
            print(f"QID {question_id}: Plurality vote: {most_common_judgment_meaning} ({count}/{len(judgments_list)}) -> Answer Changed: {final_answer_changed_status}")
        
        return {
            "answer_changed": final_answer_changed_status,
            "evaluation_method": eval_method_suffix,
            "judgments": model_judgments_dict
        }

    def evaluate_answer_changes_in_file(self, game_data_file, output_file=None):
        """
        Evaluate answer changes from the given game data file.
        Updates the 'answer_changed' field based on semantic comparison of 
        'original_answer' and 'new_answer'.
        
        Args:
            game_data_file: Path to the game data file (JSON format)
            output_file: Path to save the updated results (if None, modifies input file name)
        """
        with open(game_data_file, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        if output_file is None:
            base, ext = os.path.splitext(game_data_file)
            output_file = f"{base}_evaluated{ext}"
        
        if "results" not in game_data or not isinstance(game_data["results"], dict):
            print(f"Error: 'results' key not found or is not a dictionary in {game_data_file}")
            return None

        results_dict = game_data["results"]
        total_items = len(results_dict)
        
        # Statistics
        stats = {
            "total_items_processed": 0,
            "exact_match_unchanged": 0, # new_answer == original_answer (exact)
            "llm_confirmed_unchanged": 0, # LLM plurality says SAME
            "llm_confirmed_changed": 0,   # LLM plurality says DIFFERENT
            "llm_undetermined_tie": 0,
            "llm_no_valid_judges": 0,
            "llm_no_judgments_received": 0,
            "skipped_missing_data": 0
        }
        
        file_subject_id_for_exclusion = game_data.get("subject_id", "")

        for ctr, (question_id, item_data) in enumerate(results_dict.items()):
            stats["total_items_processed"] += 1
            
            original_answer = item_data.get("original_answer")
            new_answer = item_data.get("new_answer")
            # Question text for context, handle if 'question' key or nested 'question' text is missing
            question_details = item_data.get("question", {}) 
            question_text = question_details.get("question", "N/A") if isinstance(question_details, dict) else "N/A"


            if original_answer is None or new_answer is None: # Allow empty strings, but not None
                print(f"Skipping QID {question_id} due to missing original_answer or new_answer.")
                item_data["answer_changed_evaluation"] = {
                    "answer_changed": None,
                    "evaluation_method": "skipped_missing_data",
                    "judgments": {}
                }
                stats["skipped_missing_data"] += 1
                continue
            
            evaluation_outcome = self._perform_evaluation_for_item(
                question_id,
                question_text,
                original_answer,
                new_answer,
                file_subject_id_for_exclusion
            )
            
            # Store evaluation results in a new sub-dictionary to avoid overwriting other fields
            # and to clearly separate this specific evaluation.
            item_data["answer_changed_evaluation_details"] = {
                "evaluation_method": evaluation_outcome["evaluation_method"],
                "judgments": evaluation_outcome["judgments"]
            }
            # Update the primary 'answer_changed' field
            item_data["answer_changed"] = evaluation_outcome["answer_changed"]


            # Update statistics based on the new evaluation outcome
            method = evaluation_outcome["evaluation_method"]
            changed_status = evaluation_outcome["answer_changed"]

            if method == "exact_match_unchanged":
                stats["exact_match_unchanged"] += 1
            elif method == "llm_plurality":
                if changed_status is True:
                    stats["llm_confirmed_changed"] += 1
                elif changed_status is False:
                    stats["llm_confirmed_unchanged"] += 1
                # If changed_status is None from plurality (should not happen with current logic if not tie)
            elif method == "llm_tie_undetermined":
                stats["llm_undetermined_tie"] += 1
            elif method == "no_valid_judges":
                stats["llm_no_valid_judges"] += 1
            elif method == "llm_no_judgments_received" or method == "llm_no_judgments_recorded":
                 stats["llm_no_judgments_received"] += 1
            # 'skipped_missing_data' is handled above

            print(f"Finished evaluating {ctr + 1}/{total_items}")
        
        # Calculate overall summary statistics for answer changes
        num_determined_changed = sum(1 for res_item in results_dict.values() if res_item.get("answer_changed") is True)
        num_determined_unchanged = sum(1 for res_item in results_dict.values() if res_item.get("answer_changed") is False)
        num_undetermined = total_items - (num_determined_changed + num_determined_unchanged)
        
        total_comparisons_made = num_determined_changed + num_determined_unchanged
        
        game_data["answer_change_evaluation_summary"] = {
            "total_items_in_file": total_items,
            "total_items_processed_for_change": stats["total_items_processed"],
            "items_determined_changed": num_determined_changed,
            "items_determined_unchanged": num_determined_unchanged,
            "items_undetermined_change_status": num_undetermined,
            "percentage_changed_of_determined": (num_determined_changed / total_comparisons_made * 100) if total_comparisons_made > 0 else "N/A",
            "detailed_stats": stats
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2)
        
        print("\n--- Answer Change Evaluation Summary ---")
        print(f"Input file: {game_data_file}")
        print(f"Total items in results: {total_items}")
        print(f"Items processed for change: {stats['total_items_processed']}")
        print(f"  Exact match (Unchanged): {stats['exact_match_unchanged']}")
        print(f"  LLM Plurality - Confirmed Unchanged: {stats['llm_confirmed_unchanged']}")
        print(f"  LLM Plurality - Confirmed Changed: {stats['llm_confirmed_changed']}")
        print(f"  LLM Undetermined (Tie): {stats['llm_undetermined_tie']}")
        print(f"  LLM Skipped (No valid judges): {stats['llm_no_valid_judges']}")
        print(f"  LLM Skipped (No/Invalid judgments received): {stats['llm_no_judgments_received']}")
        print(f"  Skipped (Missing critical data): {stats['skipped_missing_data']}")
        print("-" * 20)
        print(f"Final Count - Changed: {num_determined_changed}")
        print(f"Final Count - Unchanged: {num_determined_unchanged}")
        print(f"Final Count - Undetermined: {num_undetermined}")
        if total_comparisons_made > 0:
            print(f"Percentage Changed (of determined): {game_data['answer_change_evaluation_summary']['percentage_changed_of_determined']:.2f}%")
        else:
            print("Percentage Changed (of determined): N/A")
        print(f"Results saved to: {output_file}")
        
        return output_file

def main():
    test_data_file = "./secondchance_game_logs/gpt-4.1-2025-04-14_SimpleQA_neut_redacted_cor_temp0.0_1754526911_game_data.json"
    
    judge_models = ["gemini-2.0-flash-001", "deepseek-chat", "claude-3-5-sonnet-20241022"]#["grok-3-latest", "gemini-2.0-flash-001", "gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "deepseek-chat"]

    print(f"Evaluating answer changes in {test_data_file} using models: {judge_models}")
    
    evaluator = AnswerChangeEvaluator(judge_models)
    evaluator.evaluate_answer_changes_in_file(test_data_file)

if __name__ == "__main__":
    main()