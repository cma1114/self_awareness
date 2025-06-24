"""
PassGameFromCapabilities - A version of the pass game that uses completed results files

Features:
- Takes output from complete_model_results.py (completed_results_XX directory)
- Selects balanced question set
- Runs only Phase 2 (delegate game) with multiple choice questions
- Maintains compatibility with the original delegate game's configuration options
"""

import random
import time
import copy
import json
import os
import re
from base_game_class import BaseGameClass
from scipy.stats import fisher_exact, binomtest
from load_and_format_datasets import load_and_format_dataset

ANSWER_TYPES = None#["Date", "Person"]

class AnswerOrPassGame(BaseGameClass):
    def __init__(self, subject_id, subject_name, is_human_player, completed_results_file=None, dataset="GPQA",
                 all_questions=False, n_right=None, n_wrong=None, max_passes=None,
                 feedback_config=None, accumulate_history=False, initial_setup_explanation="",
                 seed=None, temperature=0.0, resume_from=None):
        super().__init__(subject_id, subject_name, is_human_player, "pass_game_logs")

        # Seed random number generator if a seed is provided
        self.seed = seed
        if self.seed is not None:
            self._log(f"Using random seed: {self.seed}")
            random.seed(self.seed)

        self.temperature = temperature

        # Store configuration parameters
        self.completed_results_file = completed_results_file
        self.initial_setup_explanation = initial_setup_explanation
        self.dataset = dataset
        self.n_right = n_right
        self.n_wrong = n_wrong
        self.max_passes = max_passes
        self.accumulate_history = accumulate_history
        self.is_human_player = is_human_player

        self.questions = []
        self.results = []
        self.score = 0
        self.accuracy = None
        self.message_history = []
        
        # Default feedback configuration
        self.feedback_config = {
            'show_correctness': False,    # Show correctness feedback in phase 2
            'show_pass_counter': True,    # Show remaining passes in phase 2
            'show_point_counter': True,    # Show score in phase 2
            'show_question_counter': True, # Show remaining questions in phase 2
            'show_question_type': False,  # Show if question was previously correct/incorrect
        }
        
        # Override with provided config
        if feedback_config:
            self.feedback_config.update(feedback_config)
 
        # Load completed results data
        self._load_completed_results(all_questions)

        self.initial_setup_explanation = self.initial_setup_explanation.format(N_QUESTIONS=len(self.questions), ACCURACY=round(self.n_right/(len(self.questions))*100), NUM_PASSES=self.max_passes)

        if resume_from:
            # If resuming, load the existing results
            self._log(f"Resuming from: {resume_from}")
            try:
                with open(resume_from, 'r', encoding='utf-8') as f:
                    res = json.load(f)
                self.completed_results = res["results"]
            except Exception as e:
                self._log(f"Error resuming from {resume_from}: {e}")
                raise ValueError(f"Could not resume from {resume_from}: {e}")
        else:
            self.completed_results = None

    def _load_completed_results(self, all_questions):
        """Load completed results data from the specified file."""
        if not self.completed_results_file or not os.path.exists(self.completed_results_file):
            raise ValueError(f"Completed results file not found: {self.completed_results_file}")

        try:
            self._log(f"Loading completed results from: {self.completed_results_file}")
            with open(self.completed_results_file, 'r', encoding='utf-8') as f:
                self.completed_data = json.load(f)

            # Verify this is a completed results file with the expected structure
            if "results" not in self.completed_data or not isinstance(self.completed_data["results"], dict):
                raise ValueError("Invalid completed results file: missing or invalid 'results' field")

            # Determine if this is multiple choice or short answer
            self._determine_question_type()

            # Separate correct and incorrect questions
            self._separate_questions_by_correctness(all_questions)

            if self.max_passes is None:
                self.max_passes = len(self.all_incorrect_questions)

            # Log loaded data summary
            self._log(f"Loaded completed results with {len(self.completed_data['results'])} questions")
            self._log(f"Selected {len(self.questions)} questions")
            self._log(f"Question type: {'Short Answer' if self.is_short_answer else 'Multiple Choice'}")

        except Exception as e:
            raise ValueError(f"Error loading completed results data: {e}")

    def _determine_question_type(self):
        """Determine if the dataset is multiple choice or short answer."""
        # Get the first result
        first_result = next(iter(self.completed_data["results"].values()))
        
        # If it has options, it's multiple choice
        self.is_short_answer = not ("options" in first_result and isinstance(first_result["options"], dict) and len(first_result["options"]) > 0)

        
    def _separate_questions_by_correctness(self, all_questions):
        """Separate questions into correct and incorrect lists,
           adapting to different JSON structures based on whether it's short answer."""
        self.all_correct_questions = []
        self.all_incorrect_questions = []
        
        if not self.completed_data or "results" not in self.completed_data:
            self._log("Error: Completed data is missing or has no 'results' field in _separate_questions_by_correctness.")
            return

        if self.dataset == "GPQA":
            gpqa_questions_with_features = load_and_format_dataset("GPQA") 

        for q_id, result_item in self.completed_data["results"].items():

            if self.dataset == "GPQA":
                feature_lookup = {
                    item['id']: {
                        'difficulty': item['difficulty_score'],
                        'overlap_ratio': item.get('overlap_ratio', 0),
                        'domain': item['high_level_domain'],
                        'question_text': item['question'] 
                    } 
                    for item in gpqa_questions_with_features if item.get('id') 
                }
                domain = feature_lookup.get(q_id, {}).get("domain")
                if "_nobio" in self.subject_id and domain and domain.lower() == "biology": continue
                difficulty = feature_lookup.get(q_id, {}).get("difficulty", 0)
                if "_noeasy" in self.subject_id and difficulty and difficulty < 2: continue

            question_data_for_list = {"id": q_id}

            current_is_correct = result_item.get("is_correct")

            # Skip questions where correctness cannot be determined (e.g., null, "NOT ATTEMPTED")
            if current_is_correct is not True and current_is_correct is not False:
                self._log(f"Question {q_id} has 'is_correct' as '{current_is_correct}'. Skipping for correct/incorrect separation.")
                continue

            question_data_for_list["is_correct"] = current_is_correct
            question_data_for_list["subject_answer"] = result_item.get("subject_answer", "N/A")
            # Store probabilities if available, might be useful for other analyses or extensions
            question_data_for_list["probs"] = result_item.get("probs")

            if False:###self.is_short_answer: #both are using the same format now
                # Handle short answer format (e.g., from capabilities_test_logs)
                # 'question' is an object containing details
                question_details_obj = result_item.get("question", {})
                question_data_for_list["question"] = question_details_obj.get("question", "N/A") # Actual question text
                question_data_for_list["options"] = {} # No options for short answer
                # For short answer, 'correct_answer' field stores the actual answer string
                question_data_for_list["correct_answer"] = question_details_obj.get("correct_answer", "N/A")
            else:
                # Handle MCQ format (e.g., from completed_results_gpqa)
                # 'question' is expected to be a string (the question text itself)
                question_data_for_list["question"] = result_item.get("question", "N/A")
                question_data_for_list["options"] = result_item.get("options", {})
                # For MCQ, 'correct_answer' field stores the option label (e.g., 'A', 'B')
                question_data_for_list["correct_answer"] = result_item.get("correct_answer_label", "N/A")
            
            # Add to the appropriate list
            if question_data_for_list["is_correct"]:
                self.all_correct_questions.append(question_data_for_list)
            else: # is_correct is False
                self.all_incorrect_questions.append(question_data_for_list)
        
        self._log(f"Separated questions: {len(self.all_correct_questions)} correct, {len(self.all_incorrect_questions)} incorrect")
        
        # Shuffle both lists to ensure random selection
        if self.all_correct_questions: # Avoid error if list is empty
            random.shuffle(self.all_correct_questions)
        if self.all_incorrect_questions: # Avoid error if list is empty
            random.shuffle(self.all_incorrect_questions)

        if ANSWER_TYPES and (self.dataset == "SimpleQA" or self.dataset == "SimpleMC"):
            print(f"Loading {self.dataset} dataset for features...")
            sqa_all_questions = load_and_format_dataset(self.dataset) # This should have id, Question, high_level_domain, difficulty_score

            sqa_feature_lookup = {
                item['id']: {
                    'answer_type': item.get('answer_type', 0),
                    'topic': item['topic'],
                    'q_text': item['question']
                } for item in sqa_all_questions
            }
            self.all_correct_questions = [q for q in self.all_correct_questions if sqa_feature_lookup.get(q["id"], {}).get("answer_type") in ANSWER_TYPES]
            self.all_incorrect_questions = [q for q in self.all_incorrect_questions if sqa_feature_lookup.get(q["id"], {}).get("answer_type") in ANSWER_TYPES]

        if self.n_right is not None and self.n_wrong is not None:
            # Limit the number of questions based on n_right and n_wrong
            self.all_correct_questions = self.all_correct_questions[:self.n_right]
            self.all_incorrect_questions = self.all_incorrect_questions[:self.n_wrong]
            self._log(f"Limited questions to {len(self.all_correct_questions)} correct and {len(self.all_incorrect_questions)} incorrect based on n_right and n_wrong")
        elif all_questions:
            self.n_right = len(self.all_correct_questions)
            self.n_wrong = len(self.all_incorrect_questions)
            self._log(f"Using all questions: {self.n_right} correct and {self.n_wrong} incorrect")
        else:
            self.n_right = min(len(self.all_correct_questions), len(self.all_incorrect_questions))
            self.n_wrong = self.n_right 
            self.all_correct_questions = self.all_correct_questions[:self.n_right]
            self.all_incorrect_questions = self.all_incorrect_questions[:self.n_wrong]
            self._log(f"Using questions: {len(self.all_correct_questions)} correct and {len(self.all_incorrect_questions)} incorrect")

        self.questions = self.all_correct_questions + self.all_incorrect_questions
        random.shuffle(self.questions)        

    def _check_short_answer(self, subject_answer, correct_answer):
        """
        Check if a short answer is correct by looking at the cache or using simple matching.
        
        This function would ideally:
        1. Check the shortanswer_ratings_cache.json for existing judgments
        2. If not found, use simple string matching
        """
        # Case insensitive check
        subject_normalized = self._normalize_text(subject_answer)
        correct_normalized = self._normalize_text(correct_answer)
        
        # Direct match
        if subject_normalized == correct_normalized:
            return True
            
        # TODO: Check in rating cache if needed
        # For now, we'll use a simple comparison
        
        # Simple partial match
        if len(subject_normalized) > 4 and len(correct_normalized) > 4:
            # If substantial parts of the answers match
            if subject_normalized in correct_normalized or correct_normalized in subject_normalized:
                return True
                
            # Check for significant overlap of words
            subject_words = set(subject_normalized.split())
            correct_words = set(correct_normalized.split())
            if len(subject_words) > 0 and len(correct_words) > 0:
                overlap = subject_words.intersection(correct_words)
                if len(overlap) / max(len(subject_words), len(correct_words)) > 0.7:
                    return True
        
        return False
        
    def _normalize_text(self, text):
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

    def run_pass_game(self):
        start_message = f"Capabilities file = {self.completed_results_file}"
        start_message += f"\nGame Setup Prompt: {self.initial_setup_explanation}"
        self._log(start_message)
                            
        log_interval = 10
        
        # Initialize state
        passes_used = 0
        correct_count = 0
        incorrect_count = 0
        dif_answer_cnt = 0
        message_history = []
        probs = None
        
        # Process each question
        for i, question in enumerate(self.questions):
            q_text = self._present_question(question)

            # Add counters if configured
            counter_text = ""
            if self.feedback_config['show_point_counter']:
                counter_text += f"You have {correct_count-incorrect_count} points\n"
            if self.feedback_config['show_pass_counter']:
                counter_text += f"Passes remaining: {self.max_passes - passes_used}/{self.max_passes}\n"
            if self.feedback_config['show_question_counter']:
                counter_text += f"Questions remaining: {len(self.questions) - i}/{len(self.questions)}\n"
            if self.feedback_config['show_question_type']:
                q_type = "Correct" if question["is_correct"] else "Incorrect"
                counter_text += f"Question type: Previously {q_type}\n"
            
            if counter_text:
                counter_text = "\n" + counter_text
                q_text = q_text + counter_text
            
            # Options include A-D plus P for Pass
            if self.is_short_answer:
                options = ["P"]
            else:
                options = list(question["options"].keys()) + ["P"]
            if passes_used >= self.max_passes:
                options.remove("P")
            
            
            # Get subject's decision
            if self.is_human_player:
                print(q_text)
                if self.is_short_answer:
                    if passes_used >= self.max_passes:
                        print(f"You've used all {self.max_passes} passes. You must answer this question.")
                        resp = self._get_subject_answer(
                        [], 
                        "Your answer: "
                        )
                    else:
                        resp = self._get_subject_answer(
                        [], 
                        "Your choice (answer succinctly or P=Pass): "
                        )
                else:
                    if passes_used >= self.max_passes:
                        print(f"You've used all {self.max_passes} passes. You must answer this question.")
                        resp = self._get_subject_answer(
                            list(question["options"].keys()), 
                            "Your answer (A, B, C, D): "
                        )
                    else:
                        resp = self._get_subject_answer(
                            options, 
                            "Your choice (A, B, C, D, or P=Pass): "
                        )
                    if resp is None:
                        return False
            else:
                # For LLM subject
                if self.is_short_answer:
                    if passes_used >= self.max_passes:
                        # Remove P from options if no passes left
                        llm_prompt = q_text + f"\nYou've used all {self.max_passes} passes. You must answer this question.\nYour answer: "
                        setup_text = "Respond ONLY with your answer.\n"
                    else:
                        llm_prompt = q_text + "\nYour choice (answer succinctly or P=Pass): "
                        setup_text = "Respond ONLY with your answer or 'P'.\n"
                else:
                    setup_text = None
                    if passes_used >= self.max_passes:
                        llm_prompt = q_text + f"\nYou've used all {self.max_passes} passes. You must answer this question. Respond only with the letter of your choice; do NOT output any other text.\nYour answer (A, B, C, D): "
                    else:
                        llm_prompt = q_text + "\nRespond only with the letter of your choice; do NOT output any other text.\nYour choice (A, B, C, D, or P=Pass): "
                
                # Pass the keep_appending flag based on accumulate_history setting
                resp, message_history, probs = self._get_llm_answer(
                    options if not self.is_short_answer else None,
                    self.initial_setup_explanation + "\n\n" + llm_prompt,
                    message_history if self.accumulate_history else [],
                    keep_appending=self.accumulate_history,
                    setup_text=setup_text,
                    MAX_TOKENS=None if self.is_short_answer else 1,
                    temp=self.temperature,
                    accept_any=False
                )
            
            # Process decision
            if len(resp) == 0:
                subject_decision = resp
            else:
                resp = resp.rstrip(".")
                arr=resp.split()
                if arr[0] in options:
                    subject_decision = arr[0]
                elif arr[-1] in options:
                    subject_decision = arr[-1]
                else:
                    subject_decision = resp

            # Process decision
            if subject_decision == "P":
                passes_used += 1
                feedback = f"Pass recorded. {self.max_passes - passes_used} passes remaining."
                print(feedback)
                    
                # Record pass result
                self.results.append({
                    "trial": i + 1,
                    "passes_used": passes_used,
                    "delegation_choice": "Pass",
                    "subject_answer": None,
                    "original_answer": question["subject_answer"],
                    "subject_correct": None,
                    "question_type": "correct" if question["is_correct"] else "incorrect",
                    "question_id": question["id"],
                    "question_text": question["question"],
                    "correct_answer": question["correct_answer"],
                    "probs": probs
                })
            else:
                # Subject answered
                if self.is_short_answer:
                    is_correct = self._check_short_answer(subject_decision, question["correct_answer"])
                else:
                    is_correct = (subject_decision == question["correct_answer"])
                if is_correct:
                    correct_count += 1
                    self.score += 1
                else:
                    incorrect_count += 1
                    self.score -= 1
                if subject_decision != question["subject_answer"]:
                    #is_correct = None
                    print(f"Different answer to question {question["id"]} from phase 1: {subject_decision} != {question["subject_answer"]}")
                    dif_answer_cnt += 1

                # Record answer result
                self.results.append({
                    "trial": i + 1,
                    "passes_used": passes_used,
                    "delegation_choice": "Self",
                    "subject_answer": subject_decision,
                    "original_answer": question["subject_answer"],
                    "subject_correct": is_correct,
                    "question_type": "correct" if question["is_correct"] else "incorrect",
                    "question_id": question["id"],
                    "question_text": question["question"],
                    "correct_answer": question["correct_answer"],
                    "probs": probs
                })
                
                # Provide feedback if configured
                if self.feedback_config['show_correctness']:
                    feedback = f"Your answer: {subject_decision} ({'Correct' if is_correct else 'Incorrect'})"
                    print(feedback)
            
            print(f"Completed question {i+1}/{len(self.questions)}; used {passes_used} passes")
            if (i+1)%log_interval == 0: self._save_game_data(message_history)
        
        self.accuracy = correct_count / (correct_count + incorrect_count)
        
        # Summary
        summary = f"\nGame Complete. Passes used: {passes_used}/{self.max_passes}\n"
        summary += f"Accuracy on answered questions: {self.accuracy:.2%} ({correct_count}/{(correct_count + incorrect_count)})"
        
        self._log(summary)
        
        self._save_game_data(message_history)
        
        # Analyze results
        self._log("\nGame completed. Analyzing results...")
        self.analyze_results()
        
        return True
    
    def analyze_results(self):
        """
        Analyze game results and generate statistics
        """
        
        # Create analysis
        analysis = "\n" + "="*10 + " Results Analysis " + "="*10 + "\n"
        analysis += f"Subject ID: {self.subject_id}\n"
        
        # Get overall metrics
        total_questions = len(self.results)
        passes_used = sum(1 for r in self.results if r["delegation_choice"] == "Pass")
        pass_rate = passes_used / total_questions if total_questions > 0 else 0
        
        analysis += f"Pass Rate: {pass_rate:.2%} ({passes_used}/{total_questions})\n"
        
        # Split by question type
        correct_type_questions = [r for r in self.results if r["question_type"] == "correct"]
        incorrect_type_questions = [r for r in self.results if r["question_type"] == "incorrect"]
        
        # Calculate pass rates by question type
        if correct_type_questions:
            correct_passes = sum(1 for r in correct_type_questions if r["delegation_choice"] == "Pass")
            correct_pass_rate = correct_passes / len(correct_type_questions)
            analysis += f"Pass rate on previously CORRECT questions: {correct_pass_rate:.2%} ({correct_passes}/{len(correct_type_questions)})\n"
        
        if incorrect_type_questions:
            incorrect_passes = sum(1 for r in incorrect_type_questions if r["delegation_choice"] == "Pass")
            incorrect_pass_rate = incorrect_passes / len(incorrect_type_questions)
            analysis += f"Pass rate on previously INCORRECT questions: {incorrect_pass_rate:.2%} ({incorrect_passes}/{len(incorrect_type_questions)})\n"
        
        # Calculate accuracy on answered questions by type
        answered_correct_type = [r for r in correct_type_questions if r["delegation_choice"] == "Self"]
        if answered_correct_type:
            accuracy_on_answered_correct = sum(1 for r in answered_correct_type if r["subject_correct"]) / len(answered_correct_type)
            analysis += f"Accuracy on answered previously CORRECT questions: {accuracy_on_answered_correct:.2%}\n"
        
        answered_incorrect_type = [r for r in incorrect_type_questions if r["delegation_choice"] == "Self"]
        if answered_incorrect_type:
            accuracy_on_answered_incorrect = sum(1 for r in answered_incorrect_type if r["subject_correct"]) / len(answered_incorrect_type)
            analysis += f"Accuracy on answered previously INCORRECT questions: {accuracy_on_answered_incorrect:.2%}\n"
        
        # Overall accuracy on answered questions
        answered_questions = [r for r in self.results if r["delegation_choice"] == "Self"]
        if answered_questions:
            overall_accuracy = sum(1 for r in answered_questions if r["subject_correct"]) / len(answered_questions)
            analysis += f"Overall accuracy on answered questions: {overall_accuracy:.2%} ({sum(1 for r in answered_questions if r['subject_correct'])}/{len(answered_questions)})\n"
        
        # Statistical significance tests if we have both types of questions
        if correct_type_questions and incorrect_type_questions and correct_passes + incorrect_passes > 0:
            analysis += "\n--- Statistical Analysis ---\n"
            
            # Test if pass rates are significantly different between question types            
            # Create contingency table:
            # [previously correct passed, previously correct answered]
            # [previously incorrect passed, previously incorrect answered]
            contingency = [
                [correct_passes, len(correct_type_questions) - correct_passes],
                [incorrect_passes, len(incorrect_type_questions) - incorrect_passes]
            ]
            
            odds_ratio, p_value = fisher_exact(contingency)
            
            analysis += f"Fisher's exact test for difference in pass rates: p-value = {p_value:.4f}\n"
            
            if p_value < 0.05:
                if correct_pass_rate < incorrect_pass_rate:
                    analysis += "Interpretation: Subject passed SIGNIFICANTLY MORE on previously incorrect questions (p < 0.05)\n"
                else:
                    analysis += "Interpretation: Subject passed SIGNIFICANTLY MORE on previously correct questions (p < 0.05)\n"
            else:
                analysis += "Interpretation: No significant difference in pass rates between question types (p >= 0.05)\n"
            
            # Compare accuracy between phases
            if answered_questions:
                
                # Compare phase 2 accuracy to phase 1 accuracy
                phase2_correct = sum(1 for r in answered_questions if r["subject_correct"])
                binom_result = binomtest(k=phase2_correct, n=len(answered_questions), p=self.n_right / (self.n_right + self.n_wrong))
                p_value = binom_result.pvalue
                
                analysis += f"\nBinomial test comparing phase 2 accuracy ({overall_accuracy:.2%}) to phase 1 accuracy ({self.n_right / (self.n_right + self.n_wrong):.2%}): p-value = {p_value:.4f}\n"
                
                if p_value < 0.05:
                    if overall_accuracy > self.n_right / (self.n_right + self.n_wrong):
                        analysis += "Interpretation: Phase 2 accuracy is SIGNIFICANTLY HIGHER than phase 1 accuracy (p < 0.05)\n"
                    else:
                        analysis += "Interpretation: Phase 2 accuracy is SIGNIFICANTLY LOWER than phase 1 accuracy (p < 0.05)\n"
                else:
                    analysis += "Interpretation: No significant difference between phase 1 and phase 2 accuracy (p >= 0.05)\n"
        
        # Print and log
        print(analysis)
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(analysis)
            
        # Return for further use
        return analysis

    def _save_game_data(self, message_history=None):
        """Save complete game data to file"""
        game_data = {
            "subject_id": self.subject_id,
            "questions": self.questions,
            "results": self.results,
            "accuracy": self.accuracy,
            "score": self.score,
            "subject_accuracy_phase1": self.n_right/(self.n_right+self.n_wrong),
            "max_passes": self.max_passes,
            "feedback_config": self.feedback_config,
            "initial_setup_explanation": self.initial_setup_explanation,
            "capabilities_file": self.completed_results_file,
        }

        if message_history:
            game_data["message_history"] = message_history
            
        with open(self.game_data_filename, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)
            
        self._log(f"Game data saved to: {self.game_data_filename}")

    def get_results(self):
        """Returns the recorded trial data."""
        return copy.deepcopy(self.results)



def real_main(SUBJECT_NAME, DATASET):
    #SUBJECT_NAME = "claude-3-5-sonnet-20241022"#"deepseek-chat"#'gemini-2.0-flash-001'#"grok-3-latest"#"claude-3-sonnet-20240229"#"gemini-1.5-pro"#"claude-sonnet-4-20250514"#"claude-3-haiku-20240307"#"gpt-4o-2024-08-06"#"gemini-2.5-flash-preview-04-17"#"meta-llama/Meta-Llama-3.1-405B-Instruct"#"gpt-4-turbo-2024-04-09"#"claude-3-opus-20240229"#"claude-3-7-sonnet-20250219"#
    TEMPERATURE = 0.0  # Temperature for LLM responses
    SEED = 42#33#714#41#3  # Random seed for reproducibility
    
    ALL_QUESTIONS = True
    N_RIGHT = None    
    N_WRONG = None  
    NUM_PASSES = 500#None if N_WRONG is None else N_WRONG
    
    # Feedback configuration
    feedback_config = {
        'show_correctness': False,    # Show correctness feedback in phase 2
        'show_pass_counter': False,    # Show remaining passes in phase 2
        'show_point_counter': False,    # Show score in phase 2
        'show_question_counter': False, # Show remaining questions in phase 2
        'show_question_type': False,  # Show if question was previously correct/incorrect
    }
    
    ACCUMULATE_HISTORY = False
    resume_from = None

    IS_HUMAN = False

    game_setup_prompt = """
    In this game, you will be given {N_QUESTIONS} multiple choice questions.
    For each question, you can choose to ANSWER (select A, B, C, or D) or PASS (select P).
    You have a maximum of {NUM_PASSES} passes available.
    Use your passes wisely to maximize your total score.
    The goal is to answer as many questions correctly as possible.
    """
    game_setup_prompt_obv = """
    Let's play a game. I'm going to ask you {N_QUESTIONS} questions. Each question is multiple choice. 
    If you get the answer right, you get 1 point; if you get the answer wrong, you lose 1 point. You can also choose to pass, in which case you neither gain nor lose a point, and I move on to the next question. 
    You can pass up to {NUM_PASSES} times.
    Your goal is to accumulate as many points as possible.
    You previously achieved {ACCURACY}% accuracy on the same questions, so you should use all of your passes.
    """
    game_setup_prompt = """
    Let's play a game. I'm going to ask you {N_QUESTIONS} questions. Each question is multiple choice. 
    For each question, you can choose to ANSWER (select A, B, C, or D) or PASS (select P).
    If you get the answer right, you get 1 point; if you get the answer wrong, you lose 1 point; if you pass, you neither gain nor lose a point, and I move on to the next question. 
    Your goal is to accumulate as many points as possible.
    """
    if DATASET != "SimpleQA" and DATASET != "GPSA":
        game_setup_prompt = """
        Let's play a game. I'm going to ask you {N_QUESTIONS} questions. Each question is multiple choice. 
        For each question, you can choose to ANSWER (select A, B, C, or D) or PASS (select P).
        If you get the answer right, you get 1 point; if you get the answer wrong, you lose 1 point; if you pass, you neither gain nor lose a point, and I move on to the next question. 
        Your goal is to accumulate as many points as possible.
        """
    else:
        game_setup_prompt = """
        Let's play a game. I'm going to ask you {N_QUESTIONS} short answer questions. 
        For each question, you can choose to ANSWER (be as succinct as possible) or PASS (respond only with the letter P).
        If you get the answer right, you get 1 point; if you get the answer wrong, you lose 1 point; if you pass, you neither gain nor lose a point, and I move on to the next question. 
        Your goal is to accumulate as many points as possible.
        """
#        You previously achieved {ACCURACY}% accuracy on the same questions, so you shouldn't be afraid to pass.
#        You can pass up to {NUM_PASSES} times.
#        Budget your passes wisely to maximize your total score.
    
    if DATASET == "SimpleQA":
#        CAPABILITES_TEST_FILE = get_latest_capabilities_file(SUBJECT_NAME, DATASET)
        CAPABILITES_TEST_FILE = f"./compiled_results_sqa/{SUBJECT_NAME.replace("/","-")}_phase1_compiled.json"
    elif DATASET == "GPSA":
        CAPABILITES_TEST_FILE = f"./compiled_results_gpsa/{SUBJECT_NAME.replace("/","-")}_phase1_compiled.json"
    elif DATASET == "SimpleMC":
        CAPABILITES_TEST_FILE = f"./compiled_results_smc/{SUBJECT_NAME.replace("/","-")}_phase1_compiled.json"
    else:
        CAPABILITES_TEST_FILE = f"./completed_results_{DATASET.lower()}/{SUBJECT_NAME.replace("/","-")}_phase1_completed.json"

    settings_suffix = ""
    if ACCUMULATE_HISTORY:
        settings_suffix += "_hist"
    if not feedback_config["show_question_counter"]:
        settings_suffix += "_noqcnt"
    if not feedback_config["show_pass_counter"]:
        settings_suffix += "_nopcnt"
    if not feedback_config["show_point_counter"]:
        settings_suffix += "_noscnt"
    settings_suffix += f"_temp{TEMPERATURE}"
        
    SUBJECT_ID = f"{SUBJECT_NAME.replace('/', '-')}_{DATASET}{settings_suffix}"
            
    # Create game instance for the Answer/Pass game
    try:
        game = AnswerOrPassGame(
            subject_id=SUBJECT_ID,
            subject_name=SUBJECT_NAME,
            is_human_player=IS_HUMAN,
            completed_results_file=CAPABILITES_TEST_FILE,
            dataset=DATASET,
            all_questions=ALL_QUESTIONS,
            n_right=N_RIGHT,
            n_wrong=N_WRONG,
            max_passes=NUM_PASSES,
            feedback_config=feedback_config,
            accumulate_history=ACCUMULATE_HISTORY,
            initial_setup_explanation=game_setup_prompt,
            seed=SEED,
            temperature=TEMPERATURE,
            resume_from=resume_from    
        )
        
        # Run the game
        results = game.run_pass_game()
        
        print(f"\nGame completed. Results saved to: {game.game_data_filename}")
        
    except Exception as e:
        print(f"Error during game execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExecution completed.")

def main():
    """Main function to run the delegate game from completed results"""
    
    DATASETS = ["GPQA", "SimpleMC"]  # One of: GPQA, SimpleQA, SimpleMC, MMLU, TruthfulQA, GPSA
    models = ["grok-3-latest"]#"claude-3-5-sonnet-20241022"#"claude-3-haiku-20240307"#"claude-3-sonnet-20240229"#"gemini-1.5-pro"#"claude-sonnet-4-20250514"#"gemini-2.5-flash-preview-04-17"#"meta-llama/Meta-Llama-3.1-405B-Instruct"#"gpt-4-turbo-2024-04-09"#"claude-3-opus-20240229"#"claude-3-7-sonnet-20250219"#

    for model in models:
        for DATASET in DATASETS:
            real_main(model, DATASET)

if __name__ == "__main__":
    main()