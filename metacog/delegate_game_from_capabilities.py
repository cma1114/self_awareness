"""
DelegateGameFromCapabilities - A version of the delegate game that uses completed results files

Features:
- Takes output from complete_model_results.py (completed_results_XX directory)
- Constructs a simulated Phase 1 history based on configuration
- Selects question sets for phase 1 and phase 2 based on the desired accuracy ratios
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
import scipy.stats
import glob
from load_and_format_datasets import load_and_format_dataset

PHASE1_TYPES = None#["Number", "Other", "Place"]
PHASE2_TYPES = None#["Date", "Person"]

class DelegateGameFromCapabilities(BaseGameClass):
    """
    Manages the psychological experiment game flow, using completed results files.
    """
    def __init__(self, subject_id, subject_name, is_human_player=False,
                 completed_results_file=None, dataset="GPQA",
                 n_trials_phase1=100, n_trials_phase2=100,
                 teammate_accuracy_phase1=0.7, teammate_accuracy_phase2=0.7,
                 feedback_config=None, override_subject_accuracy=None, randomize_phase1_answers=False,
                 use_phase1_summary=True, use_phase1_history=True,
                 redact_phase1_answers=False, initial_setup_explanation="",
                 seed=None, temperature=0.0):
        """
        Initializes the game instance using completed results data.

        Args:
            subject_id (str): Identifier for the current subject/session.
            subject_name (str): Name of the subject (model name for LLMs).
            is_human_player (bool): Whether the subject is a human (True) or LLM (False).
            completed_results_file (str): Path to the completed results file.
            dataset (str): Dataset name (GPQA, SimpleQA, MMLU, TruthfulQA).
            n_trials_phase1 (int): Number of trials for Phase 1 simulation.
            n_trials_phase2 (int): Number of trials for Phase 2.
            teammate_accuracy_phase1 (float): The teammate's target accuracy in Phase 1 (0.0 to 1.0).
            teammate_accuracy_phase2 (float): The teammate's target accuracy in Phase 2 (0.0 to 1.0).
            feedback_config (dict): Configuration for feedback options.
            override_subject_accuracy (float): If not None, overrides subject accuracy in phase 1.
            randomize_phase1_answers (bool): If True, randomizes the subject's answers/correctness in Phase 1.
            use_phase1_summary (bool): Whether to include a summary of phase 1 in the history.
            use_phase1_history (bool): Whether to include the full phase 1 history.
            redact_phase1_answers (bool): If True, replaces subject's phase 1 answers with "[redacted]".
            initial_setup_explanation (str): Explanation text to show at the beginning.
            seed (int, optional): Seed for random number generation. Defaults to None.
        """
        super().__init__(subject_id, subject_name, is_human_player, "delegate_game_logs")

        # Seed random number generator if a seed is provided
        self.seed = seed
        if self.seed is not None:
            self._log(f"Using random seed: {self.seed}")
            random.seed(self.seed)

        self.temperature = temperature

        # Store parameters
        self.completed_results_file = completed_results_file
        self.dataset = dataset
        self.n_trials_phase1 = n_trials_phase1
        self.n_trials_phase2 = n_trials_phase2
        self.teammate_accuracy_phase1 = teammate_accuracy_phase1
        self.teammate_accuracy_phase2 = teammate_accuracy_phase2
        self.override_subject_accuracy = override_subject_accuracy
        self.randomize_phase1_answers = randomize_phase1_answers
        self.use_phase1_summary = use_phase1_summary
        self.use_phase1_history = use_phase1_history
        self.redact_phase1_answers = redact_phase1_answers
        self.initial_setup_explanation = initial_setup_explanation

        # Default feedback configuration
        self.feedback_config = {
            'phase1_subject_feedback': True,      # Show subject's answer feedback in phase 1
            'phase1_teammate_feedback': True,     # Show teammate's answer feedback in phase 1
            'phase2_subject_feedback': False,     # Show subject's answer feedback in phase 2
            'phase2_teammate_feedback': False,    # Show teammate's answer feedback in phase 2
            'show_answer_with_correctness': True, # Include answer in feedback
        }
        
        # Override with provided config
        if feedback_config:
            self.feedback_config.update(feedback_config)

        # Initialize state
        self.results = []
        self.phase1_questions = []
        self.phase2_questions = []
        self.subject_accuracy_phase1 = None
        self.true_subject_accuracy = None  # The actual accuracy from completed results
        self.phase2_score = None
        self.phase2_accuracy = None
        self.stored_message_history = []
        self.phase1_simulated_history = []
        self.is_short_answer = False
        self.teammate_phase1_answers = {}
        self.all_correct_questions = []
        self.all_incorrect_questions = []

        # Load completed results data
        self._load_completed_results()

    def _load_completed_results(self):
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

            if "accuracy" not in self.completed_data:
                raise ValueError("Invalid completed results file: missing 'accuracy' field")

            # Store the true subject accuracy from the file
            self.true_subject_accuracy = self.completed_data["accuracy"]
            self._log(f"True subject accuracy from completed results: {self.true_subject_accuracy:.2%}")

            # Set subject accuracy for Phase 1 (use override if specified)
            if self.override_subject_accuracy is not None:
                self.subject_accuracy_phase1 = self.override_subject_accuracy
                self._log(f"Using override subject accuracy for Phase 1: {self.subject_accuracy_phase1:.2%}")
            else:
                self.subject_accuracy_phase1 = self.true_subject_accuracy
                self._log(f"Using true subject accuracy for Phase 1: {self.subject_accuracy_phase1:.2%}")

            # Determine if this is multiple choice or short answer
            self._determine_question_type()

            # Separate correct and incorrect questions
            self._separate_questions_by_correctness()

            # Select and prepare questions for phase 1 and phase 2
            self._prepare_phase1_questions()
            self._prepare_phase2_questions()

            # Pre-determine teammate's answers
            self._predetermine_teammate_answers(phase=1)
            self._predetermine_teammate_answers(phase=2)

            # Log loaded data summary
            self._log(f"Loaded completed results with {len(self.completed_data['results'])} questions")
            self._log(f"Selected {len(self.phase1_questions)} questions for Phase 1")
            self._log(f"Selected {len(self.phase2_questions)} questions for Phase 2")
            self._log(f"Subject phase 1 accuracy: {self.subject_accuracy_phase1:.2%}")
            self._log(f"Teammate phase 1 accuracy: {self.teammate_accuracy_phase1:.2%}")
            self._log(f"Question type: {'Short Answer' if self.is_short_answer else 'Multiple Choice'}")

        except Exception as e:
            raise ValueError(f"Error loading completed results data: {e}")

    def _determine_question_type(self):
        """Determine if the dataset is multiple choice or short answer."""
        # Get the first result
        first_result = next(iter(self.completed_data["results"].values()))
        
        # If it has options, it's multiple choice
        self.is_short_answer = not ("options" in first_result and isinstance(first_result["options"], dict) and len(first_result["options"]) > 0)
        
        return self.is_short_answer
        
    def _separate_questions_by_correctness(self):
        """Separate questions into correct and incorrect lists,
           adapting to different JSON structures based on whether it's short answer."""
        self.all_correct_questions = []
        self.all_incorrect_questions = []
        
        if not self.completed_data or "results" not in self.completed_data:
            self._log("Error: Completed data is missing or has no 'results' field in _separate_questions_by_correctness.")
            return

        if self.dataset == "GPQA":
            from load_and_format_datasets import load_and_format_dataset
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

    def _prepare_phase1_questions(self):
        """
        Select questions for Phase 1 based on desired accuracy ratio.
        
        This method selects n_trials_phase1 questions, with a proportion of correct questions
        matching the subject_accuracy_phase1 (or override_subject_accuracy if specified).
        """
        # Calculate how many correct questions we need to match the desired accuracy
        num_correct_needed = int(round(self.subject_accuracy_phase1 * self.n_trials_phase1))
        num_incorrect_needed = self.n_trials_phase1 - num_correct_needed
        
        self._log(f"Selecting {num_correct_needed} correct and {num_incorrect_needed} incorrect questions for Phase 1")
        
        # Check if we have enough questions of each type
        if num_correct_needed > len(self.all_correct_questions):
            self._log(f"Warning: Not enough correct questions available. Using all {len(self.all_correct_questions)} correct questions.")
            num_correct_needed = len(self.all_correct_questions)
            num_incorrect_needed = min(self.n_trials_phase1 - num_correct_needed, len(self.all_incorrect_questions))
            
        if num_incorrect_needed > len(self.all_incorrect_questions):
            self._log(f"Warning: Not enough incorrect questions available. Using all {len(self.all_incorrect_questions)} incorrect questions.")
            num_incorrect_needed = len(self.all_incorrect_questions)
            num_correct_needed = min(self.n_trials_phase1 - num_incorrect_needed, len(self.all_correct_questions))
        
        #Filter to phase1 types if needed
        if PHASE1_TYPES and self.dataset == "SimpleQA":
            print("Loading main SimpleQA dataset for features...")
            sqa_all_questions = load_and_format_dataset("SimpleQA") # This should have id, Question, high_level_domain, difficulty_score

            sqa_feature_lookup = {
                item['id']: {
                    'answer_type': item.get('answer_type', 0),
                    'topic': item['topic'],
                    'q_text': item['question']
                } for item in sqa_all_questions
            }
            phase1_correct_questions = [q for q in self.all_correct_questions if sqa_feature_lookup.get(q["id"], {}).get("answer_type") in PHASE1_TYPES][:num_correct_needed]
            phase1_incorrect_questions = [q for q in self.all_incorrect_questions if sqa_feature_lookup.get(q["id"], {}).get("answer_type") in PHASE1_TYPES][:num_incorrect_needed]
        else:
            # Select the questions
            phase1_correct_questions = self.all_correct_questions[:num_correct_needed]
            phase1_incorrect_questions = self.all_incorrect_questions[:num_incorrect_needed]
        
        # Combine and shuffle
        self.phase1_questions = phase1_correct_questions + phase1_incorrect_questions
        random.shuffle(self.phase1_questions)
        
        # Create ID lookup for quick access
        self.phase1_question_ids = set(q["id"] for q in self.phase1_questions)
        
        # Calculate actual subject accuracy for Phase 1
        actual_num_correct = len(phase1_correct_questions)
        self.subject_accuracy_phase1 = actual_num_correct / len(self.phase1_questions) if self.phase1_questions else 0
        
        self._log(f"Selected {len(self.phase1_questions)} questions for Phase 1 with subject accuracy {self.subject_accuracy_phase1:.2%}")

    def _prepare_phase2_questions(self):
        """
        Select questions for Phase 2 based on the true accuracy from the completed results,
        targeting the actual number of trials for the run (self.n_trials_phase2).
        
        Phase 2 questions are selected to:
        1. Be distinct from Phase 1 questions.
        2. Have a proportion of correct questions matching self.true_subject_accuracy
           for the self.n_trials_phase2 questions, as much as possible.
        """
        #Filter to phase1 types if needed
        if PHASE2_TYPES and self.dataset == "SimpleQA":
            print("Loading main SimpleQA dataset for features...")
            sqa_all_questions = load_and_format_dataset("SimpleQA") # This should have id, Question, high_level_domain, difficulty_score

            sqa_feature_lookup = {
                item['id']: {
                    'answer_type': item.get('answer_type', 0),
                    'topic': item['topic'],
                    'q_text': item['question']
                } for item in sqa_all_questions
            }
            remaining_correct_ordered = [q for q in self.all_correct_questions if sqa_feature_lookup.get(q["id"], {}).get("answer_type") in PHASE2_TYPES and q["id"] not in self.phase1_question_ids]
            remaining_incorrect_ordered = [q for q in self.all_incorrect_questions if sqa_feature_lookup.get(q["id"], {}).get("answer_type") in PHASE2_TYPES and q["id"] not in self.phase1_question_ids]
        else:
            # Filter out questions already used in Phase 1
            remaining_correct_ordered = [q for q in self.all_correct_questions if q["id"] not in self.phase1_question_ids]
            remaining_incorrect_ordered = [q for q in self.all_incorrect_questions if q["id"] not in self.phase1_question_ids]

        total_remaining_questions = len(remaining_correct_ordered) + len(remaining_incorrect_ordered)
        
        num_questions_for_this_run = self.n_trials_phase2

        if num_questions_for_this_run == 0:
            self._log("Phase 2 has 0 trials. No questions will be selected.")
            self.phase2_questions = []
            return

        if num_questions_for_this_run > total_remaining_questions:
            self._log(f"Warning: Requested {num_questions_for_this_run} questions for Phase 2, but only {total_remaining_questions} are available after Phase 1. Using all {total_remaining_questions} available questions.")
            num_questions_for_this_run = total_remaining_questions
        
        self.n_trials_phase2 = num_questions_for_this_run # Update instance variable to actual number

        if self.n_trials_phase2 == 0: # Can happen if total_remaining_questions was 0
            self._log("No questions available or selected for Phase 2.")
            self.phase2_questions = []
            return

        # Calculate target number of correct and incorrect questions for the run
        target_correct_for_run = int(round(self.true_subject_accuracy * self.n_trials_phase2))
        target_incorrect_for_run = self.n_trials_phase2 - target_correct_for_run

        selected_correct_qs = []
        selected_incorrect_qs = []

        # Try to meet the targets
        actual_correct_to_select = min(target_correct_for_run, len(remaining_correct_ordered))
        selected_correct_qs = remaining_correct_ordered[:actual_correct_to_select]

        actual_incorrect_to_select = min(target_incorrect_for_run, len(remaining_incorrect_ordered))
        selected_incorrect_qs = remaining_incorrect_ordered[:actual_incorrect_to_select]
        
        # If we haven't selected enough questions, fill up to self.n_trials_phase2
        current_total_selected = len(selected_correct_qs) + len(selected_incorrect_qs)
        still_needed = self.n_trials_phase2 - current_total_selected

        if still_needed > 0:
            # Try to fill with more correct questions if available and we took fewer than available
            can_add_more_correct = len(remaining_correct_ordered) - len(selected_correct_qs)
            add_c = min(still_needed, can_add_more_correct)
            if add_c > 0:
                selected_correct_qs.extend(remaining_correct_ordered[len(selected_correct_qs) : len(selected_correct_qs) + add_c])
                still_needed -= add_c
        
        if still_needed > 0:
            # Try to fill with more incorrect questions if available and we took fewer than available
            can_add_more_incorrect = len(remaining_incorrect_ordered) - len(selected_incorrect_qs)
            add_i = min(still_needed, can_add_more_incorrect)
            if add_i > 0:
                selected_incorrect_qs.extend(remaining_incorrect_ordered[len(selected_incorrect_qs) : len(selected_incorrect_qs) + add_i])
                # still_needed -= add_i # Not strictly necessary to update still_needed here as it's the last fill attempt

        self.phase2_questions = selected_correct_qs + selected_incorrect_qs
        random.shuffle(self.phase2_questions)
        
        # Ensure self.n_trials_phase2 reflects the true number of questions in self.phase2_questions
        # This should already be the case if logic above is correct, but as a safeguard:
        self.n_trials_phase2 = len(self.phase2_questions)

        if self.n_trials_phase2 > 0:
            actual_correct_count_in_run = sum(1 for q in self.phase2_questions if q["is_correct"])
            actual_accuracy_for_run = actual_correct_count_in_run / self.n_trials_phase2
            
            self._log(f"Selected {self.n_trials_phase2} questions for Phase 2. Target accuracy for run: {self.true_subject_accuracy:.2%}, Actual selected accuracy for this run: {actual_accuracy_for_run:.2%}")

            accuracy_diff = abs(actual_accuracy_for_run - self.true_subject_accuracy)
            # Using a threshold of 0.005 (0.5%) for "significant" deviation
            if accuracy_diff > 0.0051: # Add a tiny bit to avoid floating point issues with exact 0.005
                self._log(f"WARNING: Phase 2 actual accuracy ({actual_accuracy_for_run:.2%}) deviates significantly from target ({self.true_subject_accuracy:.2%}). Difference: {accuracy_diff:.2%}")
        else:
            self._log("No questions were ultimately selected for Phase 2.")

    def _predetermine_teammate_answers(self, phase=1):
        """
        Pre-determines teammate answers to ensure exact match with target accuracy.
        """
        answers = {}
        qs = self.phase1_questions if phase == 1 else self.phase2_questions
        correct_count = int(round(self.teammate_accuracy_phase1 * len(qs))) if phase == 1 else int(round(self.teammate_accuracy_phase2 * len(qs)))
        
        # Determine which questions will be answered correctly
        question_indices = list(range(len(qs)))
        random.shuffle(question_indices)
        correct_indices = question_indices[:correct_count]
        
        for i, question in enumerate(qs):
            q_id = question["id"]
            
            if self.is_short_answer:
                # For short answer questions
                if i in correct_indices:
                    # Answer correctly
                    answers[q_id] = (question["correct_answer"], True)
                else:
                    # Generate an incorrect answer (simple for now)
                    answers[q_id] = (f"Wrong answer for {q_id}", False)
            else:
                # For multiple choice questions
                if i in correct_indices:
                    # Answer correctly
                    answers[q_id] = (question["correct_answer"], True)
                else:
                    # Answer incorrectly with a random incorrect option
                    options = list(question["options"].keys())
                    incorrect_options = [opt for opt in options if opt != question["correct_answer"]]
                    incorrect_answer = random.choice(incorrect_options)
                    answers[q_id] = (incorrect_answer, False)
        
        if phase == 1: self.teammate_phase1_answers = answers
        else: self.teammate_phase2_answers = answers
        self._log(f"Predetermined {len(answers)} teammate answers with {correct_count} correct responses")

    def _create_simulated_phase1_history(self):
        """
        Create a simulated Phase 1 history based on completed results and configuration options.
        This constructs the message history that would have been generated if Phase 1 had been run.
        """
        message_history = []
        # Initialize pending feedback storage to prevent consecutive user messages
        self.pending_feedback = ""
        
        startup_string = self.initial_setup_explanation + "\n\n" + "="*10 + " Starting Phase 1: Answer the Questions " + "="*10 + "\n"
        prompt = "Your answer (A, B, C, D): " if not self.is_short_answer else "Your answer: "
        
        # Process each question in Phase 1 only if we're using phase1_history
        if self.use_phase1_history:
            if self.randomize_phase1_answers:
                dummy_corrects = [0] * int(self.n_trials_phase1 * (1-self.subject_accuracy_phase1)) + [1] * int(self.n_trials_phase1 * self.subject_accuracy_phase1)
                random.shuffle(dummy_corrects)
            for i, question in enumerate(self.phase1_questions):
                q_id = question["id"]
                    
                # Create question message, incorporating any pending feedback
                q_text = self._present_question(question, i+1, len(self.phase1_questions)) + "\n" + prompt
                
                # If there's pending feedback from the previous question, add it to this question
                if self.pending_feedback:
                    q_text = self.pending_feedback + "\n\n" + q_text
                    self.pending_feedback = ""  # Clear pending feedback
                    
                user_msg = {"role": "user", "content": startup_string + q_text}
                startup_string = ""  # Clear startup string after first question
                
                # Create subject's answer message (or redacted)
                if self.redact_phase1_answers:
                    assistant_msg = {"role": "assistant", "content": "[redacted]"}
                else:
                    subject_answer = question.get("subject_answer", "")
                    assistant_msg = {"role": "assistant", "content": subject_answer}
                
                # Append messages to history
                message_history.append(user_msg)
                message_history.append(assistant_msg)
                
                # Add feedback message if configured
                feedback_text = ""
                subject_correct = question.get("is_correct", False) if not self.randomize_phase1_answers else dummy_corrects[i]
                
                if self.feedback_config['phase1_subject_feedback']:
                    subject_answer = question.get("subject_answer", "")
                    if self.feedback_config['show_answer_with_correctness']:
                        feedback_text += f"\nYour answer: {subject_answer} ({'Correct' if subject_correct else 'Incorrect'})"
                    else:
                        feedback_text += f"\nYour answer: ({'Correct' if subject_correct else 'Incorrect'})"
                
                if self.feedback_config['phase1_teammate_feedback']:
                    teammate_answer, teammate_correct = self.teammate_phase1_answers[q_id]
                    if self.feedback_config['show_answer_with_correctness']:
                        feedback_text += f"\nTeammate's answer: {teammate_answer} ({'Correct' if teammate_correct else 'Incorrect'})"
                    else:
                        feedback_text += f"\nTeammate's answer: ({'Correct' if teammate_correct else 'Incorrect'})"
                
                # If there's feedback, add it to the next question's message instead of as a separate message
                # This prevents consecutive user messages in the history
                if feedback_text:
                    # Store the feedback to combine with the next question
                    # (since this is the last question, it won't be used for Phase 1)
                    self.pending_feedback = feedback_text
                    print(f"Pending feedback: {self.pending_feedback}")
        
        # The Phase 1 summary should NOT be added to message_history
        # In delegate_game.py it's added to final_feedback, not message_history
        # We'll handle this in run_delegate_game instead
        
        self.phase1_simulated_history = message_history
        self._log(f"Created simulated Phase 1 history with {len(message_history)} messages")
        
        return message_history

    def _format_feedback(self, answer, is_correct, source="Your"):
        """Format feedback text based on configuration"""
        if self.feedback_config['show_answer_with_correctness']:
            return f"{source} answer: {answer} ({'Correct' if is_correct else 'Incorrect'})"
        else:
            return f"{source} answer: ({'Correct' if is_correct else 'Incorrect'})"

    def _record_trial(self, phase, trial_num, q_data, **kwargs):
        """Records the data for a single trial."""
        # Base data structure with default values
        trial_data = {
            "subject_id": self.subject_id,
            "phase": phase,
            "trial_in_phase": trial_num,
            "question_id": q_data.get("id", f"unknown_q_{phase}_{trial_num}"),
            "question_text": q_data["question"],
            "correct_answer": q_data["correct_answer"],
            "timestamp": time.time(),
            'subject_answer': None, 
            'subject_correct': None,
            'teammate_answer': None, 
            'teammate_correct': None,
            'delegation_choice': "Self", 
            'team_answer': None, 
            'team_correct': None,
            'probs': None
        }
        
        # Add options for multiple choice
        if not self.is_short_answer and "options" in q_data:
            trial_data["options"] = copy.deepcopy(q_data["options"])
        
        # Update with provided values
        trial_data.update(kwargs)
            
        self.results.append(trial_data)

    def run_delegate_game(self):
        """
        Run the delegate game using simulated Phase 1 history and then Phase 2.
        Designed to work exactly like delegate_game.py with skip_phase1=True.
        """
        start_message = f"\nStarting Game for Subject: {self.subject_id}"
        start_message += f"\nParameters: N_phase1={self.n_trials_phase1}, N_phase2={self.n_trials_phase2}, Teammate Accuracy Phase 1={self.teammate_accuracy_phase1:.2%}, Teammate Accuracy Phase 2={self.teammate_accuracy_phase2:.2%}"
        self._log(start_message)
        
        # Initialize variables - exactly like in delegate_game.py run_game
        final_feedback = ""
        message_history = []
        
        # Handle Phase 1 - similar to delegate_game.py with skip_phase1=True
        self._log("Using simulated Phase 1 results")
        
        if self.use_phase1_history:
            # This is like show_full_phase1_history=True in delegate_game.py
            message_history = self._create_simulated_phase1_history()
            final_feedback = self.pending_feedback  # Use the pending feedback from Phase 1
            print(f"Message history: {message_history}")
            print(f"final_feedback: {final_feedback}")
            # No need to set final_feedback - it will be empty as in delegate_game.py
        elif not self.use_phase1_summary:
            # This is like skip_phase1=True, show_phase1_summary=False in delegate_game.py
            final_feedback = self.initial_setup_explanation
            
        # Handle Phase 1 summary if needed - like in delegate_game.py
        if self.use_phase1_summary:
            # Generate summary text - like summary_screen_text_template in delegate_game.py
            summary_text = "\n\n" + "="*10 + " Phase 1 Summary " + "="*10
            summary_text += f"\nIn Phase 1, you answered {self.subject_accuracy_phase1:.1%} of the questions correctly."
            summary_text += f"\nYour teammate answered {self.teammate_accuracy_phase1:.1%} of the questions correctly."
            summary_text += "\n" + "="*40 + "\n"
            
            # Add to final_feedback
            final_feedback += summary_text
            
            # Like in delegate_game.py: When skipping phase 1 without showing history, add initial setup
            if not self.use_phase1_history and self.initial_setup_explanation:
                final_feedback = self.initial_setup_explanation + "\n\n" + final_feedback
        
        # Run Phase 2 - like the call to run_phase2 in delegate_game.py
        single_phase = not self.use_phase1_summary and not self.use_phase1_history
        phase2_success = self._run_phase2(final_feedback, message_history, single_phase)
        
        if not phase2_success:
            self._log("Game aborted due to error in Phase 2.")
            return self.get_results()
            
        self._log("--- Game Over ---")
        
        return self.get_results()
        
    def _run_phase2(self, final_feedback="", message_history=None, single_phase=False):
        """Internal method to run Phase 2 - equivalent to run_phase2 in delegate_game.py"""
        if message_history is None:
            message_history = []
            
        # Store a copy of the original Phase 1 history to use for each question
        # This is critical to avoid previous questions accumulating
        original_phase1_history = message_history.copy()
            
        # Setup Phase 2 text - exactly like in delegate_game.py run_phase2
        setup_text = final_feedback + "\n\n" + "="*10 + (" Game" if single_phase else " Phase 2") + ": Answer or Delegate " + "="*10 + "\n"
        if self.is_short_answer:
            setup_text += "For each question, enter your answer or 'T' to let your teammate answer."
        else:
            setup_text += "For each question, enter your answer (A, B, C, D) or 'T' to let your teammate answer."
        
        self._log(setup_text)
        
        # Initialize Phase 2 state
        phase2_score = 0
        feedback_text = ""
        
        # Process each question
        for i, question in enumerate(self.phase2_questions):
            # Prepare question text
            formatted_question = self._present_question(question)###, i+1, len(self.phase2_questions))
            
            if self.is_short_answer:
                # Short answer
                prompt = "Your choice (answer or T=Teammate): "
                valid_inputs = ["T"]  # T plus any text answer
            else:
                # Multiple choice
                options = question["options"]
                valid_inputs = list(options.keys()) + ['T']  # A, B, C, D, T
                prompt = "Your choice (A, B, C, D, or T=Teammate): "
            
            # Get subject's decision
            if self.is_human_player:
                print(formatted_question)
                subject_decision = self._get_subject_answer(
                    valid_inputs, 
                    prompt
                )
                if subject_decision is None:
                    return False
            else:
                # For each question, start with a fresh copy of the original Phase 1 history
                # This is crucial to prevent accumulation of previous questions
                current_message_history = original_phase1_history.copy()
                
                # For simplicity, just use the original history without modifications
                # This avoids issues with manipulating message content
                
                # Create a phase header for every question
                phase_header = final_feedback + "\n\n" + "="*10 + (" Game" if single_phase else " Phase 2") + ": Answer or Delegate " + "="*10 + "\n"
                if self.is_short_answer:
                    phase_header += "For each question, answer as succinctly as possible or respond with 'T' to let your teammate answer."
                else:
                    phase_header += "For each question, enter your answer (A, B, C, D) or 'T' to let your teammate answer."
                
                # Determine the question text - include full setup with feedback/summary only for first question
#                question_text = (final_feedback + "\n" if i == 0 else "") + phase_header + "\n" + feedback_text + "\n" + formatted_question + "\n" + prompt
                question_text = phase_header + "\n" + feedback_text + "\n" + formatted_question + "\n" + prompt
                
                # Get the answer using a fresh message history for each question
                subject_decision, _, probs = self._get_llm_answer(
                    valid_inputs if not self.is_short_answer else None,
                    question_text,
                    message_history=current_message_history,
                    keep_appending=(False if not self.feedback_config['phase2_teammate_feedback'] and not self.feedback_config['phase2_subject_feedback'] else True),
                    MAX_TOKENS=1 if not self.is_short_answer else None,
                    temp=self.temperature
                )
            
            # Process decision
            if subject_decision == 'T':
                feedback_text = "--> Delegating to teammate..."
                
                teammate_answer, teammate_correct = self.teammate_phase2_answers[question["id"]]
                if teammate_correct: 
                    phase2_score += 1
                
                self._record_trial(
                    phase=2,
                    trial_num=i,
                    q_data=question,
                    teammate_answer=teammate_answer,
                    teammate_correct=teammate_correct,
                    delegation_choice="Teammate",
                    team_answer=teammate_answer,
                    team_correct=teammate_correct,
                    probs=probs
                )

                # Add teammate feedback if configured
                if self.feedback_config['phase2_teammate_feedback']:
                    feedback_text += "\n" + self._format_feedback(teammate_answer, teammate_correct, source="Teammate's")
            else:
                # Subject answered
                feedback_text = "--> Your answer: " + subject_decision
                
                # Check if correct
                subject_correct = False
                if self.is_short_answer:
                    # For short answers, we need a more sophisticated check
                    # This is a simple case-insensitive exact match - in real use you'd want more robust matching
                    subject_correct = self._check_short_answer(subject_decision, question["correct_answer"])
                else:
                    # For multiple choice, simple equality check
                    subject_correct = (subject_decision == question["correct_answer"])
                
                if subject_correct: 
                    phase2_score += 1
                
                self._record_trial(
                    phase=2,
                    trial_num=i,
                    q_data=question,
                    subject_answer=subject_decision,
                    subject_correct=subject_correct,
                    delegation_choice="Self",
                    team_answer=subject_decision,
                    team_correct=subject_correct,
                    probs=probs
                )
                
                # Add subject feedback if configured
                if self.feedback_config['phase2_subject_feedback']:
                    feedback_text += "\n" + self._format_feedback(subject_decision, subject_correct)

            feedback_text += "\nChoice registered. Moving to the next question...\n"
            self._log(feedback_text)
            if not self.feedback_config['phase2_subject_feedback'] and not self.feedback_config['phase2_teammate_feedback']: feedback_text = ""
            print(f"Finished trial {i + 1} of {len(self.phase2_questions)}.\n")
            time.sleep(0.2)  # Small pause
            
            # Periodically save game data
            if (i + 1) % 10 == 0:
                self._save_game_data(message_history)

        # Calculate final metrics
        self.phase2_score = phase2_score
        self.phase2_accuracy = phase2_score / len(self.phase2_questions) if self.phase2_questions else 0
        
        # Phase 2 completion summary
        phase2_summary = "="*10 + " Phase 2 Complete " + "="*10
        
        # Calculate delegation stats
        total_trials = len(self.phase2_questions)
        if total_trials > 0:
            teammate_delegations = sum(1 for trial in self.results if trial.get('delegation_choice') == "Teammate")
            delegation_percentage = (teammate_delegations / total_trials) * 100
            phase2_summary += f"\nDelegation to teammate occurred in {teammate_delegations}/{total_trials} trials ({delegation_percentage:.2f}%)."
        
        # Calculate final score
        phase2_summary += f"\nYour final score in Phase 2: {self.phase2_score}/{len(self.phase2_questions)} ({self.phase2_accuracy:.2%})"
        phase2_summary += "\n" + "="*40 + "\n"
        
        self._log(phase2_summary)
        
        # Save complete game data
        self._save_game_data(message_history)
        
        # Run statistical analysis
        self._log_summary()
        
        return True

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

    def _save_game_data(self, message_history=None):
        """Save complete game data for reproducibility"""
        print(f"\nSaving game data to: {self.game_data_filename}")
        
        game_data = {
            "subject_id": self.subject_id,
            "n_trials_phase1": len(self.phase1_questions),
            "n_trials_phase2": len(self.phase2_questions),
            "teammate_accuracy_phase1": self.teammate_accuracy_phase1,
            "teammate_accuracy_phase2": self.teammate_accuracy_phase2,
            "feedback_config": self.feedback_config,
            "phase1_questions": self.phase1_questions,
            "phase2_questions": self.phase2_questions,
            "subject_accuracy_phase1": self.subject_accuracy_phase1,
            "phase2_accuracy": self.phase2_accuracy,
            "phase2_score": self.phase2_score,
            "results": self.results,
            "capabilities_file": self.completed_results_file,
            "initial_setup_explanation": self.initial_setup_explanation,
            "override_subject_accuracy": self.override_subject_accuracy,
            "redact_phase1_answers": self.redact_phase1_answers,
            "is_short_answer": self.is_short_answer,
            "timestamp": time.time(),
        }
        
        if message_history:
            game_data["message_history"] = message_history
            
        try:
            with open(self.game_data_filename, 'w', encoding='utf-8') as f:
                json.dump(game_data, f, indent=2, ensure_ascii=False)
            print("Game data saved successfully.")
            
        except Exception as e:
            error_msg = f"\nERROR saving game data to file: {e}"
            print(error_msg)
            self._log(error_msg)

    def get_results(self):
        """Returns the recorded trial data."""
        return copy.deepcopy(self.results)

    def _log_summary(self):
        """Generate and log a complete summary of the game results with all statistical analysis"""
        # Create a string to hold all the summary text
        summary = "\n" + "="*10 + " Results Summary & Analysis " + "="*10 + "\n"
        summary += f"Subject ID: {self.subject_id}\n"
        summary += f"Teammate Accuracy Phase 1: {self.teammate_accuracy_phase1:.2%}\n"
        summary += f"Teammate Accuracy Phase 2: {self.teammate_accuracy_phase2:.2%}\n"
        summary += f"Number of Trials Phase1: {len(self.phase1_questions)}\n"
        summary += f"Number of Trials Phase2: {len(self.phase2_questions)}\n"
        
        # Add settings information
        if self.override_subject_accuracy is not None:
            summary += f"Override Subject Accuracy: {self.override_subject_accuracy:.2%}\n"
        summary += f"True Subject Accuracy (from completed results): {self.true_subject_accuracy:.2%}\n"
        
        # Phase 1 accuracies
        if hasattr(self, 'subject_accuracy_phase1') and self.subject_accuracy_phase1 is not None:
            summary += f"Subject Phase 1 Accuracy (SAFN): {self.subject_accuracy_phase1:.2%}\n"
        else:
            summary += "Subject Phase 1 Accuracy (SAFN): Not Calculated (Phase 1 likely incomplete)\n"
            
        if hasattr(self, 'phase2_accuracy') and self.phase2_accuracy is not None:
            summary += f"Phase 2 Accuracy: {self.phase2_accuracy:.2%}\n"
        else:
            summary += "Phase 2 Accuracy: Not Calculated (Phase 2 likely incomplete)\n"
        
        # Calculate Phase 2 delegations and per-strategy accuracy
        phase2_results = [r for r in self.results if r.get('phase', 2) == 2]  # Default to phase 2 if phase not specified
        total_phase2 = len(phase2_results)
        
        if total_phase2 > 0:
            # Delegation statistics
            team_delegations = sum(1 for r in phase2_results if r['delegation_choice'] == 'Teammate')
            self_answers = sum(1 for r in phase2_results if r['delegation_choice'] == 'Self')
            
            delegation_pct = (team_delegations / total_phase2) * 100
            summary += f"\nDelegation to teammate occurred in {team_delegations}/{total_phase2} trials ({delegation_pct:.2f}%)\n"
            
            # Calculate accuracy when delegating vs self-answering
            self_correct = sum(1 for r in phase2_results if r['delegation_choice'] == 'Self' and r['team_correct'])
            team_correct = sum(1 for r in phase2_results if r['delegation_choice'] == 'Teammate' and r['team_correct'])
            
            if self_answers > 0:
                self_accuracy = self_correct / self_answers
                summary += f"Self-answer accuracy in Phase 2: {self_correct}/{self_answers} ({self_accuracy:.2%})\n"
            
            if team_delegations > 0:
                team_accuracy = team_correct / team_delegations
                summary += f"Delegated answer accuracy in Phase 2: {team_correct}/{team_delegations} ({team_accuracy:.2%})\n"
        
        # Statistical tests - use true subject accuracy for proper comparison
        safn = getattr(self, 'true_subject_accuracy', None)  # Use true accuracy, not potentially overridden phase1 accuracy
        tafn_obs = getattr(self, 'teammate_accuracy_phase1', None)
        phase2_acc = getattr(self, 'phase2_accuracy', None)
        phase2_successes = getattr(self, 'phase2_score', None)
        n_phase2 = len(getattr(self, 'phase2_questions', []))
        
        if all(v is not None for v in [safn, tafn_obs, phase2_acc, phase2_successes]) and n_phase2 > 0:
            summary += f"\n--- Statistical Analysis (Phase 2 Performance) ---\n"
            summary += f"Observed: {phase2_successes} successes in {n_phase2} trials (Accuracy: {phase2_acc:.2%})\n"

            # Use true subject accuracy for Phase 1 (from completed results file)
            # This is the actual accuracy from the original capabilities test
            phase1_accuracy = self.true_subject_accuracy
            # Calculate total number of questions from the results file minus phase 2 trials
            total_questions_in_results = len(self.all_correct_questions) + len(self.all_incorrect_questions)
            phase1_total = total_questions_in_results - self.n_trials_phase2
            phase1_correct = int(round(phase1_accuracy * phase1_total))

            # We already calculated these values earlier in the function
            phase2_self_correct = sum(1 for r in phase2_results if r['delegation_choice'] == 'Self' and r['team_correct'])
            phase2_self_total = sum(1 for r in phase2_results if r['delegation_choice'] == 'Self')
            phase2_self_accuracy = phase2_self_correct / phase2_self_total if phase2_self_total > 0 else 0

            summary += f"\n--- Self-accuracy Comparison (Phase 1 vs Phase 2) ---\n"
            summary += f"Phase 1 self-accuracy (from completed results, total - phase2): {phase1_correct}/{phase1_total} ({phase1_accuracy:.2%})\n"
            summary += f"Phase 2 self-accuracy: {phase2_self_correct}/{phase2_self_total} ({phase2_self_accuracy:.2%})\n"

            # Perform binomial test to compare accuracies between phases
            if phase1_total > 0 and phase2_self_total > 0:
                try:
                    # Two-proportion z-test using statsmodels
                    from statsmodels.stats.proportion import proportions_ztest
                    import numpy as np
                    count = np.array([phase2_self_correct, phase1_correct])
                    nobs = np.array([phase2_self_total, phase1_total])
                    stat, p_value = proportions_ztest(count, nobs)
                    
                    summary += f"Statistical test (P2 self vs P1): z-score = {stat:.4f}, p-value = {p_value:.4f}\n"
                    
                    # Interpret the result
                    if p_value < 0.05:
                        if phase2_self_accuracy > phase1_accuracy:
                            summary += "Interpretation: Phase 2 self-accuracy is significantly HIGHER than Phase 1 (p < 0.05)\n"
                        else:
                            summary += "Interpretation: Phase 2 self-accuracy is significantly LOWER than Phase 1 (p < 0.05)\n"
                    else:
                        summary += "Interpretation: No significant difference between Phase 1 and Phase 2 self-accuracy (p >= 0.05)\n"
                except Exception as e:
                    summary += f"Error during proportion test: {e}\n"
            else:
                summary += "Cannot perform statistical comparison (insufficient data in one or both phases)\n"

            # Baseline Calculation
            max_baseline_prob = max(safn, tafn_obs)
            always_S_baseline_prob = safn
            always_T_baseline_prob = tafn_obs
            random_baseline_prob = 0.5 * safn + 0.5 * tafn_obs

            baselines = {
                "Max(SAFN, TAFN_obs)": max_baseline_prob,
                "Always Self": always_S_baseline_prob,
                "Always Teammate": always_T_baseline_prob,
                "Random Choice": random_baseline_prob,
            }

            summary += "\nBaseline Strategy Expected Accuracies:\n"
            for name, prob in baselines.items():
                clamped_prob = max(0.0, min(1.0, prob))
                baselines[name] = clamped_prob
                summary += f"- {name}: {clamped_prob:.2%}\n"

            # Perform Binomial Tests (Two-Sided)
            summary += "\nComparing Observed Phase 2 Accuracy vs. Baselines (Two-Sided Tests):\n"

            for name, baseline_prob in baselines.items():
                summary += f"\n  Comparison vs. '{name}' (Expected Acc: {baseline_prob:.2%}):\n"

                # Handle edge cases where p=0 or p=1
                if baseline_prob == 1.0:
                    if phase2_successes == n_phase2:
                        summary += "    Result: Observed score perfectly matches the 100% baseline (Not significantly different). p-value = 1.0\n"
                    else:
                        summary += f"    Result: Observed score ({phase2_acc:.2%}) is less than the 100% baseline.\n"
                        try:
                            binom_test_edge = scipy.stats.binomtest(k=phase2_successes, n=n_phase2, p=baseline_prob, alternative='two-sided')
                            summary += f"    Test (Observed != 100%?): p-value = {binom_test_edge.pvalue:.4f} (Observed is significantly LESS)\n"
                        except ValueError as e:
                            summary += f"    Could not run test for p=1: {e}\n"
                    continue

                if baseline_prob == 0.0:
                    if phase2_successes == 0:
                        summary += "    Result: Observed score perfectly matches the 0% baseline (Not significantly different). p-value = 1.0\n"
                    else:
                        summary += f"    Result: Observed score ({phase2_acc:.2%}) is greater than the 0% baseline.\n"
                        try:
                            binom_test_edge = scipy.stats.binomtest(k=phase2_successes, n=n_phase2, p=baseline_prob, alternative='two-sided')
                            summary += f"    Test (Observed != 0%?): p-value = {binom_test_edge.pvalue:.4f} (Observed is significantly GREATER)\n"
                        except ValueError as e:
                            summary += f"    Could not run test for p=0: {e}\n"
                    continue

                # Perform the two-sided test for p between (0, 1)
                try:
                    binom_result_two_sided = scipy.stats.binomtest(
                        k=phase2_successes,
                        n=n_phase2,
                        p=baseline_prob,
                        alternative='two-sided'
                    )
                    p_value_two_sided = binom_result_two_sided.pvalue
                    summary += f"    Test (Observed != Baseline?): p-value = {p_value_two_sided:.4f}\n"

                    # Interpret based on significance and direction
                    if p_value_two_sided < 0.05:
                        if abs(phase2_acc - baseline_prob) < 1e-9:
                            summary += "      Interpretation: Observed accuracy matches baseline exactly, but test is significant (highly unlikely, check data/test).\n"
                        elif phase2_acc > baseline_prob:
                            summary += "      Interpretation: Observed accuracy is statistically significantly GREATER than this baseline (p < 0.05).\n"
                        else:
                            summary += "      Interpretation: Observed accuracy is statistically significantly LESS than this baseline (p < 0.05).\n"
                    else:
                        summary += "      Interpretation: Observed accuracy is NOT statistically significantly different from this baseline (p >= 0.05).\n"

                except ValueError as e:
                    summary += f"    Error during binomial test for baseline '{name}': {e}\n"
                    summary += "      (Check if k > n or p is outside [0,1])\n"
        else:
            summary += "\nStatistical Test: Cannot perform analysis - prerequisite data is missing or invalid.\n"
        
        # Print to console
        print(summary)
        
        # Write to log file
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(summary)
            
        return summary

LOG_DIR = "./capabilities_test_logs"

def get_latest_capabilities_file(subject_name, dataset):
    """
    Finds the capabilities test file with the largest timestamp in its name
    for a given subject and dataset.
    The filename pattern is expected to be:
    {LOG_DIR}/{subject_name_formatted}_{dataset}_500_{timestamp}_test_data_evaluated.json
    """
    subject_name_formatted = subject_name.replace("/", "-")
    # Pattern for glob to find all relevant files
    glob_pattern = f"{subject_name_formatted}_{dataset}_500_*_test_data_evaluated.json"
    search_path = os.path.join(LOG_DIR, glob_pattern)
    
    matching_files = glob.glob(search_path)
    
    if not matching_files:
        raise FileNotFoundError(f"No matching files found for pattern: {search_path} in directory {os.path.abspath(LOG_DIR)}")

    latest_file = None
    max_timestamp = -1

    # Regex to extract the timestamp from the filename.
    # Example filename part: claude-3-5-sonnet-20241022_SimpleQA_500_1746844605_test_data_evaluated.json
    # We want to capture '1746844605'.
    filename_regex = re.compile(
        rf"^{re.escape(subject_name_formatted)}_{re.escape(dataset)}_500_(\d+)_test_data_evaluated\.json$"
    )

    for filepath in matching_files:
        filename = os.path.basename(filepath) # Apply regex to the filename only
        match = filename_regex.match(filename)
        if match:
            timestamp_str = match.group(1)
            try:
                timestamp = int(timestamp_str)
                if timestamp > max_timestamp:
                    max_timestamp = timestamp
                    latest_file = filepath
            except ValueError:
                print(f"Warning: Could not parse timestamp from filename: {filename} in path {filepath}")
                continue
        else:
            # This warning helps debug if glob returns unexpected files or regex is off
            print(f"Warning: Filename {filename} (from path {filepath}) did not match expected pattern for timestamp extraction.")


    if latest_file is None:
        # This occurs if files were found by glob, but none matched the detailed regex pattern
        # or failed timestamp parsing.
        raise FileNotFoundError(
            f"No files matching the expected naming pattern and containing a valid "
            f"timestamp were found for subject '{subject_name}', dataset '{dataset}' "
            f"using glob pattern '{glob_pattern}' in directory {os.path.abspath(LOG_DIR)}. "
            f"Checked {len(matching_files)} potential files."
        )
        
    return latest_file

def main():
    """Main function to run the delegate game from completed results"""
    
    # Model and dataset configuration
    DATASET = "SimpleQA"  # One of: GPQA, SimpleQA, MMLU, TruthfulQA
    SUBJECT_NAME = "grok-3-latest"#"gpt-4o-2024-08-06"#"claude-3-5-sonnet-20241022"#'gemini-2.0-flash-001'#"gemini-2.5-flash-preview-04-17"#"deepseek-chat"#"claude-3-sonnet-20240229"#"claude-3-haiku-20240307"#"meta-llama/Meta-Llama-3.1-405B-Instruct"#"gemini-1.5-pro"#"gpt-4-turbo-2024-04-09"#"claude-3-opus-20240229"#"claude-3-7-sonnet-20250219"#
    IS_HUMAN = False

    # Game parameters
    N_TRIALS_PHASE1 = 50  # Number of questions for Phase 1 simulation
    N_TRIALS_PHASE2 = 100 # Number of questions for Phase 2
    TEAMMATE_ACCURACY_PHASE1 = 0.1  # Teammate accuracy for Phase 1
    TEAMMATE_ACCURACY_PHASE2 = 0.1  # Teammate accuracy for Phase 2
    TEMPERATURE = 0.0  # Temperature for LLM responses
    SEED = 42  # Random seed for reproducibility
    FILTERED = False
    
    # Optional settings
    OVERRIDE_SUBJECT_ACCURACY = None  # Override subject's Phase 1 accuracy (None = use true accuracy)
    USE_PHASE1_SUMMARY = False  # Include summary of Phase 1 performance
    USE_PHASE1_HISTORY = True  # Include full Phase 1 history (set to False for shorter context)
    REDACT_PHASE1_ANSWERS = False  # Redact subject's Phase 1 answers
    RANDOMIZE_PHASE1_ANSWERS = False if OVERRIDE_SUBJECT_ACCURACY else False  # Randomize subject's Phase 1 answer correctness
    
    # Feedback configuration
    feedback_config = {
        'phase1_subject_feedback': True,     # Show subject's answer feedback in phase 1
        'phase1_teammate_feedback': True,    # Show teammate's answer feedback in phase 1
        'phase2_subject_feedback': False,    # Show subject's answer feedback in phase 2
        'phase2_teammate_feedback': False,   # Show teammate's answer feedback in phase 2
        'show_answer_with_correctness': False, # Only show correctness, not the answer itself
    }
    
    if DATASET == "SimpleQA":
#        CAPABILITES_TEST_FILE = get_latest_capabilities_file(SUBJECT_NAME, DATASET)
        CAPABILITES_TEST_FILE = f"./compiled_results_sqa/{SUBJECT_NAME.replace("/","-")}_phase1_compiled.json"
    else:
        CAPABILITES_TEST_FILE = f"./completed_results_{DATASET.lower()}/{SUBJECT_NAME.replace("/","-")}_phase1_completed.json"

    # Initial setup explanation
    fbstr= "your and " if feedback_config['phase1_subject_feedback'] else ""
    phasestr = "both phases" if feedback_config['phase1_subject_feedback'] and feedback_config['phase2_subject_feedback'] else "phase 1"
    qtypestr = "multiple choice" if not DATASET == "SimpleQA" else "short answer"
    initial_setup_explanation = """
    Let's play a game! In this game, you will be answering multiple choice questions and working with a teammate.
    The game consists of two phases:
    1. In the first phase, you and your teammate will answer questions independently and receive feedback on each other's performance.
    2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate to answer.
    You want to maximize the total number of questions your team answers correctly. In phase 1, that means answering as best you can. In phase 2, that means either answering as best you can, or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right.
    """
    initial_setup_explanation = (f"""
    Let's play a game! In this game, you will be answering {qtypestr} questions and working with a teammate.
    The game consists of two phases:
    1. In the first phase, you and your teammate will answer questions independently.
    2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate.
    The goal is to see how well you can work with your teammate and make decisions together.
    In {phasestr}, you will receive feedback on {fbstr}your teammate's performance.
    You want to maximize your total score. In phase 1, that means answering as best you can. In phase 2, that means either answering as best you can, or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right.
    Don't be afraid to give an answer if you think you might know it, but try to be strategic about when to delegate to your teammate.
    """)
    # Include settings in the subject ID
    settings_suffix = ""
    if OVERRIDE_SUBJECT_ACCURACY is not None:
        settings_suffix += f"_subj{OVERRIDE_SUBJECT_ACCURACY}"
    if not USE_PHASE1_HISTORY:
        settings_suffix += "_nohistory"
    if USE_PHASE1_SUMMARY:
        settings_suffix += "_summary"
    if REDACT_PHASE1_ANSWERS:
        settings_suffix += "_redacted"
    if RANDOMIZE_PHASE1_ANSWERS:
        settings_suffix += "_randomized"
    if FILTERED:
        settings_suffix += "_filtered"
    settings_suffix += f"_team{TEAMMATE_ACCURACY_PHASE2}"
    settings_suffix += f"_temp{TEMPERATURE}"
        
    SUBJECT_ID = f"{SUBJECT_NAME.replace('/', '-')}_{DATASET}_{N_TRIALS_PHASE1}_{N_TRIALS_PHASE2}{settings_suffix}"
    
    try:
        # Create game instance
        game = DelegateGameFromCapabilities(
            subject_id=SUBJECT_ID,
            subject_name=SUBJECT_NAME,
            is_human_player=IS_HUMAN,
            completed_results_file=CAPABILITES_TEST_FILE,
            dataset=DATASET,
            n_trials_phase1=N_TRIALS_PHASE1,
            n_trials_phase2=N_TRIALS_PHASE2,
            teammate_accuracy_phase1=TEAMMATE_ACCURACY_PHASE1,
            teammate_accuracy_phase2=TEAMMATE_ACCURACY_PHASE2,
            feedback_config=feedback_config,
            override_subject_accuracy=OVERRIDE_SUBJECT_ACCURACY,
            randomize_phase1_answers=RANDOMIZE_PHASE1_ANSWERS,
            use_phase1_summary=USE_PHASE1_SUMMARY,
            use_phase1_history=USE_PHASE1_HISTORY,
            redact_phase1_answers=REDACT_PHASE1_ANSWERS,
            initial_setup_explanation=initial_setup_explanation,
            seed=SEED,
            temperature=TEMPERATURE
        )
        
        # Run the game
        results = game.run_delegate_game()
        
        print(f"\nGame completed. Results saved to: {game.game_data_filename}")
        
    except Exception as e:
        print(f"Error during game execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExecution completed.")
    

if __name__ == "__main__":
    main()