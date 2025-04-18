# -*- coding: utf-8 -*-
"""
Complete script for a psychological game experiment testing
self/other modeling using trivia questions.

Features:
- Loads questions from Hugging Face's TruthfulQA or GPQA dataset.
- Ensures unique questions across the experiment.
- Two phases: Modeling (Phase 1) and Decision (Phase 2).
- Phase 1: Subject answers questions, gets feedback on own and teammate's performance.
- Phase 2: Subject chooses to answer (A-D) or delegate ('T') for each question.
- Simulates a teammate with configurable accuracy.
- Records detailed trial data.
- Calculates Phase 1 (SAFN, TAFN_obs) and Phase 2 accuracies.
- Performs binomial test comparing Phase 2 accuracy to max(SAFN, TAFN_obs).
- Includes integration with Large Language Models (LLMs).
- Saves results to a JSON file.
"""

import anthropic
import random
import time
import copy
import json
import sys
import os
from datasets import load_dataset
import scipy.stats
import collections
from dotenv import load_dotenv
load_dotenv()

# API Configuration
hf_token = os.environ.get("HF_TOKEN")
provider = 'Anthropic'
api_key = os.environ.get("ANTHROPIC_SPAR_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

initial_setup_explanation = """
Let's play a game! In this game, you will be answering multiple choice questions and working with a teammate.
The game consists of two phases:
1. In the first phase, you and your teammate will answer questions independently and receive feedback on your performance.
2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate.
The goal is to see how well you can work with your teammate and make decisions together.
You want to maximize your total score. In phase 1, that means answering as best you can. In phase 2, that means either answering as best you can, or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right.
"""

MAX_ATTEMPTS = 5

# Configure stdout for UTF-8 if necessary
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
        print("Note: Reconfigured stdout/stderr to UTF-8.")
    except Exception as e:
        print(f"Warning: Could not reconfigure stdout/stderr to UTF-8: {e}")


# --- Data Loading and Formatting ---
def load_and_format_gpqa(num_questions_needed, hf_token=None, split="train", filter_id="recgCB0HSVt2IslDN"):
    """
    Loads the GPQA dataset and formats questions into the A-D multiple-choice format.
    """
    print(f"Attempting to load GPQA ({split} split)...")
    dataset_name = "Idavidrein/gpqa"
    config_name = "gpqa_main"
    try:
        dataset = load_dataset(dataset_name, config_name, split=split, token=hf_token, trust_remote_code=True)
        print("GPQA Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading GPQA dataset '{dataset_name}' ({config_name}, {split}): {e}")
        print("Please ensure you have the 'datasets' library installed, an internet connection,")
        print(f"and potentially a valid Hugging Face token if required (passed as hf_token).")
        return None

    formatted_questions = []
    question_ids_added = set()
    required_fields = ['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'Record ID']

    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)

    print(f"Formatting {num_questions_needed} questions from GPQA...")
    skipped_filter = 0
    skipped_missing_data = 0

    for idx in dataset_indices:
        if len(formatted_questions) >= num_questions_needed:
            break

        item = dataset[idx]

        # Check if all required fields exist and are not None/empty
        if not all(item.get(field) for field in required_fields):
            skipped_missing_data += 1
            continue

        record_id = item['Record ID']

        # Apply filtering
        if record_id == filter_id:
            skipped_filter += 1
            continue

        # Check if ID already added
        if record_id in question_ids_added:
            continue

        # Gather options
        correct_answer_text = item['Correct Answer'].strip()
        incorrect_answers_text = [
            item['Incorrect Answer 1'].strip(),
            item['Incorrect Answer 2'].strip(),
            item['Incorrect Answer 3'].strip()
        ]
        if len(correct_answer_text) == 0 or any(len(ans) == 0 for ans in incorrect_answers_text):
            continue

        # Create the pool of 4 options and shuffle
        options_list = [correct_answer_text] + incorrect_answers_text
        random.shuffle(options_list)

        # Assign labels (A-D) and find the correct one
        options_dict = {}
        correct_label = None
        labels = ["A", "B", "C", "D"]
        
        for i, option_text in enumerate(options_list):
            label = labels[i]
            options_dict[label] = option_text
            if option_text == correct_answer_text:
                correct_label = label

        # Ensure the correct answer was found
        if correct_label is None:
            print(f"Error processing GPQA Record ID {record_id}: Correct answer text not found in options list after shuffling. Skipping.")
            continue

        # Create the formatted question
        formatted_q = {
            "id": f"gpqa_{split}_{record_id}",
            "question": item['Question'],
            "options": options_dict,
            "correct_answer": correct_label
        }
        formatted_questions.append(formatted_q)
        question_ids_added.add(record_id)
        
    return formatted_questions

def load_and_format_truthfulqa(num_questions_needed, split="validation"):
    """
    Loads the TruthfulQA dataset and formats questions into the A-D multiple-choice format.
    """
    print(f"Attempting to load TruthfulQA ({split} split)...")
    try:
        dataset = load_dataset("truthful_qa", "generation", split=split, trust_remote_code=True)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading TruthfulQA dataset: {e}")
        print("Please ensure you have the 'datasets' library installed (`pip install datasets`)")
        print("and an internet connection. You might also need `trust_remote_code=True`.")
        return None

    formatted_questions = []
    attempts = 0
    max_attempts = len(dataset) * 2  # Safety break

    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)

    question_ids_added = set()  # Keep track of IDs to ensure uniqueness

    print(f"Formatting {num_questions_needed} questions...")
    for idx in dataset_indices:
        if len(formatted_questions) >= num_questions_needed:
            break

        if attempts > max_attempts:
            print("Warning: Reached max attempts trying to format questions.")
            break
        attempts += 1

        item = dataset[idx]
        potential_id = f"tqa_{split}_{idx}"

        # Skip if this exact item index was somehow processed before or has no ID
        if potential_id in question_ids_added:
            continue

        question_text = item.get('question')
        best_answer = item.get('best_answer')
        if len(best_answer.strip()) == 0:
            continue
        incorrect_answers = item.get('incorrect_answers')

        # Basic validation of required fields
        if not all([question_text, best_answer, incorrect_answers]):
            continue

        # Need at least 3 incorrect answers to form 4 options
        if not isinstance(incorrect_answers, list) or len(incorrect_answers) < 3:
            continue

        # Ensure best_answer is not accidentally in the chosen incorrect list
        possible_incorrect = [ans for ans in incorrect_answers if ans != best_answer and len(ans.strip()) > 0]
        if len(possible_incorrect) < 3:
            continue

        # Select 3 distinct incorrect answers
        try:
            chosen_incorrect = random.sample(possible_incorrect, 3)
        except ValueError:
            continue

        # Create the pool of options and shuffle
        options_list = [best_answer] + chosen_incorrect
        random.shuffle(options_list)

        # Assign labels and find the correct one
        options_dict = {}
        correct_label = None
        labels = ["A", "B", "C", "D"]
        for i, option_text in enumerate(options_list):
            label = labels[i]
            options_dict[label] = option_text
            if option_text == best_answer:
                correct_label = label

        # This should theoretically not happen if best_answer is in options_list
        if correct_label is None:
            print(f"Error processing index {idx}: Correct label not found for question: {question_text}")
            continue

        # Create the formatted dictionary
        formatted_q = {
            "id": potential_id,
            "question": question_text,
            "options": options_dict,
            "correct_answer": correct_label
        }
        formatted_questions.append(formatted_q)
        question_ids_added.add(potential_id)

    if len(formatted_questions) < num_questions_needed:
        print(f"Warning: Only able to format {len(formatted_questions)} unique questions,")
        print(f"         but {num_questions_needed} were requested.")
        print("         Consider using a different split (e.g., 'train') or reducing N.")

    print(f"Successfully formatted {len(formatted_questions)} unique questions from TruthfulQA.")
    return formatted_questions


# --- Core Game Logic ---
class PsychGame:
    """
    Manages the psychological experiment game flow.
    """
    def __init__(self, subject_id, questions, n_trials_per_phase, teammate_accuracy, 
                 feedback_config=None):
        """
        Initializes the game instance.

        Args:
            subject_id (str): Identifier for the current subject/session.
            questions (list): A list of formatted question dictionaries.
            n_trials_per_phase (int): Number of trials (N) in each phase.
            teammate_accuracy (float): The teammate's target accuracy (probability, 0.0 to 1.0).
            feedback_config (dict): Configuration for feedback options.
        """
        # Parameter validation
        if not questions:
            raise ValueError("No questions provided to the game.")
        if not (0.0 <= teammate_accuracy <= 1.0):
            raise ValueError("Teammate accuracy must be between 0.0 and 1.0")
        if not isinstance(n_trials_per_phase, int) or n_trials_per_phase <= 0:
            raise ValueError("Number of trials per phase must be a positive integer.")

        # Basic parameters
        self.subject_id = subject_id
        self.n_trials_per_phase = n_trials_per_phase
        self.teammate_accuracy_target = teammate_accuracy
        
        # Default feedback configuration
        self.feedback_config = {
            'phase1_subject_feedback': True,      # Show subject's answer feedback in phase 1
            'phase1_teammate_feedback': True,     # Show teammate's answer feedback in phase 1
            'phase2_subject_feedback': True,      # Show subject's answer feedback in phase 2
            'phase2_teammate_feedback': True,     # Show teammate's answer feedback in phase 2
            'show_answer_with_correctness': True,     
        }
        
        # Override defaults with provided config
        if feedback_config:
            self.feedback_config.update(feedback_config)

        # Calculate required question count
        total_questions_needed = n_trials_per_phase * 2

        # Check for sufficient questions
        if len(questions) < total_questions_needed:
            raise ValueError(f"Not enough questions provided ({len(questions)}) for the required {total_questions_needed}.")

        # Check uniqueness
        unique_q_ids = {q['id'] for q in questions}
        if len(unique_q_ids) < total_questions_needed:
            print(f"Warning: Input question list has only {len(unique_q_ids)} unique IDs, but {total_questions_needed} are required.")

        # Select exactly N*2 questions
        self.game_questions = questions[:total_questions_needed]

        # Final uniqueness check
        selected_q_ids = [q['id'] for q in self.game_questions]
        if len(selected_q_ids) != len(set(selected_q_ids)):
            duplicate_ids = [item for item, count in collections.Counter(selected_q_ids).items() if count > 1]
            print(f"ERROR: Duplicate question IDs detected within the final selected game questions! Duplicates: {duplicate_ids}")
            raise ValueError("Internal error: Duplicate question IDs found in the selected game set. Cannot proceed.")

        # Split questions into phases
        self.phase1_questions = self.game_questions[:n_trials_per_phase]
        self.phase2_questions = self.game_questions[n_trials_per_phase:]

        # Pre-determine teammate's answers for phase 1 to ensure exact probability match
        self.teammate_phase1_answers = self._predetermine_teammate_answers(self.phase1_questions)

        # Initialize state and results storage
        self.results = []
        self.current_phase = 0
        self.current_trial_in_phase = 0
        self.subject_accuracy_phase1 = None
        self.teammate_accuracy_phase1_observed = None
        self.phase2_score = None
        self.phase2_accuracy = None
        self.is_human_player = True  # Default to human input
        
        os.makedirs('./game_logs', exist_ok=True)
        timestamp = int(time.time())
        self.log_base_name = f"./game_logs/{subject_id}_{timestamp}"
        self.log_filename = f"{self.log_base_name}.log"
        self.results_filename = f"{self.log_base_name}.json"
        
        # Initialize log file
        with open(self.log_filename, 'w', encoding='utf-8') as f:
            f.write(f"Game Log for Subject: {subject_id}\n")
            f.write(f"Parameters: N={n_trials_per_phase}, Target Teammate Accuracy={teammate_accuracy:.2%}\n")
            f.write(f"Feedback Config: {json.dumps(self.feedback_config, indent=2)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Results file: {self.results_filename}\n\n")

    def _log(self, message):
        """Write to the log file and print to console"""
        print(message)
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def _predetermine_teammate_answers(self, questions):
        """
        Pre-determines teammate answers to ensure exact match with target accuracy.
        
        Returns:
            dict: Mapping of question IDs to (answer, is_correct) tuples
        """
        answers = {}
        correct_count = int(round(self.teammate_accuracy_target * len(questions)))
        
        # Determine which questions will be answered correctly
        question_indices = list(range(len(questions)))
        random.shuffle(question_indices)
        correct_indices = question_indices[:correct_count]
        
        for i, question in enumerate(questions):
            q_id = question["id"]
            if i in correct_indices:
                # Answer correctly
                answers[q_id] = (question["correct_answer"], True)
            else:
                # Answer incorrectly
                options = list(question["options"].keys())
                incorrect_options = [opt for opt in options if opt != question["correct_answer"]]
                if incorrect_options:
                    incorrect_answer = random.choice(incorrect_options)
                    answers[q_id] = (incorrect_answer, False)
                else:
                    # Fallback if no incorrect options exist (shouldn't happen)
                    answers[q_id] = (question["correct_answer"], True)
        
        return answers

    def _get_llm_answer(self, options, q_text, message_history):
        """Gets answer from LLM model"""
        user_msg = {"role": "user", "content": q_text}
        message_history.append(user_msg)
        resp = ""
        system_msg = "Output only the letter of your choice: " + ", ".join(options[:-1]) + f", or {options[-1]}.\n"
        #print(f"system_msg: {system_msg}")
        #print(f"message_history: {message_history}")        
        
        for attempt in range(MAX_ATTEMPTS):
            try:
                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2,
                    temperature=0.0 + attempt * 0.2,
                    system=system_msg,
                    messages=message_history
                )
                resp = message.content[0].text.strip().upper()
                if resp in options:
                    break
            except Exception as e:
                self._log(f"Error: {e}")
                
        asst_msg = {"role": "assistant", "content": resp}
        message_history.append(asst_msg)

        if resp not in options:
            self._log(f"Failed to get valid response for text: {q_text}; response: {resp}")

        return resp, message_history
        
    def _get_subject_answer(self, question_data):
        """Gets the subject's multiple-choice answer (A-D). Used ONLY in Phase 1."""
        options = question_data["options"]
        prompt = "Your answer (A, B, C, D): "

        while True:
            try:
                answer = input(prompt).strip().upper()
                if answer in options:
                    return answer
                else:
                    print("Invalid input. Please enter A, B, C, or D.")
            except EOFError:
                print("\nInput stream closed unexpectedly. Exiting trial.")
                return None

    def _get_teammate_answer(self, question_data, phase=1):
        """
        Returns the teammate's answer based on pre-determined answers (phase 1)
        or simulated on-the-fly (phase 2).
        """
        if phase == 1:
            # Use pre-determined answer for phase 1
            return self.teammate_phase1_answers[question_data["id"]]
        else:
            # Simulate for phase 2
            correct_answer = question_data["correct_answer"]
            options = question_data["options"]
            possible_answers = list(options.keys())
            
            # Determine if the teammate will be correct
            is_correct = random.random() < self.teammate_accuracy_target
            
            if is_correct:
                return correct_answer, True
            else:
                # Choose a random incorrect answer
                incorrect_options = [opt for opt in possible_answers if opt != correct_answer]
                if not incorrect_options:
                    return correct_answer, True  # Can't be incorrect
                chosen_incorrect_answer = random.choice(incorrect_options)
                return chosen_incorrect_answer, False

    def _present_question(self, question_data):
        """Formats a question for display"""
        formatted_question = ""
        formatted_question += "-" * 30 + "\n"
        formatted_question += f"Question {self.current_trial_in_phase + 1}:\n"
        formatted_question += question_data["question"] + "\n"
        formatted_question += "-" * 10 + "\n"
        for key, value in question_data["options"].items():
            formatted_question += f"  {key}: {value}\n"
        
        formatted_question += "-" * 30
        return formatted_question

    def _record_trial(self, phase, trial_num, q_data, **kwargs):
        """Records the data for a single trial."""
        # Base data structure with default values
        trial_data = {
            "subject_id": self.subject_id,
            "phase": phase,
            "trial_in_phase": trial_num,
            "teammate_accuracy_target": self.teammate_accuracy_target,
            "question_id": q_data.get("id", f"unknown_q_{phase}_{trial_num}"),
            "question_text": q_data["question"],
            "options": copy.deepcopy(q_data["options"]),
            "correct_answer_label": q_data["correct_answer"],
            "correct_answer_text": q_data["options"].get(q_data["correct_answer"], "ERROR: Correct answer key invalid"),
            "timestamp": time.time(),
            # Default values for all possible fields
            'subject_answer_p1': None, 'subject_correct_p1': None,
            'teammate_answer_p1': None, 'teammate_correct_p1': None,
            'delegation_choice': None, 'chosen_source': None,
            'final_answer': None, 'final_answer_text': None, 'final_correct': None
        }
        
        # Update with provided values
        trial_data.update(kwargs)
        
        # Derive final_answer_text if needed
        if trial_data['final_answer'] and trial_data['final_answer'] in trial_data['options']:
            trial_data['final_answer_text'] = trial_data['options'][trial_data['final_answer']]
            
        self.results.append(trial_data)

    def _format_feedback(self, answer, is_correct, source="Your"):
        """Format feedback text based on configuration"""
        if self.feedback_config['show_answer_with_correctness']:
            return f"{source} answer: {answer} ({'Correct' if is_correct else 'Incorrect'})"
        else:
            return f"{source} answer: ({'Correct' if is_correct else 'Incorrect'})"

    def run_phase1(self):
        """Runs Phase 1: Modeling."""
        setup_text = initial_setup_explanation
        setup_text += "\n\n" + "="*10 + " Starting Phase 1: Answer the Questions " + "="*10
        self._log(setup_text)
        
        self.current_phase = 1
        subject_correct_count = 0
        teammate_correct_count = 0

        message_history = []
        feedback_text = ""
        
        for i, q_data in enumerate(self.phase1_questions):
            self.current_trial_in_phase = i
            formatted_question = self._present_question(q_data)

            # Get Subject Answer
            if self.is_human_player:
                print(formatted_question)
                subject_answer = self._get_subject_answer(q_data)
                if subject_answer is None:
                    return False
            else:
                subject_answer, message_history = self._get_llm_answer(
                    list(q_data["options"].keys()), 
                    (setup_text if i==0 else "") + feedback_text + "\n" + formatted_question + "\n", 
                    message_history
                )

            subject_correct = (subject_answer == q_data["correct_answer"])
            if subject_correct:
                subject_correct_count += 1

            # Get Teammate Answer (using pre-determined answers)
            teammate_answer, teammate_correct = self._get_teammate_answer(q_data, phase=1)
            if teammate_correct:
                teammate_correct_count += 1

            # Prepare Feedback
            feedback_text = ""
            if self.feedback_config['phase1_subject_feedback']:
                feedback_text += "\n" + self._format_feedback(subject_answer, subject_correct)
            
            if self.feedback_config['phase1_teammate_feedback']:
                feedback_text += "\n" + self._format_feedback(teammate_answer, teammate_correct, source="Teammate's")
            
            self._log(feedback_text)

            # Record Data
            self._record_trial(
                phase=1, trial_num=i, q_data=q_data,
                subject_answer_p1=subject_answer,
                subject_correct_p1=subject_correct,
                teammate_answer_p1=teammate_answer,
                teammate_correct_p1=teammate_correct
            )
            
            time.sleep(0.2)  # Small pause

        # Calculate Phase 1 accuracies
        self.subject_accuracy_phase1 = subject_correct_count / self.n_trials_per_phase
        self.teammate_accuracy_phase1_observed = teammate_correct_count / self.n_trials_per_phase
        
        phase1_summary = "="*10 + " Phase 1 Complete " + "="*10
        phase1_summary += f"\nYour accuracy in Phase 1 (SAFN): {self.subject_accuracy_phase1:.2%}"
        phase1_summary += f"\nTeammate's observed accuracy in Phase 1 (TAFN_obs): {self.teammate_accuracy_phase1_observed:.2%}"
        phase1_summary += "\n" + "="*40 + "\n"
        
        self._log(phase1_summary)
        time.sleep(2)  # Pause before Phase 2
        
        return (feedback_text, message_history) if not self.is_human_player else ("", [])

    def run_phase2(self, final_feedback="", message_history=[]):
        """Runs Phase 2: Combined Answer (A-D) or Delegate ('T') Input."""
        setup_text = final_feedback + "\n" + "="*10 + " Starting Phase 2: Answer or Delegate " + "="*10 + "\n"
        setup_text += "For each question, enter your answer (A, B, C, D) or 'T' to let your teammate answer."
        
        self.current_phase = 2
        phase2_score = 0

        self._log(setup_text)
        feedback_text = ""
        
        for i, q_data in enumerate(self.phase2_questions):
            self.current_trial_in_phase = i
            formatted_question = self._present_question(q_data)

            options = q_data["options"]
            valid_inputs = list(options.keys()) + ['T']  # A, B, C, D, T
            prompt = "Your choice (A, B, C, D, or T=Teammate): "

            # Get input
            if self.is_human_player:
                print(formatted_question)
                while True:
                    try:
                        raw_input_str = input(prompt).strip().upper()
                        if raw_input_str in valid_inputs:
                            subject_choice = raw_input_str
                            break
                        else:
                            print(f"Invalid input. Please enter A, B, C, D, or T.")
                    except EOFError:
                        print("\nInput stream closed unexpectedly. Exiting trial.")
                        return False
            else:
                subject_choice, message_history = self._get_llm_answer(
                    valid_inputs, 
                    (setup_text if i==0 else "") + feedback_text + "\n" + formatted_question + "\n" + prompt + "\n", 
                    message_history
                )

            # Process choice
            if subject_choice == 'T':
                delegation_choice = "Teammate"
                feedback_text = "--> Delegating to teammate..."
                
                teammate_answer, teammate_correct = self._get_teammate_answer(q_data, phase=2)
                final_answer_label = teammate_answer
                final_correct = teammate_correct
                
                # Add teammate feedback if configured
                if self.feedback_config['phase2_teammate_feedback']:
                    feedback_text += "\n" + self._format_feedback(teammate_answer, teammate_correct, source="Teammate's")
            else:
                delegation_choice = "Self"
                subject_answer = subject_choice
                final_answer_label = subject_answer
                final_correct = (subject_answer == q_data["correct_answer"])
                
                feedback_text = "--> Your answer: " + subject_answer
                # Add subject feedback if configured
                if self.feedback_config['phase2_subject_feedback'] and subject_answer != 'T':
                    feedback_text = self._format_feedback(subject_answer, final_correct)

            if final_correct:
                phase2_score += 1

            # Record trial data
            self._record_trial(
                phase=2,
                trial_num=i,
                q_data=q_data,
                delegation_choice=delegation_choice,
                final_answer=final_answer_label,
                final_correct=final_correct
            )

            feedback_text += "\nChoice registered. Moving to the next question...\n"
            self._log(feedback_text)
            time.sleep(0.2)  # Small pause

        # Phase 2 Completion Summary
        phase2_summary = "="*10 + " Phase 2 Complete " + "="*10
        
        # Calculate delegation stats
        total_trials = len(self.phase2_questions)
        if total_trials > 0:
            teammate_delegations = sum(1 for trial in self.results if trial.get('delegation_choice') == "Teammate")
            delegation_percentage = (teammate_delegations / total_trials) * 100
            phase2_summary += f"\nDelegation to teammate occurred in {teammate_delegations}/{total_trials} trials ({delegation_percentage:.2f}%)."
        
        # Calculate final score
        self.phase2_score = phase2_score
        self.phase2_accuracy = phase2_score / self.n_trials_per_phase if self.n_trials_per_phase > 0 else 0
        phase2_summary += f"\nYour final score in Phase 2: {self.phase2_score}/{self.n_trials_per_phase} ({self.phase2_accuracy:.2%})"
        phase2_summary += "\n" + "="*40 + "\n"
        
        self._log(phase2_summary)
        return True

    def run_game(self):
        """Runs both phases of the game."""
        start_message = f"\nStarting Game for Subject: {self.subject_id}"
        start_message += f"\nParameters: N={self.n_trials_per_phase}, Target Teammate Accuracy={self.teammate_accuracy_target:.2%}"
        self._log(start_message)

        # Check if questions are loaded
        if not self.phase1_questions or not self.phase2_questions:
            self._log("ERROR: Cannot run game - questions not properly loaded or insufficient.")
            return None

        # Run Phase 1
        final_feedback, message_history = self.run_phase1()
        if final_feedback is False:  # Check if phase 1 was aborted
            self._log("Game aborted due to error in Phase 1.")
            return self.get_results()

        # Run Phase 2
        phase2_success = self.run_phase2(final_feedback, message_history)
        if not phase2_success:
            self._log("Game aborted due to error in Phase 2.")
            return self.get_results()

        self._log("--- Game Over ---")
        self._log_summary()
        self._save_results()
    
        return self.get_results()

    def get_results(self):
        """Returns the recorded trial data."""
        return copy.deepcopy(self.results)  # Return a copy to prevent accidental modification

    def _save_results(self):
        """Save detailed results to JSON file"""
        print(f"\nSaving detailed results to: {self.results_filename}")
        try:
            with open(self.results_filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print("Results saved successfully.")
            
            # Add a note to the log file
            with open(self.log_filename, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\nDetailed trial data saved to: {self.results_filename}\n")
                
        except Exception as e:
            error_msg = f"\nERROR saving results to file: {e}"
            print(error_msg)
            
            # Log the error
            with open(self.log_filename, 'a', encoding='utf-8') as log_f:
                log_f.write(error_msg + "\n")

    def _log_summary(self):
        """Generate and log a complete summary of the game results with all statistical analysis"""
        # Create a string to hold all the summary text
        summary = "\n" + "="*10 + " Results Summary & Analysis " + "="*10 + "\n"
        summary += f"Subject ID: {self.subject_id}\n"
        summary += f"Target Teammate Accuracy: {self.teammate_accuracy_target:.2%}\n"
        summary += f"Number of Trials per Phase (N): {self.n_trials_per_phase}\n"
        
        # Phase 1 accuracies
        if hasattr(self, 'subject_accuracy_phase1') and self.subject_accuracy_phase1 is not None:
            summary += f"Subject Phase 1 Accuracy (SAFN): {self.subject_accuracy_phase1:.2%}\n"
        else:
            summary += "Subject Phase 1 Accuracy (SAFN): Not Calculated (Phase 1 likely incomplete)\n"
            
        if hasattr(self, 'teammate_accuracy_phase1_observed') and self.teammate_accuracy_phase1_observed is not None:
            summary += f"Observed Teammate Phase 1 Accuracy (TAFN_obs): {self.teammate_accuracy_phase1_observed:.2%}\n"
        else:
            summary += "Observed Teammate Phase 1 Accuracy (TAFN_obs): Not Calculated (Phase 1 likely incomplete)\n"
            
        if hasattr(self, 'phase2_accuracy') and self.phase2_accuracy is not None:
            summary += f"Phase 2 Accuracy: {self.phase2_accuracy:.2%}\n"
        else:
            summary += "Phase 2 Accuracy: Not Calculated (Phase 2 likely incomplete)\n"
        
        # Calculate Phase 2 delegations and per-strategy accuracy
        phase2_results = [r for r in self.results if r['phase'] == 2]
        total_phase2 = len(phase2_results)
        
        if total_phase2 > 0:
            # Delegation statistics
            team_delegations = sum(1 for r in phase2_results if r['delegation_choice'] == 'Teammate')
            self_answers = sum(1 for r in phase2_results if r['delegation_choice'] == 'Self')
            
            delegation_pct = (team_delegations / total_phase2) * 100
            summary += f"\nDelegation to teammate occurred in {team_delegations}/{total_phase2} trials ({delegation_pct:.2f}%)\n"
            
            # Calculate accuracy when delegating vs self-answering
            self_correct = sum(1 for r in phase2_results if r['delegation_choice'] == 'Self' and r['final_correct'])
            team_correct = sum(1 for r in phase2_results if r['delegation_choice'] == 'Teammate' and r['final_correct'])
            
            if self_answers > 0:
                self_accuracy = self_correct / self_answers
                summary += f"Self-answer accuracy in Phase 2: {self_correct}/{self_answers} ({self_accuracy:.2%})\n"
            
            if team_delegations > 0:
                team_accuracy = team_correct / team_delegations
                summary += f"Delegated answer accuracy in Phase 2: {team_correct}/{team_delegations} ({team_accuracy:.2%})\n"
        
        # Statistical tests
        safn = getattr(self, 'subject_accuracy_phase1', None)
        tafn_obs = getattr(self, 'teammate_accuracy_phase1_observed', None)
        phase2_acc = getattr(self, 'phase2_accuracy', None)
        phase2_successes = getattr(self, 'phase2_score', None)
        n_phase2 = getattr(self, 'n_trials_per_phase', 0)
        
        if all(v is not None for v in [safn, tafn_obs, phase2_acc, phase2_successes]) and n_phase2 > 0:
            summary += f"\n--- Statistical Analysis (Phase 2 Performance) ---\n"
            summary += f"Observed: {phase2_successes} successes in {n_phase2} trials (Accuracy: {phase2_acc:.2%})\n"

            # Compare Phase 1 vs Phase 2 self-accuracy
            phase1_correct = sum(1 for r in self.results if r['phase'] == 1 and r['subject_correct_p1'])
            phase1_total = sum(1 for r in self.results if r['phase'] == 1)
            phase1_accuracy = phase1_correct / phase1_total if phase1_total > 0 else 0

            # We already calculated these values earlier in the function
            phase2_self_correct = sum(1 for r in phase2_results if r['delegation_choice'] == 'Self' and r['final_correct'])
            phase2_self_total = sum(1 for r in phase2_results if r['delegation_choice'] == 'Self')
            phase2_self_accuracy = phase2_self_correct / phase2_self_total if phase2_self_total > 0 else 0

            summary += f"\n--- Self-accuracy Comparison (Phase 1 vs Phase 2) ---\n"
            summary += f"Phase 1 self-accuracy: {phase1_correct}/{phase1_total} ({phase1_accuracy:.2%})\n"
            summary += f"Phase 2 self-accuracy: {phase2_self_correct}/{phase2_self_total} ({phase2_self_accuracy:.2%})\n"

            # Perform binomial test to compare accuracies between phases
            if phase1_total > 0 and phase2_self_total > 0:
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
    
def main():
    """Main function to run the psychological experiment game"""
    # Game Configuration
    NUM_TRIALS_PER_PHASE = 100
    TEAMMATE_ACCURACY_TARGET = 0.8
    IS_HUMAN = False
    DATASET_NAME = "GPQA"  # "TruthfulQA" or "GPQA"
    
    # Feedback configuration
    feedback_config = {
        "phase1_subject_feedback": False,     # Show subject's answer feedback in phase 1
        "phase1_teammate_feedback": True,    # Show teammate's answer feedback in phase 1
        "phase2_subject_feedback": False,     # Show subject's answer feedback in phase 2
        "phase2_teammate_feedback": False,    # Show teammate's answer feedback in phase 2
        "show_answer_with_correctness": False,    
    }
    
    playerstr = "LLM" if not IS_HUMAN else "Human"
    SUBJECT_ID = f"{playerstr}Test_{DATASET_NAME}_{TEAMMATE_ACCURACY_TARGET}_{int(time.time())}"

    TOTAL_QUESTIONS_NEEDED = NUM_TRIALS_PER_PHASE * 2

    # Load and Format Questions
    print("-" * 50)
    if DATASET_NAME == "GPQA":
        formatted_questions = load_and_format_gpqa(num_questions_needed=TOTAL_QUESTIONS_NEEDED, hf_token=hf_token)
    else:
        formatted_questions = load_and_format_truthfulqa(num_questions_needed=TOTAL_QUESTIONS_NEEDED)
    print("-" * 50)

    # Create and Run Game Instance
    if formatted_questions and len(formatted_questions) >= TOTAL_QUESTIONS_NEEDED:
        try:
            game = PsychGame(
                subject_id=SUBJECT_ID,
                questions=formatted_questions,
                n_trials_per_phase=NUM_TRIALS_PER_PHASE,
                teammate_accuracy=TEAMMATE_ACCURACY_TARGET,
                feedback_config=feedback_config  # Pass the feedback configuration
            )

            # Set player type
            game.is_human_player = IS_HUMAN
            print(f"Player type set to: {'Human' if IS_HUMAN else 'LLM'}")

            # Run the game
            all_results = game.run_game()

        except ValueError as e:
            print(f"\nError during game initialization or execution: {e}")
            all_results = None

    else:
        print("\nFATAL: Game setup failed - Could not load or format sufficient questions.")
        print("Please check dataset availability/connection, required number of questions vs N, and dataset split.")

    print("\nScript finished.")

# Execute main function when script is run directly
if __name__ == "__main__":
    main()