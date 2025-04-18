# -*- coding: utf-8 -*-
"""
Complete script for a psychological game experiment testing
self/other modeling using trivia questions.

Features:
- Loads questions from Hugging Face's TruthfulQA dataset.
- Ensures unique questions across the experiment.
- Two phases: Modeling (Phase 1) and Decision (Phase 2).
- Phase 1: Subject answers questions, gets feedback on own and teammate's performance.
- Phase 2: Subject chooses to answer (A-D) or delegate ('T') for each question.
- Simulates a teammate with configurable accuracy.
- Records detailed trial data.
- Calculates Phase 1 (SAFN, TAFN_obs) and Phase 2 accuracies.
- Performs binomial test comparing Phase 2 accuracy to max(SAFN, TAFN_obs).
- Includes placeholders for integration with Large Language Models (LLMs).
- Saves results to a JSON file.
"""

import anthropic
import random
import time
import copy
import json
import sys # To ensure UTF-8 encoding for printing if needed
from datasets import load_dataset
import scipy.stats
import os
from dotenv import load_dotenv
load_dotenv()
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
# --- Configuration ---
# Configure stdout for UTF-8 if necessary, especially on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
        print("Note: Reconfigured stdout/stderr to UTF-8.")
    except Exception as e:
        print(f"Warning: Could not reconfigure stdout/stderr to UTF-8: {e}")


# --- Data Loading and Formatting ---
def load_and_format_gpqa(num_questions_needed, hf_token=hf_token, split="train", filter_id="recgCB0HSVt2IslDN"):
    """
    Loads the GPQA dataset (Idavidrein/gpqa) and formats questions
    into the A-D multiple-choice format required by PsychGame.

    Args:
        num_questions_needed (int): The total number of unique questions required.
        hf_token (str, optional): Hugging Face API token for private/gated datasets. Defaults to None.
        split (str): The dataset split to use (likely "train"). Defaults to "train".
        filter_id (str, optional): The 'Record ID' to filter out. Defaults to "recgCB0HSVt2IslDN".

    Returns:
        list: A list of question dictionaries formatted for PsychGame, or None if loading fails.
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
    random.shuffle(dataset_indices) # Shuffle to get a random sample if needed

    print(f"Formatting {num_questions_needed} questions from GPQA...")
    skipped_filter = 0
    skipped_missing_data = 0

    for idx in dataset_indices:
        if len(formatted_questions) >= num_questions_needed:
            break

        item = dataset[idx]

        # Check if all required fields exist and are not None/empty
        if not all(item.get(field) for field in required_fields):
            skipped_missing_data +=1
            # print(f"Skipping GPQA index {idx}: Missing one or more required fields.")
            continue

        record_id = item['Record ID']

        # Apply filtering
        if record_id == filter_id:
            skipped_filter += 1
            continue

        # Check if ID already added (shouldn't happen with unique Record IDs, but safeguard)
        if record_id in question_ids_added:
            continue

        # Gather options
        correct_answer_text = item['Correct Answer'].strip()
        incorrect_answers_text = [
            item['Incorrect Answer 1'].strip(),
            item['Incorrect Answer 2'].strip(),
            item['Incorrect Answer 3'].strip()
        ]
        if len(correct_answer_text) == 0 or len(incorrect_answers_text[0]) == 0 or len(incorrect_answers_text[1]) == 0 or len(incorrect_answers_text[2]) == 0: continue

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
            # Compare texts to find the label corresponding to the correct answer
            if option_text == correct_answer_text:
                correct_label = label

        # This check ensures the correct answer was found among the options
        if correct_label is None:
             print(f"Error processing GPQA Record ID {record_id}: Correct answer text not found in options list after shuffling. Skipping.")
             continue

        # Create the formatted dictionary using a unique ID derived from Record ID
        formatted_q = {
            "id": f"gpqa_{split}_{record_id}", # Unique ID
            "question": item['Question'],
            "options": options_dict,
            "correct_answer": correct_label # The letter 'A', 'B', 'C', or 'D'
        }
        formatted_questions.append(formatted_q)
        question_ids_added.add(record_id) # Add original Record ID to prevent duplicates
    return formatted_questions

def load_and_format_truthfulqa(num_questions_needed, split="validation"):
    """
    Loads the TruthfulQA dataset (generation config) and formats questions
    into the A-D multiple-choice format required by PsychGame.

    Args:
        num_questions_needed (int): The total number of unique questions required.
        split (str): The dataset split to use (e.g., "validation").

    Returns:
        list: A list of question dictionaries formatted for PsychGame, or None if loading fails.
    """
    print(f"Attempting to load TruthfulQA ({split} split)...")
    try:
        dataset = load_dataset("truthful_qa", "generation", split=split, trust_remote_code=True) # Added trust_remote_code=True potentially needed by HF datasets
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading TruthfulQA dataset: {e}")
        print("Please ensure you have the 'datasets' library installed (`pip install datasets`)")
        print("and an internet connection. You might also need `trust_remote_code=True`.")
        return None

    formatted_questions = []
    attempts = 0
    max_attempts = len(dataset) * 2 # Safety break

    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)

    question_ids_added = set() # Keep track of IDs to ensure uniqueness

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
        if len(best_answer.strip()) == 0: continue
        incorrect_answers = item.get('incorrect_answers')

        # Basic validation of required fields
        if not all([question_text, best_answer, incorrect_answers]):
            # print(f"Skipping item index {idx}: Missing required fields.")
            continue

        # Need at least 3 incorrect answers to form 4 options
        if not isinstance(incorrect_answers, list) or len(incorrect_answers) < 3:
            # print(f"Skipping question index {idx} due to insufficient incorrect answers: {question_text[:50]}...")
            continue

        # Ensure best_answer is not accidentally in the chosen incorrect list
        possible_incorrect = [ans for ans in incorrect_answers if ans != best_answer and len(ans.strip()) > 0]
        if len(possible_incorrect) < 3:
            # print(f"Skipping question index {idx} due to overlap or insufficient unique incorrect answers: {question_text[:50]}...")
            continue

        # Select 3 distinct incorrect answers
        try:
            chosen_incorrect = random.sample(possible_incorrect, 3)
        except ValueError:
             # print(f"Skipping question index {idx}: Not enough unique incorrect answers to sample from.")
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
            "correct_answer": correct_label # The letter 'A', 'B', 'C', or 'D'
        }
        formatted_questions.append(formatted_q)
        question_ids_added.add(potential_id) # Add ID after successful formatting

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
                 config=None):
        """
        Initializes the game instance.

        Args:
            subject_id (str): Identifier for the current subject/session.
            questions (list): A list of formatted question dictionaries.
            n_trials_per_phase (int): Number of trials (N) in each phase.
            teammate_accuracy (float): The teammate's target accuracy (probability, 0.0 to 1.0).
            config (dict, optional): Configuration options for game behavior.
        """
        if not questions:
             raise ValueError("No questions provided to the game.")
        if not (0.0 <= teammate_accuracy <= 1.0):
            raise ValueError("Teammate accuracy must be between 0.0 and 1.0")
        if not isinstance(n_trials_per_phase, int) or n_trials_per_phase <= 0:
             raise ValueError("Number of trials per phase must be a positive integer.")

        # Default configuration
        self.config = {
            'show_subject_feedback_p1': True,   # Show feedback to subject about their answers in Phase 1
            'show_teammate_feedback_p1': True,  # Show feedback about teammate's answers in Phase 1
            'show_subject_feedback_p2': True,   # Show feedback to subject about their answers in Phase 2
            'show_teammate_feedback_p2': False, # Show feedback about teammate's answers in Phase 2
        }
        
        # Update configuration with provided values
        if config:
            self.config.update(config)

        self.subject_id = subject_id
        self.n_trials_per_phase = n_trials_per_phase
        self.teammate_accuracy_target = teammate_accuracy
        
        # Prepare teammate's answers in advance to ensure exact accuracy percentage
        self.teammate_correct_answers = self._prepare_teammate_answers(n_trials_per_phase * 2)

        total_questions_needed = n_trials_per_phase * 2

        # Ensure sufficient questions
        if len(questions) < total_questions_needed:
            raise ValueError(f"Not enough questions provided ({len(questions)}) for the required {total_questions_needed}.")

        # Select exactly N*2 questions
        self.game_questions = questions[:total_questions_needed]
        
        # Split into phases
        self.phase1_questions = self.game_questions[:n_trials_per_phase]
        self.phase2_questions = self.game_questions[n_trials_per_phase:]

        # Initialize state and results storage
        self.results = []
        self.current_phase = 0
        self.current_trial_in_phase = 0
        self.subject_accuracy_phase1 = None
        self.teammate_accuracy_phase1_observed = None
        self.phase2_score = None
        self.phase2_accuracy = None
        self.is_human_player = True  # Default to human input
        
        # Track teammate's answer index to use pre-computed answers in sequence
        self.teammate_answer_index = 0

    def _prepare_teammate_answers(self, total_questions):
        """
        Pre-computes whether the teammate will answer correctly for each question.
        This ensures the actual accuracy matches the target accuracy exactly.
        
        Args:
            total_questions (int): The total number of questions in the game.
            
        Returns:
            list: A list of booleans indicating whether the teammate will answer correctly.
        """
        # Calculate exact number of correct answers needed
        # Create separate arrays for phase 1 and phase 2
        half_count = total_questions // 2
        
        # For phase 1, exactly match the target accuracy
        phase1_correct = round(half_count * self.teammate_accuracy_target)
        phase1_answers = [True] * phase1_correct + [False] * (half_count - phase1_correct)
        random.shuffle(phase1_answers)
        
        # For phase 2, exactly match the target accuracy
        phase2_correct = round(half_count * self.teammate_accuracy_target)
        phase2_answers = [True] * phase2_correct + [False] * (half_count - phase2_correct)
        random.shuffle(phase2_answers)
        
        # Combine the answers from both phases
        answers = phase1_answers + phase2_answers
        return answers

    def _get_llm_answer(self, options, q_text, message_history):
        """Gets an answer from the LLM."""
        user_msg = {"role": "user", "content": q_text}
        message_history.append(user_msg)
        resp = ""
        system_msg = "Output only the letter of your choice: " + ", ".join(options[:-1]) + f", or {options[-1]}.\n"

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
                print(f"Error: {e}")
                
        asst_msg = {"role": "assistant", "content": resp}
        message_history.append(asst_msg)

        if resp not in options: 
            print(f"Failed to get valid response for text: {q_text}; response: {resp}")

        return resp, message_history
        
    def _get_subject_answer(self, question_data):
        """Gets the subject's multiple-choice answer (A-D)."""
        options = question_data["options"]
        prompt = "Your answer (A, B, C, D): "

        valid_choice = False
        while not valid_choice:
            try:
                answer = input(prompt).strip().upper()
                if answer in options:
                    return answer
                else:
                    print("Invalid input. Please enter A, B, C, or D.")
            except Exception as e:
                print(f"\nError getting input: {str(e)}")
                print("Selecting a random answer and continuing...")
                return random.choice(list(options.keys()))

    def _simulate_teammate_answer(self, question_data):
        """
        Simulates the teammate's answer based on pre-computed correctness.
        
        Returns:
            tuple: (teammate_answer_str (A-D), is_correct (bool))
        """
        correct_answer = question_data["correct_answer"]
        options = question_data["options"]
        possible_answers = list(options.keys())
        
        # Use pre-computed correctness
        is_correct = self.teammate_correct_answers[self.teammate_answer_index]
        self.teammate_answer_index += 1

        if is_correct:
            return correct_answer, True
        else:
            # Choose a random incorrect answer
            incorrect_options = [opt for opt in possible_answers if opt != correct_answer]
            if not incorrect_options:
                return correct_answer, True  # Can't be incorrect (edge case)
            return random.choice(incorrect_options), False

    def _present_question(self, question_data):
        """Formats a question for display."""
        formatted_question = "-" * 30 + "\n"
        formatted_question += f"Question {self.current_trial_in_phase + 1}:\n"
        formatted_question += question_data["question"] + "\n"
        formatted_question += "-" * 10 + "\n"
        for key, value in question_data["options"].items():
            formatted_question += f"  {key}: {value}\n"
        formatted_question += "-" * 30
        return formatted_question

    def _record_trial(self, phase, trial_num, q_data, **kwargs):
        """Records the data for a single trial."""
        # Base data structure with default None values
        trial_data = {
            "subject_id": self.subject_id,
            "phase": phase,
            "trial_in_phase": trial_num,
            "teammate_accuracy_target": self.teammate_accuracy_target,
            "question_id": q_data.get("id", f"unknown_q_{phase}_{trial_num}"),
            "question_text": q_data["question"],
            "options": copy.deepcopy(q_data["options"]),
            "correct_answer_label": q_data["correct_answer"],
            "correct_answer_text": q_data["options"].get(q_data["correct_answer"], "ERROR"),
            "timestamp": time.time(),
            # Default values for all trial fields
            'subject_answer_p1': None, 'subject_correct_p1': None,
            'teammate_answer_p1': None, 'teammate_correct_p1': None,
            'delegation_choice': None, 'chosen_source': None,
            'final_answer': None, 'final_answer_text': None, 'final_correct': None
        }
        
        # Update with actual data
        trial_data.update(kwargs)

        # Derive answer text if needed
        if trial_data['final_answer'] and trial_data['final_answer'] in trial_data['options']:
            trial_data['final_answer_text'] = trial_data['options'][trial_data['final_answer']]

        self.results.append(trial_data)

    def run_phase1(self):
        """Runs Phase 1: Modeling."""
        setup_text = initial_setup_explanation
        setup_text += "\n\n" + "="*10 + " Starting Phase 1: Answer the Questions " + "="*10
        print(setup_text)
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
                    return ("", [])  # Return empty tuple instead of False
            else:
                subject_answer, message_history = self._get_llm_answer(
                    list(q_data["options"].keys()), 
                    (setup_text if i==0 else "") + feedback_text + "\n" + formatted_question + "\n", 
                    message_history
                )

            subject_correct = (subject_answer == q_data["correct_answer"])
            if subject_correct:
                subject_correct_count += 1

            # Simulate Teammate Answer
            teammate_answer, teammate_correct = self._simulate_teammate_answer(q_data)
            if teammate_correct:
                teammate_correct_count += 1

            # Build feedback text based on configuration
            feedback_text = "\n"
            if self.config['show_subject_feedback_p1']:
                feedback_text += f"Your answer: {subject_answer} ({'Correct' if subject_correct else 'Incorrect'})\n"
            
            if self.config['show_teammate_feedback_p1']:
                feedback_text += f"Teammate's answer: {teammate_answer} ({'Correct' if teammate_correct else 'Incorrect'})\n"
                
                
            print(feedback_text)

            # Record data
            self._record_trial(
                phase=1, trial_num=i, q_data=q_data,
                subject_answer_p1=subject_answer,
                subject_correct_p1=subject_correct,
                teammate_answer_p1=teammate_answer,
                teammate_correct_p1=teammate_correct
            )
            time.sleep(0.2)

        # Calculate Phase 1 accuracies
        self.subject_accuracy_phase1 = subject_correct_count / self.n_trials_per_phase
        self.teammate_accuracy_phase1_observed = teammate_correct_count / self.n_trials_per_phase
        
        print("="*10 + " Phase 1 Complete " + "="*10)
        print(f"Your accuracy in Phase 1 (SAFN): {self.subject_accuracy_phase1:.2%}")
        print(f"Teammate's observed accuracy in Phase 1 (TAFN_obs): {self.teammate_accuracy_phase1_observed:.2%}")
        print("="*40 + "\n")
        time.sleep(2)
        
        return (feedback_text, message_history) if not self.is_human_player else ("", [])

    def run_phase2(self, final_feedback="", message_history=[]):
        """Runs Phase 2: Combined Answer (A-D) or Delegate ('T') Input."""
        setup_text = final_feedback + "\n" + "="*10 + " Starting Phase 2: Answer or Delegate " + "="*10 + "\n"
        setup_text += "For each question, enter your answer (A, B, C, D) or 'T' to let your teammate answer."
        self.current_phase = 2
        phase2_score = 0

        print(setup_text)
        feedback_text = ""
        
        for i, q_data in enumerate(self.phase2_questions):
            self.current_trial_in_phase = i
            formatted_question = self._present_question(q_data)

            options = q_data["options"]
            valid_inputs = list(options.keys()) + ['T']
            prompt = "Your choice (A, B, C, D, or T=Teammate): "

            # Get subject choice
            if self.is_human_player:
                print(formatted_question)
                valid_choice = False
                while not valid_choice:
                    try:
                        choice = input(prompt).strip().upper()
                        if choice in valid_inputs:
                            subject_choice = choice
                            valid_choice = True
                        else:
                            print(f"Invalid input. Please enter A, B, C, D, or T.")
                    except Exception as e:
                        print(f"\nError getting input: {str(e)}")
                        print("Defaulting to self-answer...")
                        subject_choice = list(options.keys())[0]
                        valid_choice = True
            else:
                subject_choice, message_history = self._get_llm_answer(
                    valid_inputs, 
                    (setup_text if i==0 else "") + feedback_text + "\n" + formatted_question + "\n" + prompt + "\n", 
                    message_history
                )

            # Process subject's choice
            feedback_text = ""
            if subject_choice == 'T':
                delegation_choice = "Teammate"
                feedback_text = "--> Delegating to teammate...\n"
                
                teammate_answer, teammate_correct = self._simulate_teammate_answer(q_data)
                final_answer = teammate_answer
                final_correct = teammate_correct
                
                if self.config['show_teammate_feedback_p2']:
                    feedback_text += f"Teammate's answer: {teammate_answer}\n"
            else:
                delegation_choice = "Self"
                final_answer = subject_choice
                final_correct = (final_answer == q_data["correct_answer"])
                
                if self.config['show_subject_feedback_p2']:
                    feedback_text += f"--> Your answer: {subject_choice}\n"


            if final_correct:
                phase2_score += 1

            # Record data
            self._record_trial(
                phase=2,
                trial_num=i,
                q_data=q_data,
                delegation_choice=delegation_choice,
                final_answer=final_answer,
                final_correct=final_correct
            )

            feedback_text += "Choice registered. Moving to the next question...\n"
            print(feedback_text)
            time.sleep(0.2)

        # Phase 2 completion
        print("="*10 + " Phase 2 Complete " + "="*10)
        
        # Calculate delegation statistics
        total_trials = len(self.phase2_questions)
        if total_trials > 0:
            teammate_delegations = sum(1 for trial in self.results 
                                       if trial.get('delegation_choice') == "Teammate")
            delegation_percentage = (teammate_delegations / total_trials) * 100
            print(f"Delegation to teammate: {teammate_delegations}/{total_trials} ({delegation_percentage:.2f}%).")
        
        self.phase2_score = phase2_score
        self.phase2_accuracy = phase2_score / self.n_trials_per_phase if self.n_trials_per_phase > 0 else 0
        print(f"Final score in Phase 2: {self.phase2_score}/{self.n_trials_per_phase} ({self.phase2_accuracy:.2%})")
        print("="*40 + "\n")
        return True

    def run_game(self):
        """Runs both phases of the game."""
        print(f"\nStarting Game for Subject: {self.subject_id}")
        print(f"Parameters: N={self.n_trials_per_phase}, Target Teammate Accuracy={self.teammate_accuracy_target:.2%}")

        # Check questions
        if not self.phase1_questions or not self.phase2_questions:
            print("ERROR: Questions not properly loaded or insufficient.")
            return None

        # Run Phase 1
        final_feedback, message_history = self.run_phase1()

        # Run Phase 2
        phase2_success = self.run_phase2(final_feedback, message_history)
        if not phase2_success:
            print("Game aborted due to error in Phase 2.")
            return self.get_results()

        print("--- Game Over ---")
        return self.get_results()

    def get_results(self):
        """Returns the recorded trial data."""
        return copy.deepcopy(self.results)

    def set_player_type(self, is_human=True):
        """Set whether the player is human or LLM."""
        self.is_human_player = is_human
        print(f"Player type set to: {'Human' if is_human else 'LLM'}")


# --- Main Execution Block ---

def run_statistical_analysis(game, all_results):
    """Perform statistical analysis on game results."""
    print("\n" + "="*10 + " Results Summary & Analysis " + "="*10)
    print(f"Subject ID: {game.subject_id}")
    print(f"Target Teammate Accuracy: {game.teammate_accuracy_target:.2%}")
    print(f"Number of Trials per Phase (N): {game.n_trials_per_phase}")

    # Get calculated accuracies
    safn = game.subject_accuracy_phase1
    tafn_obs = game.teammate_accuracy_phase1_observed
    phase2_acc = game.phase2_accuracy
    phase2_successes = game.phase2_score
    n_phase2 = game.n_trials_per_phase

    print(f"Subject Phase 1 Accuracy (SAFN): {safn:.2%}")
    print(f"Observed Teammate Phase 1 Accuracy (TAFN_obs): {tafn_obs:.2%}")
    print(f"Phase 2 Accuracy: {phase2_acc:.2%}")

    # Statistical tests
    print(f"\n--- Statistical Analysis (Phase 2 Performance) ---")
    print(f"Observed: {phase2_successes} successes in {n_phase2} trials (Accuracy: {phase2_acc:.2%})")

    # Define baseline strategies
    baselines = {
        "Max(SAFN, TAFN_obs)": max(safn, tafn_obs),
        "Always Self": safn,
        "Always Teammate": tafn_obs,
        "Random Choice": 0.5 * safn + 0.5 * tafn_obs,
    }

    # Display baseline expectations
    print("\nBaseline Strategy Expected Accuracies:")
    for name, prob in baselines.items():
        # Clamp probabilities to prevent floating point issues
        clamped_prob = max(0.0, min(1.0, prob))
        baselines[name] = clamped_prob
        print(f"- {name}: {clamped_prob:.2%}")

    # Perform binomial tests
    print("\nComparing Observed Phase 2 Accuracy vs. Baselines (Two-Sided Tests):")

    for name, baseline_prob in baselines.items():
        print(f"\n  Comparison vs. '{name}' (Expected Acc: {baseline_prob:.2%}):")
        
        # Handle edge cases p=0 or p=1
        if baseline_prob in (0.0, 1.0):
            if (baseline_prob == 1.0 and phase2_successes == n_phase2) or \
               (baseline_prob == 0.0 and phase2_successes == 0):
                print(f"    Result: Observed score perfectly matches the {baseline_prob:.0%} baseline")
            else:
                direction = "less" if baseline_prob == 1.0 else "greater"
                print(f"    Result: Observed score ({phase2_acc:.2%}) is {direction} than the {baseline_prob:.0%} baseline")
                try:
                    test = scipy.stats.binomtest(
                        k=phase2_successes, 
                        n=n_phase2, 
                        p=baseline_prob, 
                        alternative='two-sided'
                    )
                    print(f"    Test p-value: {test.pvalue:.4f} (Observed is significantly {direction.upper()})")
                except ValueError as e:
                    print(f"    Could not run test: {e}")
            continue
            
        # Regular binomial test for p between 0 and 1
        try:
            test = scipy.stats.binomtest(
                k=phase2_successes,
                n=n_phase2,
                p=baseline_prob,
                alternative='two-sided'
            )
            p_value = test.pvalue
            print(f"    Test (Observed != Baseline?): p-value = {p_value:.4f}")
            
            # Interpret results
            if p_value < 0.05:
                if phase2_acc > baseline_prob:
                    print("      Interpretation: Observed accuracy is significantly GREATER than baseline (p < 0.05)")
                else:
                    print("      Interpretation: Observed accuracy is significantly LESS than baseline (p < 0.05)")
            else:
                print("      Interpretation: Observed accuracy is NOT significantly different from baseline (p >= 0.05)")
                
        except ValueError as e:
            print(f"    Error during test: {e}")

def save_results(subject_id, all_results, game=None):
    """Save game results to a JSON file."""
    timestamp = int(time.time())
    results_filename = f"./game_logs/{subject_id}_results_{timestamp}.json"
    print(f"\nSaving detailed results to: {results_filename}")
    
    # Add summary stats if game object is provided
    if game:
        summary_stats = {
            "subject_id": game.subject_id,
            "timestamp": timestamp,
            "target_teammate_accuracy": game.teammate_accuracy_target,
            "observed_teammate_accuracy": game.teammate_accuracy_phase1_observed,
            "subject_accuracy_phase1": game.subject_accuracy_phase1,
            "phase2_accuracy": game.phase2_accuracy,
            "n_trials_per_phase": game.n_trials_per_phase,
            "delegation_rate": sum(1 for trial in game.results if trial.get('delegation_choice') == "Teammate") / game.n_trials_per_phase
        }
        
        # Combine summary with detailed results
        output_data = {
            "summary": summary_stats,
            "trial_data": all_results
        }
    else:
        output_data = all_results
    
    try:
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print("Results saved successfully.")
    except Exception as e:
        print(f"\nERROR saving results to file: {e}")

def main(num_trials=100, teammate_accuracy=0.2, is_human=False, dataset_name="GPQA", config=None, subject_id=None):
    """
    Main function to set up and run the game.
    
    Args:
        num_trials (int): Number of trials per phase
        teammate_accuracy (float): Target accuracy for teammate (0.0-1.0)
        is_human (bool): Whether the subject is human (True) or LLM (False)
        dataset_name (str): Dataset to use ("GPQA" or "TruthfulQA")
        config (dict): Configuration for feedback visibility
        subject_id (str): Custom subject ID (generated if None)
    """
    # Generate subject ID if not provided
    player_type = "Human" if is_human else "LLM"
    if subject_id is None:
        subject_id = f"{player_type}Test_{dataset_name}_{teammate_accuracy}_{int(time.time())}"
    
    # Default configuration
    default_config = {
        'show_subject_feedback_p1': True,
        'show_teammate_feedback_p1': True,
        'show_subject_feedback_p2': False,
        'show_teammate_feedback_p2': False,
    }
    
    # Override with any provided config
    if config:
        default_config.update(config)
        
    # Load questions
    total_questions_needed = num_trials * 2
    print("-" * 50)
    if dataset_name == "GPQA":
        formatted_questions = load_and_format_gpqa(
            num_questions_needed=total_questions_needed, 
            hf_token=hf_token
        )
    else:
        formatted_questions = load_and_format_truthfulqa(
            num_questions_needed=total_questions_needed
        )
    print("-" * 50)
    
    # Create and run game if we have enough questions
    if not formatted_questions or len(formatted_questions) < total_questions_needed:
        print("\nFATAL: Could not load or format sufficient questions.")
        print("Please check dataset availability and connection.")
        return
        
    try:
        # Initialize game
        game = PsychGame(
            subject_id=subject_id,
            questions=formatted_questions,
            n_trials_per_phase=num_trials,
            teammate_accuracy=teammate_accuracy,
            config=default_config
        )
        
        # Set player type and run game
        game.set_player_type(is_human=is_human)
        all_results = game.run_game()
        
        # Process results
        if all_results:
            run_statistical_analysis(game, all_results)
            save_results(subject_id, all_results, game)
        else:
            print("\nGame did not complete successfully.")
            
    except Exception as e:
        print(f"\nError during game: {e}")
    
    print("\nScript finished.")

if __name__ == "__main__":
    # Import collections for the run - adding it here to prevent error in case used internally
    import collections
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run the delegation game experiment')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials per phase')
    parser.add_argument('--teammate_accuracy', type=float, default=0.2, help='Target teammate accuracy (0.0-1.0)')
    parser.add_argument('--is_human', type=bool, default=False, help='Is the subject a human (True) or LLM (False)')
    parser.add_argument('--dataset_name', choices=['GPQA', 'TruthfulQA'], default='GPQA', help='Dataset to use')
    
    # Feedback configuration options
    parser.add_argument('--subject-feedback-p1', dest='show_subject_feedback_p1', action='store_true', help='Show feedback on subject answers in Phase 1')
    parser.add_argument('--no-subject-feedback-p1', dest='show_subject_feedback_p1', action='store_false', help='Hide feedback on subject answers in Phase 1')
    parser.add_argument('--teammate-feedback-p1', dest='show_teammate_feedback_p1', action='store_true', help='Show feedback on teammate answers in Phase 1')
    parser.add_argument('--no-teammate-feedback-p1', dest='show_teammate_feedback_p1', action='store_false', help='Hide feedback on teammate answers in Phase 1')
    parser.add_argument('--subject-feedback-p2', dest='show_subject_feedback_p2', action='store_true', help='Show feedback on subject answers in Phase 2')
    parser.add_argument('--no-subject-feedback-p2', dest='show_subject_feedback_p2', action='store_false', help='Hide feedback on subject answers in Phase 2')
    parser.add_argument('--teammate-feedback-p2', dest='show_teammate_feedback_p2', action='store_true', help='Show feedback on teammate answers in Phase 2')
    parser.add_argument('--no-teammate-feedback-p2', dest='show_teammate_feedback_p2', action='store_false', help='Hide feedback on teammate answers in Phase 2')
    
    # Set defaults for the boolean flags
    parser.set_defaults(
        show_subject_feedback_p1=True,
        show_teammate_feedback_p1=True,
        show_subject_feedback_p2=False,
        show_teammate_feedback_p2=False,
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create config from parsed arguments
    config = {
        'show_subject_feedback_p1': args.show_subject_feedback_p1,
        'show_teammate_feedback_p1': args.show_teammate_feedback_p1,
        'show_subject_feedback_p2': args.show_subject_feedback_p2,
        'show_teammate_feedback_p2': args.show_teammate_feedback_p2,
    }
    
    # Override default parameters
    main_args = {
        'num_trials': args.num_trials,
        'teammate_accuracy': args.teammate_accuracy,
        'is_human': args.is_human,
        'dataset_name': args.dataset_name,
        'config': config
    }
    
    main(**main_args)