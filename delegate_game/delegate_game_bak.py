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
    def __init__(self, subject_id, questions, n_trials_per_phase, teammate_accuracy):
        """
        Initializes the game instance.

        Args:
            subject_id (str): Identifier for the current subject/session.
            questions (list): A list of formatted question dictionaries.
            n_trials_per_phase (int): Number of trials (N) in each phase.
            teammate_accuracy (float): The teammate's target accuracy (probability, 0.0 to 1.0).
        """
        if not questions:
             raise ValueError("No questions provided to the game.")
        if not (0.0 <= teammate_accuracy <= 1.0):
            raise ValueError("Teammate accuracy must be between 0.0 and 1.0")
        if not isinstance(n_trials_per_phase, int) or n_trials_per_phase <= 0:
             raise ValueError("Number of trials per phase must be a positive integer.")

        self.subject_id = subject_id
        self.n_trials_per_phase = n_trials_per_phase
        self.teammate_accuracy_target = teammate_accuracy

        total_questions_needed = n_trials_per_phase * 2

        # --- Uniqueness and Quantity Checks ---
        if len(questions) < total_questions_needed:
            raise ValueError(f"Not enough questions provided ({len(questions)}) for the required {total_questions_needed}. Check loading source or N.")

        unique_q_ids_provided = {q['id'] for q in questions}
        if len(unique_q_ids_provided) < total_questions_needed:
             # This check might be redundant if the input `questions` list is already pre-filtered for uniqueness
             print(f"Warning: Input question list has only {len(unique_q_ids_provided)} unique IDs, but {total_questions_needed} are required based on N={n_trials_per_phase}.")
             # We proceed but might reuse questions if the list itself had duplicates initially.
             # Let's select the first N*2 available, assuming the loading function did its best to provide unique ones.

        # Select exactly N*2 questions. Assume `questions` list has unique items if load func worked.
        self.game_questions = questions[:total_questions_needed]

        # Final check on the selected set (important safeguard)
        selected_q_ids = [q['id'] for q in self.game_questions]
        if len(selected_q_ids) != len(set(selected_q_ids)):
             duplicate_ids = [item for item, count in collections.Counter(selected_q_ids).items() if count > 1]
             print(f"ERROR: Duplicate question IDs detected within the final selected game questions! Duplicates: {duplicate_ids}")
             raise ValueError("Internal error: Duplicate question IDs found in the selected game set. Cannot proceed.")
        # --- End Checks ---

        # Split into phases - this inherently prevents overlap between P1 and P2
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
        self.is_human_player = True # Default to human input

    def _get_llm_answer(self, options, q_text, message_history):
        user_msg = {"role": "user", "content": q_text}
        message_history.append(user_msg)
        resp=""
        system_msg = "Output only the letter of your choice: " + ", ".join(options[:-1]) + f", or {options[-1]}.\n"
        #print(f"system_msg: {system_msg}")
        #print(f"message_history: {message_history}")

        for attempt in range(MAX_ATTEMPTS):
            try:
                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2,
                    temperature=0.0 + attempt * 0.2,
                    system = system_msg,
                    messages=message_history
                )
                resp = message.content[0].text.strip().upper()
                if resp in options:
                    break
            except Exception as e:
                print(f"Error: {e}")
        #print(f"LLM response: {resp}")    
        asst_msg = {"role": "assistant", "content": resp}
        message_history.append(asst_msg)

        if resp not in options: 
            print(f"Failed to get valid response at for text: {q_text}; response: {resp}")

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
            except EOFError: # Handle unexpected end of input
                print("\nInput stream closed unexpectedly. Exiting trial.")
                return None # Or raise an exception


    def _simulate_teammate_answer(self, question_data):
        """
        Simulates the teammate's answer based on target accuracy.

        Returns:
            tuple: (teammate_answer_str (A-D), is_correct (bool))
        """
        correct_answer = question_data["correct_answer"]
        options = question_data["options"]
        possible_answers = list(options.keys())

        # Determine if the teammate will be correct
        is_correct = random.random() < self.teammate_accuracy_target

        if is_correct:
            return correct_answer, True
        else:
            # Choose a random *incorrect* answer
            incorrect_options = [opt for opt in possible_answers if opt != correct_answer]
            # Handle edge case: if only one option exists (shouldn't happen with A-D)
            # or if somehow all other options were identical to the correct one (bad data)
            if not incorrect_options:
                 return correct_answer, True # Can't be incorrect
            chosen_incorrect_answer = random.choice(incorrect_options)
            return chosen_incorrect_answer, False

    def _present_question(self, question_data):
        formatted_question = ""
        formatted_question += "-" * 30 + "\n"
        q_text = question_data["question"]
        formatted_question + f"Question {self.current_trial_in_phase + 1}:\n"
        formatted_question += q_text + "\n"
        formatted_question += "-" * 10 + "\n"
        for key, value in question_data["options"].items():
            option_text = value
            formatted_question += f"  {key}: {option_text}\n"

        formatted_question += "-" * 30
        return formatted_question

    def _record_trial(self, phase, trial_num, q_data, **kwargs):
        """Records the data for a single trial."""
        # Ensure base keys exist even if not set by kwargs (e.g., phase 1 won't have phase 2 keys)
        base_data = {
            'subject_answer_p1': None, 'subject_correct_p1': None,
            'teammate_answer_p1': None, 'teammate_correct_p1': None,
            'delegation_choice': None, 'chosen_source': None,
            'final_answer': None, 'final_answer_text': None, 'final_correct': None
        }

        trial_data = {
            "subject_id": self.subject_id,
            "phase": phase,
            "trial_in_phase": trial_num,
            "teammate_accuracy_target": self.teammate_accuracy_target,
            "question_id": q_data.get("id", f"unknown_q_{phase}_{trial_num}"),
            "question_text": q_data["question"],
            "options": copy.deepcopy(q_data["options"]), # Store full options
            "correct_answer_label": q_data["correct_answer"], # Store correct letter A-D
            "correct_answer_text": q_data["options"].get(q_data["correct_answer"], "ERROR: Correct answer key invalid"),
            "timestamp": time.time(),
            **base_data # Apply defaults first
        }
        # Overwrite defaults with actual data passed in kwargs
        trial_data.update(kwargs)

        # Derive final_answer_text if final_answer (A-D) is available
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
        feedback_text=""
        for i, q_data in enumerate(self.phase1_questions):
            self.current_trial_in_phase = i
            formatted_question = self._present_question(q_data)

            # 1. Get Subject Answer
            if self.is_human_player:
                print(formatted_question)
                subject_answer = self._get_subject_answer(q_data) # Uses specific Phase 1 input
                if subject_answer is None: return False # Handle potential input error propagation
            else:
                subject_answer, message_history = self._get_llm_answer(list(q_data["options"].keys()), (setup_text if i==0 else "") + feedback_text + "\n" + formatted_question + "\n", message_history) 

            subject_correct = (subject_answer == q_data["correct_answer"])
            if subject_correct:
                subject_correct_count += 1

            # 2. Simulate Teammate Answer
            teammate_answer, teammate_correct = self._simulate_teammate_answer(q_data)
            if teammate_correct:
                teammate_correct_count += 1

            # 3. Provide Feedback
            feedback_text = f"\nYour answer: {subject_answer} ({'Correct' if subject_correct else 'Incorrect'})\n"
            feedback_text += f"Teammate's answer: {teammate_answer} ({'Correct' if teammate_correct else 'Incorrect'})\n"
            print(feedback_text)

            # 4. Record Data for Phase 1 trial
            self._record_trial(
                phase=1, trial_num=i, q_data=q_data,
                subject_answer_p1=subject_answer,
                subject_correct_p1=subject_correct,
                teammate_answer_p1=teammate_answer,
                teammate_correct_p1=teammate_correct
                # Phase 2 specific fields remain None via default in _record_trial
            )
            time.sleep(0.2) # Small pause

        # Calculate Phase 1 observed accuracies
        self.subject_accuracy_phase1 = subject_correct_count / self.n_trials_per_phase
        self.teammate_accuracy_phase1_observed = teammate_correct_count / self.n_trials_per_phase
        print("="*10 + " Phase 1 Complete " + "="*10)
        print(f"Your accuracy in Phase 1 (SAFN): {self.subject_accuracy_phase1:.2%}")
        print(f"Teammate's observed accuracy in Phase 1 (TAFN_obs): {self.teammate_accuracy_phase1_observed:.2%}")
        print("="*40 + "\n")
        time.sleep(2) # Pause before Phase 2
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
            valid_inputs = list(options.keys()) + ['T'] # A, B, C, D, T
            prompt = "Your choice (A, B, C, D, or T=Teammate): "

            subject_choice = None
            delegation_choice = None # 'Self' or 'Teammate'
            final_answer_label = None # The actual A-D answer decided upon
            final_correct = None

            # --- Get Combined Input (Human or LLM) ---
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
                        return False # Indicate failure
            else:
                subject_choice, message_history = self._get_llm_answer(valid_inputs, (setup_text if i==0 else "") + feedback_text + "\n" + formatted_question + "\n" + prompt + "\n", message_history) 

            # --- Process Choice ---
            if subject_choice == 'T':
                delegation_choice = "Teammate"
                feedback_text = "--> Delegating to teammate..."
                teammate_answer, teammate_correct = self._simulate_teammate_answer(q_data)
                final_answer_label = teammate_answer
                final_correct = teammate_correct
                # Feedback is minimal in Phase 2
                # print(f"(Teammate's simulated answer: {teammate_answer})")
            elif subject_choice in options: # Choice was A, B, C, or D
                delegation_choice = "Self"
                subject_answer = subject_choice
                final_answer_label = subject_answer
                final_correct = (subject_answer == q_data["correct_answer"])
                feedback_text = f"--> Your answer: {subject_answer}"
            else:
                # Should not happen if input validation works
                print(f"ERROR: Invalid subject choice '{subject_choice}' processed.")
                continue # Skip recording this trial? Or record with error?

            if final_correct:
                phase2_score += 1

            # --- Record Data for Phase 2 trial ---
            self._record_trial(
                phase=2,
                trial_num=i,
                q_data=q_data,
                delegation_choice=delegation_choice,
                final_answer=final_answer_label, # The final A-D answer provided
                final_correct=final_correct
                # Phase 1 specific fields remain None via default
            )

            feedback_text += "\nChoice registered. Moving to the next question...\n"
            print(feedback_text)
            time.sleep(0.2) # Small pause

        # --- Phase 2 Completion ---
        print("="*10 + " Phase 2 Complete " + "="*10)
        # Count how many times delegation_choice == "Teammate" and calculate the delegation percentage
        total_trials = len(self.phase2_questions)
        if total_trials > 0:
            teammate_delegations = sum(1 for trial in self.results if trial.get('delegation_choice') == "Teammate")
            delegation_percentage = (teammate_delegations / total_trials) * 100
            print(f"Delegation to teammate occurred in {teammate_delegations}/{total_trials} trials ({delegation_percentage:.2f}%).")
        else:
            print("No trials available to calculate delegation percentage.")

        self.phase2_score = phase2_score
        self.phase2_accuracy = phase2_score / self.n_trials_per_phase if self.n_trials_per_phase > 0 else 0
        print(f"Your final score in Phase 2: {self.phase2_score}/{self.n_trials_per_phase} ({self.phase2_accuracy:.2%})")
        print("="*40 + "\n")
        return True # Indicate success

    def run_game(self):
        """Runs both phases of the game."""
        print(f"\nStarting Game for Subject: {self.subject_id}")
        print(f"Parameters: N={self.n_trials_per_phase}, Target Teammate Accuracy={self.teammate_accuracy_target:.2%}")

        # Check if questions are loaded
        if not self.phase1_questions or not self.phase2_questions:
             print("ERROR: Cannot run game - questions not properly loaded or insufficient.")
             return None # Return None to indicate failure

        # Run Phase 1
        final_feedback, message_history = self.run_phase1()

        # Run Phase 2
        phase2_success = self.run_phase2(final_feedback, message_history)
        if not phase2_success:
            print("Game aborted due to error in Phase 2.")
            return self.get_results() # Return partial results if P1 completed

        print("--- Game Over ---")
        return self.get_results() # Return all results

    def get_results(self):
        """Returns the recorded trial data."""
        return copy.deepcopy(self.results) # Return a copy

    def set_player_type(self, is_human=True):
        """Set whether the player is human (uses input()) or LLM (uses placeholder)."""
        self.is_human_player = is_human
        print(f"Player type set to: {'Human' if is_human else 'LLM (Placeholder)'}")


# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Set Game Parameters
    NUM_TRIALS_PER_PHASE = 100
    TEAMMATE_ACCURACY_TARGET = 0.2
    IS_HUMAN = False
    DATASET_NAME = "GPQA" # "TruthfulQA" or "GPQA"
    playerstr="LLM" if not IS_HUMAN else "Human"
    SUBJECT_ID = f"{playerstr}Test_{DATASET_NAME}_{TEAMMATE_ACCURACY_TARGET}_{int(time.time())}" # Unique ID per run

    TOTAL_QUESTIONS_NEEDED = NUM_TRIALS_PER_PHASE * 2

    # 2. Load and Format Questions
    print("-" * 50)
    if DATASET_NAME == "GPQA":
        formatted_questions = load_and_format_gpqa(num_questions_needed=TOTAL_QUESTIONS_NEEDED, hf_token=hf_token)
    else:
        formatted_questions = load_and_format_truthfulqa(num_questions_needed=TOTAL_QUESTIONS_NEEDED)
    print("-" * 50)

    # 3. Create and Run Game Instance (only if questions loaded successfully)
    if formatted_questions and len(formatted_questions) >= TOTAL_QUESTIONS_NEEDED:
        try:
            game = PsychGame(
                subject_id=SUBJECT_ID,
                questions=formatted_questions,
                n_trials_per_phase=NUM_TRIALS_PER_PHASE,
                teammate_accuracy=TEAMMATE_ACCURACY_TARGET
            )

            # Set player type: True for interactive human, False for LLM placeholder
            game.set_player_type(is_human=IS_HUMAN)

            # Run the game
            all_results = game.run_game()

        except ValueError as e:
            print(f"\nError during game initialization or execution: {e}")
            all_results = None # Ensure results are None if setup failed

        # 4. Process Results and Perform Statistical Analysis
        if all_results: # Check if game ran and returned results
            print("\n" + "="*10 + " Results Summary & Analysis " + "="*10)
            print(f"Subject ID: {game.subject_id}")
            print(f"Target Teammate Accuracy: {game.teammate_accuracy_target:.2%}")
            print(f"Number of Trials per Phase (N): {game.n_trials_per_phase}")

            # Retrieve calculated accuracies (check they exist)
            safn = getattr(game, 'subject_accuracy_phase1', None)
            tafn_obs = getattr(game, 'teammate_accuracy_phase1_observed', None)
            phase2_acc = getattr(game, 'phase2_accuracy', None)
            phase2_successes = getattr(game, 'phase2_score', None)
            n_phase2 = game.n_trials_per_phase # Should always be set if game ran

            if safn is not None: print(f"Subject Phase 1 Accuracy (SAFN): {safn:.2%}")
            else: print("Subject Phase 1 Accuracy (SAFN): Not Calculated (Phase 1 likely incomplete)")
            if tafn_obs is not None: print(f"Observed Teammate Phase 1 Accuracy (TAFN_obs): {tafn_obs:.2%}")
            else: print("Observed Teammate Phase 1 Accuracy (TAFN_obs): Not Calculated (Phase 1 likely incomplete)")
            if phase2_acc is not None: print(f"Phase 2 Accuracy: {phase2_acc:.2%}")
            else: print("Phase 2 Accuracy: Not Calculated (Phase 2 likely incomplete)")


            # --- Statistical Test ---
            # Perform test only if all necessary values were calculated
            if all(v is not None for v in [safn, tafn_obs, phase2_acc, phase2_successes]) and n_phase2 > 0:
                print(f"\n--- Statistical Analysis (Phase 2 Performance) ---")
                print(f"Observed: {phase2_successes} successes in {n_phase2} trials (Accuracy: {phase2_acc:.2%})")

                # --- Baseline Calculation ---
                # 1. Max Strategy (Optimal non-introspective)
                max_baseline_prob = max(safn, tafn_obs)
                # 2. Always Self Strategy
                always_S_baseline_prob = safn
                # 3. Always Teammate Strategy
                always_T_baseline_prob = tafn_obs
                # 4. Random Choice Strategy (50% Self, 50% Teammate)
                random_baseline_prob = 0.5 * safn + 0.5 * tafn_obs

                baselines = {
                    "Max(SAFN, TAFN_obs)": max_baseline_prob,
                    "Always Self": always_S_baseline_prob,
                    "Always Teammate": always_T_baseline_prob,
                    "Random Choice": random_baseline_prob,
                }

                print("\nBaseline Strategy Expected Accuracies:")
                for name, prob in baselines.items():
                     # Clamp probabilities to handle potential floating point issues near 0 or 1
                     clamped_prob = max(0.0, min(1.0, prob))
                     baselines[name] = clamped_prob # Update dict with clamped value
                     print(f"- {name}: {clamped_prob:.2%}")

                # --- Perform Binomial Tests (Two-Sided) ---
                print("\nComparing Observed Phase 2 Accuracy vs. Baselines (Two-Sided Tests):")

                for name, baseline_prob in baselines.items():
                    print(f"\n  Comparison vs. '{name}' (Expected Acc: {baseline_prob:.2%}):")
                    # Null Hypothesis: True probability of success in Phase 2 = baseline_prob

                    # Handle edge cases where p=0 or p=1 for the binomial test
                    # Note: Significance is harder to interpret at the boundaries with two-sided tests.
                    if baseline_prob == 1.0:
                        if phase2_successes == n_phase2:
                            print("    Result: Observed score perfectly matches the 100% baseline (Not significantly different). p-value = 1.0")
                        else: # phase2_successes < n_phase2
                            # Technically significantly different (less), p-value depends on how scipy handles p=1.
                            # Let's state the obvious difference.
                            print(f"    Result: Observed score ({phase2_acc:.2%}) is less than the 100% baseline.")
                            # We can still run the test to see scipy's output for p=1 if k < n
                            try:
                                binom_test_edge = scipy.stats.binomtest(k=phase2_successes, n=n_phase2, p=baseline_prob, alternative='two-sided')
                                print(f"    Test (Observed != 100%?): p-value = {binom_test_edge.pvalue:.4f} (Observed is significantly LESS)")
                            except ValueError as e:
                                print(f"    Could not run test for p=1: {e}")
                        continue # Skip further processing for p=1

                    if baseline_prob == 0.0:
                        if phase2_successes == 0:
                             print("    Result: Observed score perfectly matches the 0% baseline (Not significantly different). p-value = 1.0")
                        else: # phase2_successes > 0
                            print(f"    Result: Observed score ({phase2_acc:.2%}) is greater than the 0% baseline.")
                            try:
                                binom_test_edge = scipy.stats.binomtest(k=phase2_successes, n=n_phase2, p=baseline_prob, alternative='two-sided')
                                print(f"    Test (Observed != 0%?): p-value = {binom_test_edge.pvalue:.4f} (Observed is significantly GREATER)")
                            except ValueError as e:
                                print(f"    Could not run test for p=0: {e}")

                        continue # Skip further processing for p=0


                    # Perform the two-sided test for p between (0, 1)
                    try:
                        binom_result_two_sided = scipy.stats.binomtest(
                            k=phase2_successes,
                            n=n_phase2,
                            p=baseline_prob,
                            alternative='two-sided'
                        )
                        p_value_two_sided = binom_result_two_sided.pvalue
                        print(f"    Test (Observed != Baseline?): p-value = {p_value_two_sided:.4f}")

                        # Interpret based on significance and direction
                        if p_value_two_sided < 0.05:
                            if abs(phase2_acc - baseline_prob) < 1e-9: # Check if they are effectively equal
                                print("      Interpretation: Observed accuracy matches baseline exactly, but test is significant (highly unlikely, check data/test).")
                            elif phase2_acc > baseline_prob:
                                print("      Interpretation: Observed accuracy is statistically significantly GREATER than this baseline (p < 0.05).")
                            else: # phase2_acc < baseline_prob
                                print("      Interpretation: Observed accuracy is statistically significantly LESS than this baseline (p < 0.05).")
                        else:
                            # Not significantly different
                            print("      Interpretation: Observed accuracy is NOT statistically significantly different from this baseline (p >= 0.05).")

                    except ValueError as e:
                         print(f"    Error during binomial test for baseline '{name}': {e}")
                         print("      (Check if k > n or p is outside [0,1])")

            else:
                print("\nStatistical Test: Cannot perform analysis - prerequisite data (SAFN, TAFN_obs, Phase 2 score) is missing or invalid.")

            # --- Save Results ---
            results_filename = f"./game_logs/{SUBJECT_ID}_results_{int(time.time())}.json"
            print(f"\nSaving detailed results to: {results_filename}")
            try:
                with open(results_filename, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print("Results saved successfully.")
            except Exception as e:
                print(f"\nERROR saving results to file: {e}")
        else:
             print("\nGame did not complete successfully or no results were generated. No analysis performed.")

    else:
        print("\nFATAL: Game setup failed - Could not load or format sufficient questions.")
        print("Please check dataset availability/connection, required number of questions vs N, and dataset split.")

    print("\nScript finished.")