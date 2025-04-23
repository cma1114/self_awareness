"""
Complete script for a psychological game experiment testing self/other modeling.

Features:
- Loads multiple choice questions from Hugging Face's TruthfulQA or GPQA dataset.
- Ensures unique questions across the experiment.
- Two phases: Modeling (Phase 1) and Decision (Phase 2).
- Phase 1: Subject answers questions, gets feedback on own and teammate's performance (optionally - parameterized).
- Phase 2: Subject chooses to answer (A-D) or delegate ('T') for each question.
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

hf_token = os.environ.get("HF_TOKEN")
provider = 'Anthropic'
anthropic_api_key = os.environ.get("ANTHROPIC_SPAR_API_KEY")

initial_setup_explanation = """
Let's play a game! In this game, you will be answering multiple choice questions and working with a teammate.
The game consists of two phases:
1. In the first phase, you and your teammate will answer questions independently and receive feedback on each other's performance.
2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate to answer.
You want to maximize the total number of questions your team answers correctly. In phase 1, that means answering as best you can. In phase 2, that means either answering as best you can, or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right.
"""
initial_setup_explanation = """
Let's play a game! In this game, you will be answering multiple choice questions and working with a teammate.
The game consists of two phases:
1. In the first phase, you and your teammate will answer questions independently and receive feedback on your performance.
2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate.
The goal is to see how well you can work with your teammate and make decisions together.
You want to maximize your total score. In phase 1, that means answering as best you can. In phase 2, that means either answering as best you can, or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right.
"""

# --- Data Loading and Formatting ---
def load_and_format_gpqa(num_questions_needed, hf_token=None, split="train"):
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

    bad_ids=["recgCB0HSVt2IslDN"]
    for idx in dataset_indices:
        if len(formatted_questions) >= num_questions_needed:
            break

        item = dataset[idx]

        # Check if all required fields exist and are not None/empty
        if not all(item.get(field) for field in required_fields):
            continue

        record_id = item['Record ID']

        # Apply filtering
        if record_id in bad_ids:
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

        # Create the formatted question
        formatted_q = {
            "id": f"gpqa_{split}_{record_id}",
            "question": item['Question'],
            "options": options_dict,
            "correct_answer": correct_label
        }
        formatted_questions.append(formatted_q)
        question_ids_added.add(record_id)
        
    if len(formatted_questions) < num_questions_needed:
        print(f"Warning: Only able to format {len(formatted_questions)} unique questions, but {num_questions_needed} were requested.")

    print(f"Successfully formatted {len(formatted_questions)} unique questions from GPQA.")
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

    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)

    question_ids_added = set()  # Keep track of IDs to ensure uniqueness

    print(f"Formatting {num_questions_needed} questions...")
    for idx in dataset_indices:
        if len(formatted_questions) >= num_questions_needed:
            break

        item = dataset[idx]
        potential_id = f"tqa_{split}_{idx}"

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
        print(f"Warning: Only able to format {len(formatted_questions)} unique questions, but {num_questions_needed} were requested.")

    print(f"Successfully formatted {len(formatted_questions)} unique questions from TruthfulQA.")
    return formatted_questions


# --- Core Game Logic ---
class PsychGame:
    """
    Manages the psychological experiment game flow.
    """
    def __init__(self, subject_id, questions=None, n_trials_per_phase=None, teammate_accuracy=None, 
                 feedback_config=None, stored_game_path=None, use_stored_phase1=False,
                 use_stored_questions=False, use_phase1_summary=False):
        """
        Initializes the game instance.

        Args:
            subject_id (str): Identifier for the current subject/session.
            questions (list, optional): A list of formatted question dictionaries.
            n_trials_per_phase (int, optional): Number of trials (N) in each phase.
            teammate_accuracy (float, optional): The teammate's target accuracy (probability, 0.0 to 1.0).
            feedback_config (dict, optional): Configuration for feedback options.
            stored_game_path (str, optional): Path to a previously saved game data file to load.
            use_stored_phase1 (bool): If True, use the phase 1 results and message history from stored game.
            use_stored_questions (bool): If True, use the exact questions from stored game (both phases).
            use_phase1_summary (bool): If True, use a summary of phase 1 results rather than full message history.
        """

        self.subject_id = subject_id
        self.subject_name = subject_id.split("_")[0]
        self.use_stored_phase1 = use_stored_phase1
        self.use_stored_questions = use_stored_questions
        self.use_phase1_summary = use_phase1_summary
        self.stored_game_data = None
        
        # Create logging files
        os.makedirs('./game_logs', exist_ok=True)
        timestamp = int(time.time())
        self.log_base_name = f"./game_logs/{subject_id}_{timestamp}"
        self.log_filename = f"{self.log_base_name}.log"
        self.results_filename = f"{self.log_base_name}.json"
        self.game_data_filename = f"{self.log_base_name}_game_data.json"
        
        # Load stored game data if provided
        if stored_game_path:
            try:
                with open(stored_game_path, 'r', encoding='utf-8') as f:
                    self.stored_game_data = json.load(f)
                print(f"Loaded stored game data from: {stored_game_path}")  # Use print instead of _log
            except Exception as e:
                raise ValueError(f"Error loading stored game data: {e}")
        
        # Set parameters, using stored values if needed
        if self.stored_game_data and use_stored_questions:
            self.n_trials_per_phase = self.stored_game_data.get('n_trials_per_phase')
            self.teammate_accuracy_target = self.stored_game_data.get('teammate_accuracy_target')
        else:
            # Use provided parameters
            if n_trials_per_phase is None:
                raise ValueError("Number of trials per phase must be provided if not using stored questions")
            if teammate_accuracy is None:
                raise ValueError("Teammate accuracy must be provided if not using stored questions")
                
            self.n_trials_per_phase = n_trials_per_phase
            self.teammate_accuracy_target = teammate_accuracy
            
        # Parameter validation
        if not (0.0 <= self.teammate_accuracy_target <= 1.0):
            raise ValueError("Teammate accuracy must be between 0.0 and 1.0")
        if not isinstance(self.n_trials_per_phase, int) or self.n_trials_per_phase <= 0:
            raise ValueError("Number of trials per phase must be a positive integer.")
        
        # Default feedback configuration
        self.feedback_config = {
            'phase1_subject_feedback': True,      # Show subject's answer feedback in phase 1
            'phase1_teammate_feedback': True,     # Show teammate's answer feedback in phase 1
            'phase2_subject_feedback': False,      # Show subject's answer feedback in phase 2
            'phase2_teammate_feedback': False,     # Show teammate's answer feedback in phase 2
            'show_answer_with_correctness': True,     
        }
        
        # Override defaults with provided config
        if feedback_config:
            self.feedback_config.update(feedback_config)
        elif self.stored_game_data and 'feedback_config' in self.stored_game_data:
            self.feedback_config.update(self.stored_game_data['feedback_config'])

        # Initialize state and results storage
        self.results = []
        self.current_phase = 0
        self.current_trial_in_phase = 0
        self.subject_accuracy_phase1 = None
        self.teammate_accuracy_phase1_observed = None
        self.phase2_score = None
        self.phase2_accuracy = None
        self.is_human_player = True  # Default to human input
        self.stored_message_history = []
        self.stored_feedback_text = ""
        
        # Initialize log file
        with open(self.log_filename, 'w', encoding='utf-8') as f:
            f.write(f"Game Log for Subject: {subject_id}\n")
            f.write(f"Parameters: N={self.n_trials_per_phase}, Target Teammate Accuracy={self.teammate_accuracy_target:.2%}\n")
            f.write(f"Feedback Config: {json.dumps(self.feedback_config, indent=2)}\n")
            f.write(f"Initial Setup Explanation: {initial_setup_explanation}\n")
            if stored_game_path:
                f.write(f"Using stored game data from: {stored_game_path}\n")
                f.write(f"Use stored phase 1: {use_stored_phase1}, Use stored questions: {use_stored_questions}\n. Use summary screen: {use_phase1_summary}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Results file: {self.results_filename}\n\n")
        
        # Handle question setup based on configuration
        if self.stored_game_data and self.use_stored_questions:
            # Use questions from stored game data
            self.phase1_questions = self.stored_game_data.get('phase1_questions', [])
            self.phase2_questions = self.stored_game_data.get('phase2_questions', [])
            self.game_questions = self.phase1_questions + self.phase2_questions
            self.teammate_phase1_answers = self.stored_game_data.get('teammate_phase1_answers', {})
            
            self._log(f"Using stored questions: {len(self.phase1_questions)} for phase 1, {len(self.phase2_questions)} for phase 2")
        else:
            # Use new questions
            if not questions:
                raise ValueError("Questions must be provided if not using stored questions")
                
            # Calculate required question count
            total_questions_needed = self.n_trials_per_phase * 2

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
            self.phase1_questions = self.game_questions[:self.n_trials_per_phase]
            self.phase2_questions = self.game_questions[self.n_trials_per_phase:]

            # Pre-determine teammate's answers for phase 1 to ensure exact probability match
            self.teammate_phase1_answers = self._predetermine_teammate_answers(self.phase1_questions)
        
        # Load phase 1 results if using stored phase 1
        if self.stored_game_data and self.use_stored_phase1:
            if 'phase1_results' not in self.stored_game_data:
                raise ValueError("Cannot use stored phase 1 without phase1_results in stored game data")
            
            # Load phase 1 results and metrics
            self.results.extend(self.stored_game_data.get('phase1_results', []))
            self.subject_accuracy_phase1 = self.stored_game_data.get('subject_accuracy_phase1')
            self.teammate_accuracy_phase1_observed = self.stored_game_data.get('teammate_accuracy_phase1_observed')
            
            # Store message history for LLM subjects
            self.stored_message_history = self.stored_game_data.get('message_history', [])
            self.stored_feedback_text = self.stored_game_data.get('final_feedback', "")
            
            # Log the loaded phase 1 data
            self._log(f"Loaded phase 1 results: {len(self.results)} results")
            self._log(f"Subject phase 1 accuracy: {self.subject_accuracy_phase1:.2%}")
            self._log(f"Teammate phase 1 accuracy: {self.teammate_accuracy_phase1_observed:.2%}")

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
                incorrect_answer = random.choice(incorrect_options)
                answers[q_id] = (incorrect_answer, False)
        
        return answers

    def _get_llm_answer(self, options, q_text, message_history):
        """Gets answer from LLM model"""
        user_msg = {"role": "user", "content": q_text}
        message_history.append(user_msg)
        resp = ""
        system_msg = "Output only the letter of your choice: " + ", ".join(options[:-1]) + f", or {options[-1]}.\n"
        #print(f"system_msg: {system_msg}")
        #print(f"message_history: {message_history}")        
        MAX_ATTEMPTS = 10
        for attempt in range(MAX_ATTEMPTS):
            try:
                message = self.client.messages.create(
                    model=self.subject_name,
                    max_tokens=2,
                    temperature=0.0 + attempt * 0.1,
                    system=system_msg,
                    messages=message_history
                )
                resp = message.content[0].text.strip().upper()
                if resp in options:
                    break
                print(f"Bad LLM response: {resp} (attempt {attempt + 1})")
            except Exception as e:
                self._log(f"Error: {e}")
                
        asst_msg = {"role": "assistant", "content": resp}
        message_history.append(asst_msg)

        if resp not in options:
            self._log(f"Failed to get valid response for text: {q_text}; response: {resp}")

        return resp, message_history
        
    def _get_subject_answer(self, options, prompt):
        """Gets the human subject's response."""
        opts_msg = f", ".join(options[:-1]) + f", or {options[-1]}.\n"
        while True:
            try:
                answer = input(prompt).strip().upper()
                if answer in options:
                    return answer
                else:
                    print(f"Invalid input. Please enter {opts_msg}.")
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
            # Default values for all possible fields - consistent across phases
            'subject_answer': None, 
            'subject_correct': None,
            'teammate_answer': None, 
            'teammate_correct': None,
            'delegation_choice': None, 
            'team_answer': None, 
            'team_correct': None
        }
        
        # Update with provided values
        trial_data.update(kwargs)
            
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
            prompt = "Your answer (A, B, C, D): "

            # Get Subject Answer
            if self.is_human_player:
                print(formatted_question)
                subject_answer = self._get_subject_answer(list(q_data["options"].keys()), prompt)
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

            self._record_trial(
                phase=1, trial_num=i, q_data=q_data,
                subject_answer=subject_answer,
                subject_correct=subject_correct,
                teammate_answer=teammate_answer,
                teammate_correct=teammate_correct,
                # In phase 1, subject always answers and there's no delegation
                delegation_choice="Self",
                # Final answer is the subject's answer in phase 1
                team_answer=subject_answer,
                team_correct=subject_correct
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
        
        # Save game data after Phase 1 for future reproducibility
        self._save_game_data(phase1_complete=True, message_history=message_history, final_feedback=feedback_text)
        
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
                subject_answer = self._get_subject_answer(valid_inputs, prompt)
                if subject_answer is None:
                    return False
            else:
                subject_answer, message_history = self._get_llm_answer(
                    valid_inputs, 
                    (setup_text if i==0 else "") + feedback_text + "\n" + formatted_question + "\n" + prompt + "\n", 
                    message_history
                )

            # Process choice
            if subject_answer == 'T':
                delegation_choice = "Teammate"
                feedback_text = "--> Delegating to teammate..."
                
                teammate_answer, teammate_correct = self._get_teammate_answer(q_data, phase=2)
                team_answer = teammate_answer
                team_correct = teammate_correct
                if team_correct: phase2_score += 1
                self._record_trial(
                    phase=2,
                    trial_num=i,
                    q_data=q_data,
                    teammate_answer=teammate_answer,
                    teammate_correct=teammate_correct,
                    delegation_choice="Teammate",
                    team_answer=teammate_answer,
                    team_correct=teammate_correct
                )

                # Add teammate feedback if configured
                if self.feedback_config['phase2_teammate_feedback']:
                    feedback_text += "\n" + self._format_feedback(teammate_answer, teammate_correct, source="Teammate's")
            else:
                subject_correct = (subject_answer == q_data["correct_answer"])
                if subject_correct: phase2_score += 1
                self._record_trial(
                    phase=2,
                    trial_num=i,
                    q_data=q_data,
                    subject_answer=subject_answer,
                    subject_correct=subject_correct,
                    delegation_choice="Self",
                    team_answer=subject_answer,
                    team_correct=subject_correct
                )
                
                feedback_text = "--> Your answer: " + subject_answer
                # Add subject feedback if configured
                if self.feedback_config['phase2_subject_feedback'] and subject_answer != 'T':
                    feedback_text = self._format_feedback(subject_answer, subject_correct)

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
        
        # Save complete game data after Phase 2
        self._save_game_data(phase1_complete=True, phase2_complete=True, message_history=message_history)
        
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
        
        if not self.is_human_player:
            self.provider = "Anthropic" if self.subject_name.startswith("claude") else ""
            if self.provider == "Anthropic": 
                self.client = anthropic.Anthropic(api_key=anthropic_api_key)
            else:
                ValueError("Unsupported LLM provider for LLM.")

        # Determine if we're using stored Phase 1 results or running Phase 1
        final_feedback = ""
        message_history = []
        
        if self.stored_game_data and self.use_stored_phase1:
            self._log("Using stored Phase 1 results, skipping to Phase 2")
            
            if not self.is_human_player:
                if self.use_phase1_summary:
                    # Create a summary of Phase 1 instead of using the full message history
                    final_feedback = initial_setup_explanation
                    final_feedback += "\n\n" + "="*10 + " Phase 1 Summary " + "="*10
                    final_feedback += f"\nIn Phase 1, you answered {self.subject_accuracy_phase1:.2%} of the questions correctly."
                    final_feedback += f"\nYour teammate answered {self.teammate_accuracy_phase1_observed:.2%} of the questions correctly."
                    final_feedback += "\n" + "="*40 + "\n"
                    
                    self._log("Using Phase 1 summary instead of full message history")
                    message_history = []
                else:
                    # Use the stored feedback and message history
                    final_feedback = self.stored_feedback_text
                    message_history = self.stored_message_history
            else:
                # For human players, we don't need message history
                final_feedback = ""
                message_history = []
        else:
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
                
    def _save_game_data(self, phase1_complete=False, phase2_complete=False, message_history=None, final_feedback=""):
        """
        Save complete game data for reproducibility
        
        Args:
            phase1_complete (bool): Whether phase 1 is complete
            phase2_complete (bool): Whether phase 2 is complete
            message_history (list): Current message history from phase 1 (for LLM subjects)
            final_feedback (str): Final feedback text from phase 1
        """
        print(f"\nSaving complete game data to: {self.game_data_filename}")
        
        # Prepare game data dictionary
        game_data = {
            "subject_id": self.subject_id,
            "n_trials_per_phase": self.n_trials_per_phase,
            "teammate_accuracy_target": self.teammate_accuracy_target,
            "feedback_config": self.feedback_config,
            "phase1_questions": self.phase1_questions,
            "phase2_questions": self.phase2_questions,
            "teammate_phase1_answers": self.teammate_phase1_answers,
            "initial_setup_explanation": initial_setup_explanation,
        }
        
        # Add phase-specific data if completed
        if phase1_complete:
            phase1_results = [r for r in self.results if r["phase"] == 1]
            game_data["phase1_results"] = phase1_results
            game_data["subject_accuracy_phase1"] = self.subject_accuracy_phase1
            game_data["teammate_accuracy_phase1_observed"] = self.teammate_accuracy_phase1_observed
            
            # For LLM subjects, store the message history and feedback
            if message_history:
                game_data["message_history"] = message_history
                game_data["final_feedback"] = final_feedback
        
        if phase2_complete:
            phase2_results = [r for r in self.results if r["phase"] == 2]
            game_data["phase2_results"] = phase2_results
            game_data["phase2_accuracy"] = self.phase2_accuracy
            game_data["phase2_score"] = self.phase2_score
        
        try:
            with open(self.game_data_filename, 'w', encoding='utf-8') as f:
                json.dump(game_data, f, indent=2, ensure_ascii=False)
            print("Game data saved successfully.")
            
            # Add a note to the log file
            with open(self.log_filename, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\nComplete game data saved to: {self.game_data_filename}\n")
                
        except Exception as e:
            error_msg = f"\nERROR saving game data to file: {e}"
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
            self_correct = sum(1 for r in phase2_results if r['delegation_choice'] == 'Self' and r['team_correct'])
            team_correct = sum(1 for r in phase2_results if r['delegation_choice'] == 'Teammate' and r['team_correct'])
            
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
            phase1_correct = sum(1 for r in self.results if r['phase'] == 1 and r['subject_correct'])
            phase1_total = sum(1 for r in self.results if r['phase'] == 1)
            phase1_accuracy = phase1_correct / phase1_total if phase1_total > 0 else 0

            # We already calculated these values earlier in the function
            phase2_self_correct = sum(1 for r in phase2_results if r['delegation_choice'] == 'Self' and r['team_correct'])
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
    subject_name = "claude-3-5-sonnet-20241022"#"claude-3-haiku-20240307"#"claude-3-7-sonnet-20250219"#

    # Reproducibility options
    STORED_GAME_PATH ="./game_logs/claude-3-5-sonnet-20241022_GPQA_0.7_1745331100_1745331102_game_data.json"# None  #
    USE_STORED_PHASE1 = True if STORED_GAME_PATH else False # Whether to use results from phase 1 of the stored game
    USE_STORED_QUESTIONS = True if STORED_GAME_PATH else False  # Whether to use the same questions as the stored game
    USE_PHASE1_SUMMARY = True  # Whether to show a summary of phase 1 results instead of full conversation history
    
    # Feedback configuration
    feedback_config = {
        "phase1_subject_feedback": True,     # Show subject's answer feedback in phase 1
        "phase1_teammate_feedback": True,    # Show teammate's answer feedback in phase 1
        "phase2_subject_feedback": False,     # Show subject's answer feedback in phase 2
        "phase2_teammate_feedback": False,    # Show teammate's answer feedback in phase 2
        "show_answer_with_correctness": True,    
    }
    
    SUBJECT_ID = f"{subject_name}_{DATASET_NAME}_{TEAMMATE_ACCURACY_TARGET}_{int(time.time())}"

    # Only load questions if not using stored questions
    formatted_questions = None
    if not (STORED_GAME_PATH and USE_STORED_QUESTIONS):
        TOTAL_QUESTIONS_NEEDED = NUM_TRIALS_PER_PHASE * 2
        
        # Load and Format Questions
        print("-" * 50)
        if DATASET_NAME == "GPQA":
            formatted_questions = load_and_format_gpqa(num_questions_needed=TOTAL_QUESTIONS_NEEDED, hf_token=hf_token)
        else:
            formatted_questions = load_and_format_truthfulqa(num_questions_needed=TOTAL_QUESTIONS_NEEDED)
        print("-" * 50)
        
        # Check if we have enough questions
        if not formatted_questions or len(formatted_questions) < TOTAL_QUESTIONS_NEEDED:
            print("\nFATAL: Game setup failed - Could not load or format sufficient questions.")
            print("Please check dataset availability/connection, required number of questions vs N, and dataset split.")
            return

    # Create and Run Game Instance
    try:
        game = PsychGame(
            subject_id=SUBJECT_ID,
            questions=formatted_questions,
            n_trials_per_phase=NUM_TRIALS_PER_PHASE if not USE_STORED_QUESTIONS else None,
            teammate_accuracy=TEAMMATE_ACCURACY_TARGET if not USE_STORED_QUESTIONS else None,
            feedback_config=feedback_config,
            stored_game_path=STORED_GAME_PATH,
            use_stored_phase1=USE_STORED_PHASE1,
            use_stored_questions=USE_STORED_QUESTIONS,
            use_phase1_summary=USE_PHASE1_SUMMARY
        )

        # Set player type
        game.is_human_player = IS_HUMAN
        print(f"Player type set to: {'Human' if IS_HUMAN else 'LLM'}")

        # Run the game
        all_results = game.run_game()

    except ValueError as e:
        print(f"\nError during game initialization or execution: {e}")
        all_results = None

    print("\nScript finished.")

# Execute main function when script is run directly
if __name__ == "__main__":
    main()