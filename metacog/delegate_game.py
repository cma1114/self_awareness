"""
Complete script for a psychological game experiment testing self/other modeling.

Features:
- Loads multiple choice questions from Hugging Face's TruthfulQA or GPQA dataset.
- Ensures unique questions across the experiment.
- Two phases: Modeling (Phase 1) and Decision (Phase 2).
- Phase 1: Subject answers questions, gets feedback on own and teammate's performance (optionally - parameterized).
- Phase 2: Subject chooses to answer (A-D) or delegate ('T') for each question.
"""

import random
import time
import copy
import json
import os
from load_and_format_datasets import load_and_format_dataset
from base_game_class import BaseGameClass
import scipy.stats



# --- Core Game Logic ---
class PsychGame(BaseGameClass):
    """
    Manages the psychological experiment game flow.
    """
    def __init__(self, subject_id, subject_name, dataset_name, is_human_player=True, n_trials_phase1=None, n_trials_phase2=None, 
                 teammate_accuracy_phase1=None, teammate_accuracy_phase2=None, feedback_config=None, stored_game_path=None, 
                 skip_phase1=False, show_phase1_summary=False, show_full_phase1_history=False,
                 phase1_summary=None, initial_setup_explanation="", use_phase2_data=True,
                 override_subject_accuracy=None, override_teammate_accuracy=None, redact_phase1_answers=False):
        """
        Initializes the game instance.

        Args:
            subject_id (str): Identifier for the current subject/session.
            n_trials_phase1 (int, optional): Number of trials (N) in Phase 1.
            n_trials_phase2 (int, optional): Number of trials (N) in Phase 2.
            teammate_accuracy_phase1 (float, optional): The teammate's target accuracy in Phase 1 (probability, 0.0 to 1.0).
            teammate_accuracy_phase2 (float, optional): The teammate's target accuracy in Phase 2 (probability, 0.0 to 1.0).
            feedback_config (dict, optional): Configuration for feedback options.
            stored_game_path (str, optional): Path to a previously saved game data file to load.
            skip_phase1 (bool): If True, skip running phase 1 and use stored results.
            show_phase1_summary (bool): Whether to show a summary of phase 1 performance.
            show_full_phase1_history (bool): Whether to show full phase 1 conversation history.
            phase1_summary (dict, optional): Controls what appears in the phase 1 summary. Include 's_acc' key to show 
                                     subject accuracy (value can be None to use real value or a float to override).
                                     Include 't_acc' key to show teammate accuracy (same value behavior). 
                                     If not provided or key is missing, that accuracy line won't be shown.
            use_phase2_data (bool): If False, don't use phase2 questions from stored game data. Default is True.
            override_subject_accuracy (float, optional): If not None, modifies the message history to show this accuracy rate
                                     for the subject in phase 1. Value should be between 0.0 and 1.0.
            override_teammate_accuracy (float, optional): If not None, modifies the message history to show this accuracy rate
                                     for the teammate in phase 1. Value should be between 0.0 and 1.0.
            redact_phase1_answers (bool): If True, replaces the subject's phase 1 answers with "[redacted]" in the message history.
        """
        super().__init__(subject_id, subject_name, is_human_player, "delegate_game_logs")

        self.skip_phase1 = skip_phase1
        self.show_phase1_summary = show_phase1_summary  
        self.show_full_phase1_history = show_full_phase1_history
        self.phase1_summary = phase1_summary
        self.stored_game_data = None
        self.initial_setup_explanation = initial_setup_explanation
        self.use_phase2_data = use_phase2_data
        self.override_subject_accuracy = override_subject_accuracy
        self.override_teammate_accuracy = override_teammate_accuracy
        self.redact_phase1_answers = redact_phase1_answers
        self.dataset_name = dataset_name

        # Default feedback configuration
        self.feedback_config = {
            'phase1_subject_feedback': True,      # Show subject's answer feedback in phase 1
            'phase1_teammate_feedback': True,     # Show teammate's answer feedback in phase 1
            'phase2_subject_feedback': False,      # Show subject's answer feedback in phase 2
            'phase2_teammate_feedback': False,     # Show teammate's answer feedback in phase 2
            'show_answer_with_correctness': True,     
        }
        # Initialize state and results storage
        self.results = []
        self.current_phase = 0
        self.current_trial_in_phase = 0
        self.subject_accuracy_phase1 = None
        self.teammate_accuracy_phase1 = None
        self.phase2_score = None
        self.phase2_accuracy = None
        self.stored_message_history = []
        self.orig_phase1_message_history = None
        self.stored_feedback_text = ""
        
        # Load stored game data if provided
        if stored_game_path:
            try:
                with open(stored_game_path, 'r', encoding='utf-8') as f:
                    self.stored_game_data = json.load(f)
                self._log(f"Loaded stored game data from: {stored_game_path}") 
            except Exception as e:
                raise ValueError(f"Error loading stored game data: {e}")
        
            ## Set trial counts and teammate accuracy from stored data when appropriate, load questions if needed
            # Always use phase1 trials count from stored data
            self.n_trials_phase1 = min(self.stored_game_data.get('n_trials_phase1'), n_trials_phase1)
            self.phase1_questions = self.stored_game_data.get('phase1_questions', [])[:self.n_trials_phase1]
            
            # For phase2, use provided trial count from stored data unless overridden by USE_PHASE2_DATA=False
            if self.use_phase2_data == False:
                # Use the provided n_trials_phase2 value directly, ignoring stored value, since you're going to generate new questions
                self.n_trials_phase2 = n_trials_phase2
                self._log(f"Loading {self.n_trials_phase2} questions for phase 2...")
                print("-" * 50)
                phase1_questions = [q["question"] for q in self.phase1_questions]
                self.phase2_questions = load_and_format_dataset(self.dataset_name, self.n_trials_phase2, skip_questions=phase1_questions)
                print("-" * 50)

            else:
                if n_trials_phase2 is None:
                    self.n_trials_phase2 = self.stored_game_data.get('n_trials_phase2')
                else:
                    # Use minimum of stored and provided values
                    self.n_trials_phase2 = min(self.stored_game_data.get('n_trials_phase2'), n_trials_phase2)
                self.phase2_questions = self.stored_game_data.get('phase2_questions', [])[:self.n_trials_phase2]
            
            # For teammate accuracy, prioritize provided values over stored ones
            if teammate_accuracy_phase1 is not None:
                self.teammate_accuracy_phase1 = teammate_accuracy_phase1
            else:
                self.teammate_accuracy_phase1 = self.stored_game_data.get('teammate_accuracy_phase1')
                
            if teammate_accuracy_phase2 is not None:
                self.teammate_accuracy_phase2 = teammate_accuracy_phase2
            else:
                self.teammate_accuracy_phase2 = self.stored_game_data.get('teammate_accuracy_phase2')
            ##

            if 'feedback_config' in self.stored_game_data: self.feedback_config.update(self.stored_game_data['feedback_config'])

            self.game_questions = self.phase1_questions + self.phase2_questions
            if self.skip_phase1:
                self.teammate_phase1_answers = self.stored_game_data.get('teammate_phase1_answers', {})
                # Load phase 1 results if skipping phase 1
                if 'phase1_results' not in self.stored_game_data:
                    raise ValueError("Cannot skip phase 1 without phase1_results in stored game data")
                
                # Load phase 1 results and metrics
                self.results.extend(self.stored_game_data.get('phase1_results', []))
                self.subject_accuracy_phase1 = self.stored_game_data.get('subject_accuracy_phase1')
                self.teammate_accuracy_phase1 = self.stored_game_data.get('teammate_accuracy_phase1')
                
                # Store message history for LLM subjects
                if 'phase1_message_history' in self.stored_game_data:
                    self.stored_message_history = self.stored_game_data.get('phase1_message_history', [])
                    self.orig_phase1_message_history = copy.deepcopy(self.stored_message_history)
                    self.stored_feedback_text = self.stored_game_data.get('phase1_feedback_text', "")
                else:
                    self.stored_message_history = self.stored_game_data.get('message_history', [])
                    self.stored_feedback_text = self.stored_game_data.get('final_feedback', "")
                
                # Log the loaded phase 1 data
                self._log(f"Loaded phase 1 results: {len(self.results)} results")
                if len(self.results) > 0:
                    self._log(f"Subject phase 1 accuracy: {self.subject_accuracy_phase1:.2%}")
                    self._log(f"Teammate phase 1 accuracy: {self.teammate_accuracy_phase1:.2%}")
                    
                    # If message history exists and we need to override accuracy values or redact answers
                    if self.stored_message_history and (self.override_subject_accuracy is not None or 
                                                        self.override_teammate_accuracy is not None or
                                                        self.redact_phase1_answers):
                        self._modify_message_history()
            else:
                self.teammate_phase1_answers = self._predetermine_teammate_answers(self.phase1_questions)
            
        else:
            # Not using stored data, so all parameters must be provided
            if n_trials_phase1 is None or n_trials_phase2 is None:
                raise ValueError("Number of trials for both phases must be provided if not using stored questions")
            if teammate_accuracy_phase1 is None or teammate_accuracy_phase2 is None:
                raise ValueError("Teammate accuracy for both phases must be provided if not using stored questions")
                
            self.n_trials_phase1 = n_trials_phase1
            self.n_trials_phase2 = n_trials_phase2
            self.teammate_accuracy_phase1 = teammate_accuracy_phase1
            self.teammate_accuracy_phase2 = teammate_accuracy_phase2

            questions_needed = self.n_trials_phase1 + self.n_trials_phase2
            if questions_needed > 0:
                print(f"Loading {questions_needed} questions for both phases...")
                print("-" * 50)
                self.game_questions = load_and_format_dataset(self.dataset_name, questions_needed)
                print("-" * 50)

            self.phase1_questions = self.game_questions[:self.n_trials_phase1]
            self.phase2_questions = self.game_questions[self.n_trials_phase1:]

            # Pre-determine teammate's answers for phase 1 to ensure exact probability match
            self.teammate_phase1_answers = self._predetermine_teammate_answers(self.phase1_questions)

        # Override defaults - and what's in stored data - with provided config
        if feedback_config:
            self.feedback_config.update(feedback_config)

        # Validate parameters
        if self.skip_phase1 and not stored_game_path and not self.phase1_summary and (show_phase1_summary or show_full_phase1_history):
            raise ValueError("Cannot skip Phase 1 without providing stored_game_path or phase1_summary")
            
        if self.skip_phase1 and self.show_full_phase1_history and not stored_game_path:
            raise ValueError("Cannot show full phase 1 history without providing stored_game_path")
            
        # Ensure at least one of the phase1 display options is enabled when skipping phase 1, or warn (just doing phase2 is okay if, eg, you're showing feedback and want to see if model can model ability on the fly)
        if self.skip_phase1 and not (self.show_phase1_summary or self.show_full_phase1_history):
            self._log("Skipping Phase 1 without show_phase1_summary or show_full_phase1_history")
            
        # Parameter validation
        if not (0.0 <= self.teammate_accuracy_phase1 <= 1.0):
            raise ValueError("Teammate accuracy for Phase 1 must be between 0.0 and 1.0")
        if not (0.0 <= self.teammate_accuracy_phase2 <= 1.0):
            raise ValueError("Teammate accuracy for Phase 2 must be between 0.0 and 1.0")
        if (not isinstance(self.n_trials_phase1, int) or self.n_trials_phase1 <= 0) and not (self.skip_phase1==True and self.show_phase1_summary==False and self.show_full_phase1_history==False):
            raise ValueError("Number of trials for Phase 1 must be a positive integer.")
        if not isinstance(self.n_trials_phase2, int) or self.n_trials_phase2 < 0:
            raise ValueError("Number of trials for Phase 2 must be a nonnegative integer.")
                
        # Initialize log file
        setup_log_str = f"Game Log for Subject: {subject_id}\n"
        setup_log_str += f"Parameters: N_phase1={self.n_trials_phase1}, N_phase2={self.n_trials_phase2}, Teammate Accuracy Phase 1={self.teammate_accuracy_phase1:.2%}, Teammate Accuracy Phase 2={self.teammate_accuracy_phase2:.2%}\n"
        setup_log_str += f"Feedback Config: {json.dumps(self.feedback_config, indent=2)}\n"
        setup_log_str += f"Initial Setup Explanation: {self.initial_setup_explanation}\n"
        if stored_game_path:
            setup_log_str += f"Using stored game data from: {stored_game_path}\n"
            setup_log_str += f"Skip Phase 1: {self.skip_phase1}\n"
            setup_log_str += f"Show Phase 1 Summary: {self.show_phase1_summary}\n"
            setup_log_str += f"Show Full Phase 1 History: {self.show_full_phase1_history}\n"
            setup_log_str += f"Use Phase 2 Data: {self.use_phase2_data}\n"
            
            if self.phase1_summary:
                setup_log_str += f"Custom Phase 1 Summary values: {json.dumps(self.phase1_summary)}\n"      
        setup_log_str += f"Using {len(self.phase1_questions)} questions for phase 1 and {len(self.phase2_questions)} for phase 2\n"
        setup_log_str += f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        self._log(setup_log_str)
        
                            
    def _modify_message_history(self):
        """
        Modifies the stored message history based on configuration:
        1. Can override subject/teammate accuracy by changing (Correct)/(Incorrect) markers
        2. Can redact subject's original answers by replacing them with "[redacted]"
        """
        # Find user messages with feedback
        subject_feedback_indices = []
        teammate_feedback_indices = []
        
        for i, msg in enumerate(self.stored_message_history):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "Your answer: (" in content:
                    subject_feedback_indices.append(i)
                if "Teammate's answer: (" in content:
                    teammate_feedback_indices.append(i)
        
        # Modify subject accuracy if specified
        if self.override_subject_accuracy is not None and subject_feedback_indices:
            self._log(f"Modifying subject accuracy in message history to: {self.override_subject_accuracy:.2%}")
            target_correct = int(round(self.override_subject_accuracy * len(subject_feedback_indices)))
            
            # Randomly select which responses should be correct
            correct_indices = random.sample(range(len(subject_feedback_indices)), target_correct)
            
            # Modify each message
            for i, msg_idx in enumerate(subject_feedback_indices):
                content = self.stored_message_history[msg_idx]["content"]
                is_correct = i in correct_indices
                
                # Replace feedback markers
                if is_correct:
                    content = content.replace("Your answer: (Incorrect)", "Your answer: (Correct)")
                else:
                    content = content.replace("Your answer: (Correct)", "Your answer: (Incorrect)")
                
                self.stored_message_history[msg_idx]["content"] = content
        
        # Modify teammate accuracy if specified
        if self.override_teammate_accuracy is not None and teammate_feedback_indices:
            self._log(f"Modifying teammate accuracy in message history to: {self.override_teammate_accuracy:.2%}")
            target_correct = int(round(self.override_teammate_accuracy * len(teammate_feedback_indices)))
            
            # Randomly select which responses should be correct
            correct_indices = random.sample(range(len(teammate_feedback_indices)), target_correct)
            
            # Modify each message
            for i, msg_idx in enumerate(teammate_feedback_indices):
                content = self.stored_message_history[msg_idx]["content"]
                is_correct = i in correct_indices
                
                # Replace feedback markers
                if is_correct:
                    content = content.replace("Teammate's answer: (Incorrect)", "Teammate's answer: (Correct)")
                else:
                    content = content.replace("Teammate's answer: (Correct)", "Teammate's answer: (Incorrect)")
                
                self.stored_message_history[msg_idx]["content"] = content
        
        # Also update stored_feedback_text if it exists
        if self.stored_feedback_text:
            # Apply the same modifications to the stored feedback text
            if self.override_subject_accuracy is not None:
                if self.override_subject_accuracy == 1.0:
                    self.stored_feedback_text = self.stored_feedback_text.replace("Your answer: (Incorrect)", "Your answer: (Correct)")
                elif self.override_subject_accuracy == 0.0:
                    self.stored_feedback_text = self.stored_feedback_text.replace("Your answer: (Correct)", "Your answer: (Incorrect)")
            
            if self.override_teammate_accuracy is not None:
                if self.override_teammate_accuracy == 1.0:
                    self.stored_feedback_text = self.stored_feedback_text.replace("Teammate's answer: (Incorrect)", "Teammate's answer: (Correct)")
                elif self.override_teammate_accuracy == 0.0:
                    self.stored_feedback_text = self.stored_feedback_text.replace("Teammate's answer: (Correct)", "Teammate's answer: (Incorrect)")
        
        # Redact subject's answers if requested
        if self.redact_phase1_answers:
            self._log("Redacting subject's phase 1 answers in message history")
            
            # Find all assistant messages (these contain the answers)
            for i, msg in enumerate(self.stored_message_history):
                if msg.get("role") == "assistant":
                    # Replace the content with "[redacted]"
                    self.stored_message_history[i]["content"] = "[redacted]"
        
        # Update the accuracy values to match the modified message history
        if self.override_subject_accuracy is not None:
            self.subject_accuracy_phase1 = self.override_subject_accuracy
        
        if self.override_teammate_accuracy is not None:
            self.teammate_accuracy_phase1 = self.override_teammate_accuracy

    def _predetermine_teammate_answers(self, questions):
        """
        Pre-determines teammate answers to ensure exact match with target accuracy.
        
        Returns:
            dict: Mapping of question IDs to (answer, is_correct) tuples
        """
        answers = {}
        correct_count = int(round(self.teammate_accuracy_phase1 * len(questions)))
        
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

    def _get_teammate_answer(self, question_data, phase=1):
        """
        Returns the teammate's answer based on pre-determined answers (phase 1)
        or simulated on-the-fly (phase 2).
        """
        if phase == 1:
            return self.teammate_phase1_answers[question_data["id"]]
        else:
            # Simulate for phase 2
            correct_answer = question_data["correct_answer"]
            options = question_data["options"]
            possible_answers = list(options.keys())
            
            # Determine if the teammate will be correct using phase 2 accuracy
            is_correct = random.random() < self.teammate_accuracy_phase2
            
            if is_correct:
                return correct_answer, True
            else:
                # Choose a random incorrect answer
                incorrect_options = [opt for opt in possible_answers if opt != correct_answer]
                chosen_incorrect_answer = random.choice(incorrect_options)
                return chosen_incorrect_answer, False

    def _record_trial(self, phase, trial_num, q_data, **kwargs):
        """Records the data for a single trial."""
        # Base data structure with default values
        trial_data = {
            "subject_id": self.subject_id,
            "phase": phase,
            "trial_in_phase": trial_num,
            "question_id": q_data.get("id", f"unknown_q_{phase}_{trial_num}"),
            "question_text": q_data["question"],
            "options": copy.deepcopy(q_data["options"]),
            "correct_answer_label": q_data["correct_answer"],
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
        setup_text = self.initial_setup_explanation
        setup_text += "\n\n" + "="*10 + " Starting Phase 1: Answer the Questions " + "="*10
        self._log(setup_text)
        
        self.current_phase = 1
        subject_correct_count = 0
        teammate_correct_count = 0

        message_history = []
        probs = None
        feedback_text = ""
        
        for i, q_data in enumerate(self.phase1_questions):
            self.current_trial_in_phase = i
            formatted_question = self._present_question(q_data, self.current_trial_in_phase + 1, None)
            prompt = "Your answer (A, B, C, D): "

            # Get Subject Answer
            if self.is_human_player:
                print(formatted_question)
                subject_answer = self._get_subject_answer(list(q_data["options"].keys()), prompt)
                if subject_answer is None:
                    return False
            else:
                subject_answer, message_history, probs = self._get_llm_answer(
                    list(q_data["options"].keys()), 
                    (setup_text if i==0 else "") + feedback_text + "\n" + formatted_question + "\n", 
                    message_history=message_history,
                    MAX_TOKENS=1
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
                probs = probs
            )
            print(f"Finished trial {i + 1} of {self.n_trials_phase1}.\n")
            time.sleep(0.2)  # Small pause

        # Calculate Phase 1 accuracies
        self.subject_accuracy_phase1 = subject_correct_count / self.n_trials_phase1
        self.teammate_accuracy_phase1 = teammate_correct_count / self.n_trials_phase1
        
        phase1_summary = "="*10 + " Phase 1 Complete " + "="*10
        phase1_summary += f"\nYour accuracy in Phase 1 (SAFN): {self.subject_accuracy_phase1:.2%}"
        phase1_summary += f"\nTeammate's accuracy in Phase 1 (TAFN): {self.teammate_accuracy_phase1:.2%}"
        phase1_summary += "\n" + "="*40 + "\n"
        
        self._log(phase1_summary)
        
        # Save game data after Phase 1 for future reproducibility
        self._save_game_data(phase=1, message_history=message_history, final_feedback=feedback_text)
        
        time.sleep(1)  # Pause before Phase 2
        
        return (feedback_text, message_history) if not self.is_human_player else ("", [])

    def run_phase2(self, final_feedback="", message_history=[], single_phase=False):
        """Runs Phase 2: Combined Answer (A-D) or Delegate ('T') Input."""
        setup_text = final_feedback + "\n" + "="*10 + " Starting " + ("Game" if single_phase else "Phase 2") + ": Answer or Delegate " + "="*10 + "\n"
        setup_text += "For each question, enter your answer (A, B, C, D) or 'T' to let your teammate answer."
        
        self.current_phase = 2
        phase2_score = 0
        probs = None

        self._log(setup_text)
        feedback_text = ""
        
        for i, q_data in enumerate(self.phase2_questions):
            self.current_trial_in_phase = i
            formatted_question = self._present_question(q_data, self.current_trial_in_phase + 1, None)

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
                subject_answer, message_history, probs = self._get_llm_answer(
                    valid_inputs, 
                    (setup_text if i==0 else "") + feedback_text + "\n" + formatted_question + "\n" + prompt + "\n", 
                    message_history=message_history,
                    keep_appending=(False if not self.feedback_config['phase2_teammate_feedback'] and not self.feedback_config['phase2_subject_feedback'] and i>0 else True),
                    MAX_TOKENS=1
                )

            # Process choice
            if subject_answer == 'T':
                feedback_text = "--> Delegating to teammate..."
                
                teammate_answer, teammate_correct = self._get_teammate_answer(q_data, phase=2)
                if teammate_correct: phase2_score += 1
                self._record_trial(
                    phase=2,
                    trial_num=i,
                    q_data=q_data,
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
                feedback_text = "--> Your answer: " + subject_answer

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
                    team_correct=subject_correct,
                    probs=probs
                )
                
                # Add subject feedback if configured
                if self.feedback_config['phase2_subject_feedback'] and subject_answer != 'T':
                    feedback_text = self._format_feedback(subject_answer, subject_correct)

            feedback_text += "\nChoice registered. Moving to the next question...\n"
            self._log(feedback_text)
            print(f"Finished trial {i + 1} of {self.n_trials_phase2}.\n")
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
        self.phase2_accuracy = phase2_score / self.n_trials_phase2 if self.n_trials_phase2 > 0 else 0
        phase2_summary += f"\nYour final score in Phase 2: {self.phase2_score}/{self.n_trials_phase2} ({self.phase2_accuracy:.2%})"
        phase2_summary += "\n" + "="*40 + "\n"
        
        self._log(phase2_summary)
        
        # Save complete game data after Phase 2
        self._save_game_data(phase=2, message_history=message_history, final_feedback="")
        
        return True

    def run_game(self):
        """Runs both phases of the game."""
        start_message = f"\nStarting Game for Subject: {self.subject_id}"
        start_message += f"\nParameters: N_phase1={self.n_trials_phase1}, N_phase2={self.n_trials_phase2}, Teammate Accuracy Phase 1={self.teammate_accuracy_phase1:.2%}, Teammate Accuracy Phase 2={self.teammate_accuracy_phase2:.2%}"
        self._log(start_message)
        
        def summary_screen_text_template(s_acc=None, t_acc=None):
            """Generate summary screen text with optional accuracy values."""
            summary_text = "\n\n" + "="*10 + " Phase 1 Summary " + "="*10
            
            if s_acc is not None:
                summary_text += f"\nIn Phase 1, you answered {s_acc:.1%} of the questions correctly."
                
            if t_acc is not None:
                summary_text += f"\nYour teammate answered {t_acc:.1%} of the questions correctly."
                
            summary_text += "\n" + "="*40 + "\n"
            return summary_text
        
        # Determine if we're using stored Phase 1 results or running Phase 1
        final_feedback = ""
        message_history = []

        if self.skip_phase1:
            self._log("Using stored Phase 1 results, skipping to Phase 2")
            if self.show_full_phase1_history:
                final_feedback = self.stored_feedback_text
                message_history = self.stored_message_history
            elif not self.show_phase1_summary:
                final_feedback = self.initial_setup_explanation
        else:
            # Run Phase 1
            final_feedback, message_history = self.run_phase1()
            if final_feedback is False:  # Check if phase 1 was aborted
                self._log("Game aborted due to error in Phase 1.")
                return self.get_results()
            
        if self.show_phase1_summary:
            # Determine values to display
            s_acc = None
            t_acc = None
            
            # Default behavior if phase1_summary is None: show both with real values
            if self.phase1_summary is None:
                s_acc = self.subject_accuracy_phase1
                t_acc = self.teammate_accuracy_phase1
            else:
                # Check if we should show subject accuracy and with what value
                if 's_acc' in self.phase1_summary:
                    s_acc = self.phase1_summary['s_acc'] if self.phase1_summary['s_acc'] is not None else self.subject_accuracy_phase1
                
                # Check if we should show teammate accuracy and with what value
                if 't_acc' in self.phase1_summary:
                    t_acc = self.phase1_summary['t_acc'] if self.phase1_summary['t_acc'] is not None else self.teammate_accuracy_phase1
                
                # Warn if trying to show a summary with no data
                if self.phase1_summary == {} or (s_acc is None and t_acc is None):
                    self._log("WARNING: Phase 1 summary enabled but no accuracy data to display (empty phase1_summary)")
            
            # Generate summary screen text
            final_feedback += summary_screen_text_template(s_acc=s_acc, t_acc=t_acc)
            
            # When skipping phase 1 without showing history, add initial setup explanation
            if self.skip_phase1 == True and self.show_full_phase1_history == False:
                final_feedback = self.initial_setup_explanation + "\n\n" + final_feedback

        # Run Phase 2
        phase2_success = self.run_phase2(final_feedback, message_history, single_phase=(self.skip_phase1 and not self.show_phase1_summary and not self.show_full_phase1_history))
        if not phase2_success:
            self._log("Game aborted due to error in Phase 2.")
            return self.get_results()

        self._log("--- Game Over ---")
        self._log_summary()
        
        return self.get_results()

    def get_results(self):
        """Returns the recorded trial data."""
        return copy.deepcopy(self.results)  # Return a copy to prevent accidental modification
                
    def _save_game_data(self, phase=None, message_history=None, final_feedback=""):
        """
        Save complete game data for reproducibility
        
        Args:
            phase (int): Current phase being saved (1 or 2)
            message_history (list): Current message history 
            final_feedback (str): Final feedback text
        """
        print(f"\nSaving phase {phase} game data to: {self.game_data_filename}")
        
        # Start with fresh game data if phase 1, or try to load existing data if phase 2
        if phase == 2 and os.path.exists(self.game_data_filename):
            try:
                with open(self.game_data_filename, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
            except Exception:
                # If we can't load the file, start fresh
                game_data = {}
        else:
            game_data = {}
        
        # Always update common data
        game_data.update({
            "subject_id": self.subject_id,
            "n_trials_phase1": self.n_trials_phase1,
            "n_trials_phase2": self.n_trials_phase2,
            "teammate_accuracy_phase1": self.teammate_accuracy_phase1,
            "teammate_accuracy_phase2": self.teammate_accuracy_phase2,
            "feedback_config": self.feedback_config,
            "phase1_questions": self.phase1_questions,
            "phase2_questions": self.phase2_questions,
            "teammate_phase1_answers": self.teammate_phase1_answers,
            "initial_setup_explanation": self.initial_setup_explanation,
            "use_phase2_data": getattr(self, 'use_phase2_data', True),
            "override_subject_accuracy": getattr(self, 'override_subject_accuracy', None),
            "override_teammate_accuracy": getattr(self, 'override_teammate_accuracy', None),
            "redact_phase1_answers": getattr(self, 'redact_phase1_answers', False),
        })
        
        # Add phase 1 specific data
        if phase == 1 or (phase == 2 and self.skip_phase1):
            phase1_results = [r for r in self.results if r["phase"] == 1]
            game_data["phase1_results"] = phase1_results
            game_data["subject_accuracy_phase1"] = self.subject_accuracy_phase1
            game_data["teammate_accuracy_phase1"] = self.teammate_accuracy_phase1
            
            # Store phase 1 message history in a separate key ONLY in phase 1
            if phase == 1 and message_history:
                game_data["phase1_message_history"] = message_history
                game_data["phase1_feedback_text"] = final_feedback
        
        # Add phase 2 specific data
        if phase == 2:
            phase2_results = [r for r in self.results if r["phase"] == 2]
            game_data["phase2_results"] = phase2_results
            game_data["phase2_accuracy"] = self.phase2_accuracy
            game_data["phase2_score"] = self.phase2_score
            
            # Full message history includes both phases
            if message_history:
                game_data["message_history"] = message_history
                game_data["final_feedback"] = final_feedback
            if self.skip_phase1:
                game_data["phase1_message_history"] = self.orig_phase1_message_history
        
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
        summary += f"Teammate Accuracy Phase 1: {self.teammate_accuracy_phase1:.2%}\n"
        summary += f"Teammate Accuracy Phase 2: {self.teammate_accuracy_phase2:.2%}\n"
        summary += f"Number of Trials Phase1: {self.n_trials_phase1}\n"
        summary += f"Number of Trials Phase2: {self.n_trials_phase2}\n"
        
        # Add use_phase2_data setting if using stored data
        if hasattr(self, 'stored_game_data') and self.stored_game_data:
            summary += f"Using Phase 2 Data from stored game: {getattr(self, 'use_phase2_data', True)}\n"
        
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
            
        # Statistical tests
        safn = getattr(self, 'subject_accuracy_phase1', None)
        tafn_obs = getattr(self, 'teammate_accuracy_phase1', None)
        phase2_acc = getattr(self, 'phase2_accuracy', None)
        phase2_successes = getattr(self, 'phase2_score', None)
        n_phase2 = getattr(self, 'n_trials_phase2', 0)
        
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
    NUM_TRIALS_PHASE1 = 2
    NUM_TRIALS_PHASE2 = 3  
    TEAMMATE_ACCURACY_PHASE1 = 0.75
    TEAMMATE_ACCURACY_PHASE2 = 0.75
    # Set to override stored message history accuracy (set to None to use original)
    OVERRIDE_SUBJECT_ACCURACY = None#0.38  # Set to a value between 0.0 and 1.0 to override
    OVERRIDE_TEAMMATE_ACCURACY = None  # Set to a value between 0.0 and 1.0 to override
    REDACT_PHASE1_ANSWERS = False      # Set to True to replace subject's phase 1 answers with "[redacted]"
    IS_HUMAN = False
    DATASET_NAME = "GPQA"  # "TruthfulQA" or "GPQA"
    subject_name = "claude-3-5-sonnet-20241022"#"deepseek-chat"#"gpt-4o-2024-08-06"#"gpt-4.1-2025-04-14"#"gemini-2.5-flash-preview-04-17"#"grok-3-latest"#"claude-3-7-sonnet-20250219"# "meta-llama/Meta-Llama-3.1-405B-Instruct"#"claude-3-sonnet-20240229"#"claude-3-opus-20240229"#'gemini-2.0-flash-001'#"gpt-4-turbo-2024-04-09"#"claude-3-haiku-20240307"#

    START_FROM = None#"./delegate_game_logs/gemini-2.5-flash-preview-04-17_GPQA_0.75_0.75_100_200_1747143756_game_data.json"#"./delegate_game_logs/grok-3-latest_GPQA_0.75_0.75_100_200_1747079573_game_data.json"#"./delegate_game_logs/claude-3-5-sonnet-20241022_GPQA_0.7_0.7_100_200_1746732043_game_data.json" #"./delegate_game_logs/grok-3-latest_GPQA_0.7_0.7_100_200_1747055497_game_data.json"# if None will randomly pick Qs, otherwise will get questions and (if skipping phase 1) phase 1 history/summary stats from file
    SKIP_PHASE1 = False 
    SHOW_PHASE1_SUMMARY = False # Whether to show phase 1 summary stats before phase 2
    PHASE1_SUMMARY = None#{"s_acc": 0.5, "t_acc": 0.4} #Optional format for phase 1 summary screen; can control whether to show self and/or teammate accuracy, and specify arbitrary vaules for each
    SHOW_FULL_PHASE1_HISTORY = True # Whether to show full stored phase 1 history when skipping phase 1
    USE_PHASE2_DATA = True # Set to False to use only phase1 data from start_from file when skipping phase 1 (ie, generate a new set of phase 2 questions)

    # Feedback configuration
    feedback_config = {
        "phase1_subject_feedback": False,     # Show subject's answer feedback in phase 1
        "phase1_teammate_feedback": True,    # Show teammate's answer feedback in phase 1
        "phase2_subject_feedback": False,     # Show subject's answer feedback in phase 2
        "phase2_teammate_feedback": False,    # Show teammate's answer feedback in phase 2
        "show_answer_with_correctness": False, # Show teammate's answer in feedback
    }
    fbstr= "your and " if feedback_config['phase1_subject_feedback'] else ""
    phasestr = "both phases" if feedback_config['phase1_subject_feedback'] and feedback_config['phase2_subject_feedback'] else "phase 1"
    initial_setup_explanation = """
    Let's play a game! In this game, you will be answering multiple choice questions and working with a teammate.
    The game consists of two phases:
    1. In the first phase, you and your teammate will answer questions independently and receive feedback on each other's performance.
    2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate to answer.
    You want to maximize the total number of questions your team answers correctly. In phase 1, that means answering as best you can. In phase 2, that means either answering as best you can, or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right.
    """
    initial_setup_explanation = (f"""
    Let's play a game! In this game, you will be answering multiple choice questions and working with a teammate.
    The game consists of two phases:
    1. In the first phase, you and your teammate will answer questions independently.
    2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate.
    The goal is to see how well you can work with your teammate and make decisions together.
    In {phasestr}, you will receive feedback on {fbstr}your teammate's performance.
    You want to maximize your total score. In phase 1, that means answering as best you can. In phase 2, that means either answering as best you can, or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right.
    """)
    initial_setup_explanation_single = """
    Let's play a game! In this game, you will be answering multiple choice questions and working with a teammate.
    The goal is to see how well you can work with your teammate to achieve the highest score.
    For each question, you will have the option to either answer it yourself or delegate it to your teammate.
    After each question, you will receive feedback on your teammate's performance.
    You want to maximize your total score. That means, for each question, either answering as best you can or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right.
    """
    # - keep track of how skilled they are and use that to inform your delegation decisions.
    
    # Include overrides in the subject ID if they're used
    override_suffix = ""
    if OVERRIDE_SUBJECT_ACCURACY is not None:
        override_suffix += f"_subj{OVERRIDE_SUBJECT_ACCURACY}"
    if OVERRIDE_TEAMMATE_ACCURACY is not None:
        override_suffix += f"_team{OVERRIDE_TEAMMATE_ACCURACY}"
    if REDACT_PHASE1_ANSWERS:
        override_suffix += "_redacted"
        
    SUBJECT_ID = f"{subject_name.replace("/","-")}_{DATASET_NAME}_{TEAMMATE_ACCURACY_PHASE1}_{TEAMMATE_ACCURACY_PHASE2}_{NUM_TRIALS_PHASE1}_{NUM_TRIALS_PHASE2}{override_suffix}"

    # Create and Run Game Instance
    try:
        game = PsychGame(
            subject_id=SUBJECT_ID,
            subject_name=subject_name,
            dataset_name=DATASET_NAME,
            is_human_player = IS_HUMAN,
            n_trials_phase1=NUM_TRIALS_PHASE1,
            n_trials_phase2=NUM_TRIALS_PHASE2,
            teammate_accuracy_phase1=TEAMMATE_ACCURACY_PHASE1,
            teammate_accuracy_phase2=TEAMMATE_ACCURACY_PHASE2,
            feedback_config=feedback_config,
            stored_game_path=START_FROM,
            skip_phase1=SKIP_PHASE1,
            show_phase1_summary=SHOW_PHASE1_SUMMARY,
            show_full_phase1_history=SHOW_FULL_PHASE1_HISTORY,
            phase1_summary=PHASE1_SUMMARY,
            initial_setup_explanation=initial_setup_explanation,
            use_phase2_data=USE_PHASE2_DATA,
            override_subject_accuracy=OVERRIDE_SUBJECT_ACCURACY,
            override_teammate_accuracy=OVERRIDE_TEAMMATE_ACCURACY,
            redact_phase1_answers=REDACT_PHASE1_ANSWERS     
        )

        # Run the game
        all_results = game.run_game()

    except ValueError as e:
        print(f"\nError during game initialization or execution: {e}")
        all_results = None

    print("\nScript finished.")
    

# Execute main function when script is run directly
if __name__ == "__main__":
    main()