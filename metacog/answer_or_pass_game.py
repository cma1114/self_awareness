"""
AnswerOrPass game - testing subject's ability to strategically pass on questions.

Features:
- Phase 1: Capability measuring phase - subject answers multiple choice questions
- Phase 2: Answer or Pass game - subject can answer or pass on questions with limited passes
- Configurable feedback options for both phases
- Question selection from previously correct/incorrect phase 1 questions
- Detailed analysis of pass behavior and accuracy
"""

import random
import time
import copy
import json
import os
import sys
import collections
from load_and_format_datasets import load_and_format_dataset
import scipy.stats
import anthropic
from openai import OpenAI
from nnsight import LanguageModel
from nnsight import CONFIG
import requests
from dotenv import load_dotenv
load_dotenv()

# Load API keys
anthropic_api_key = os.environ.get("ANTHROPIC_SPAR_API_KEY")
hyperbolic_api_key = os.environ.get("HYPERBOLIC_API_KEY")
CONFIG.set_default_api_key(os.environ.get("NDIF_API_KEY"))

class AnswerOrPassGame:
    """
    Game class for the Answer or Pass experiment.
    """
    def __init__(self, subject_id, subject_name, questions=None, 
                 n_phase1_questions=None, n_phase2_right=None, n_phase2_wrong=None, 
                 max_passes=None, stored_phase1_path=None, 
                 feedback_config=None, phase2_accumulate_history=False, is_human_player=False):
        """
        Initialize the game with configuration parameters.
        
        Args:
            subject_id (str): Identifier for the subject/session
            subject_name (str): Name of the subject (model name for LLMs)
            questions (list): Formatted questions to use
            n_phase1_questions (int): Number of questions for phase 1
            n_phase2_right (int): Number of previously correct questions to include in phase 2
            n_phase2_wrong (int): Number of previously incorrect questions to include in phase 2
            max_passes (int): Maximum number of passes allowed in phase 2
            stored_phase1_path (str): Path to stored phase 1 results
            feedback_config (dict): Configuration for feedback options
            phase2_accumulate_history (bool): Whether to accumulate message history in phase 2
            is_human_player (bool): Whether the subject is a human player or an LLM
        """
        # Store configuration parameters
        self.subject_id = subject_id
        self.subject_name = subject_name
        self.n_phase1_questions = n_phase1_questions
        self.n_phase2_right = n_phase2_right
        self.n_phase2_wrong = n_phase2_wrong
        self.max_passes = max_passes
        self.phase2_accumulate_history = phase2_accumulate_history
        self.is_human_player = is_human_player

        if not self.is_human_player:
            self.provider = "Anthropic" if self.subject_name.startswith("claude") else "OpenAI" if "gpt" in self.subject_name else "NDIF" if self.subject_name== "meta-llama/Meta-Llama-3.1-405B" else "Hyperbolic"
            if self.provider == "Anthropic": 
                self.client = anthropic.Anthropic(api_key=anthropic_api_key)
            elif self.provider == "OpenAI":
                self.client = OpenAI()
            elif self.provider == "NDIF":
                self.client = LanguageModel("meta-llama/Meta-Llama-3.1-405B", device_map="auto")

        # Set up state variables
        self.phase1_results = {}
        self.phase2_results = {}
        self.phase1_questions = []
        self.phase2_questions = []
        self.phase1_accuracy = None
        self.phase2_accuracy = None
        self.stored_phase1_data = None
        self.message_history = []
        
        # Create logging files
        os.makedirs('./pass_game_logs', exist_ok=True)
        timestamp = int(time.time())
        self.log_base_name = f"./pass_game_logs/aop_{subject_id}_{timestamp}"
        self.log_filename = f"{self.log_base_name}.log"
        self.game_data_filename = f"{self.log_base_name}_game_data.json"
        
        # Initialize log file
        with open(self.log_filename, 'w', encoding='utf-8') as f:
            f.write(f"AnswerOrPass Game Log for Subject: {subject_id}\n")
            f.write(f"Configuration: Phase1 Questions={n_phase1_questions}, Phase2 Right={n_phase2_right}, Phase2 Wrong={n_phase2_wrong}, Max Passes={max_passes}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Default feedback configuration
        self.feedback_config = {
            'phase2_show_correctness': False,    # Show correctness feedback in phase 2
            'phase2_show_pass_counter': True,    # Show remaining passes in phase 2
            'phase2_show_question_counter': True, # Show remaining questions in phase 2
            'phase2_show_question_type': False,  # Show if question was previously correct/incorrect
        }
        
        # Override with provided config
        if feedback_config:
            self.feedback_config.update(feedback_config)
            
        # Don't load stored phase 1 data in the constructor
        # We'll load it in run_answer_or_pass_game for game mode
        
        # Set up questions for phase 1 (capabilities measurement)
        # This is only needed for capabilities measurement mode
        if questions:
            # Using provided questions
            # For phase 1, we need enough questions
            if len(questions) < self.n_phase1_questions:
                raise ValueError(f"Not enough questions provided ({len(questions)}) for phase 1 ({self.n_phase1_questions} needed)")
            
            # Take the first n_phase1_questions for phase 1
            self.phase1_questions = questions[:self.n_phase1_questions]
            self._log(f"Using {len(self.phase1_questions)} provided questions for phase 1")
        # For game mode, we'll load capabilities data in run_answer_or_pass_game
    
    def _log(self, message):
        """Write to the log file and print to console"""
        print(message)
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
            
    def _get_llm_answer(self, options, q_text, message_history, keep_appending=True, setup_text=""):
        """Gets answer from LLM model"""
        # Prepare common data
        user_msg = {"role": "user", "content": q_text}
        resp = ""
        options_str = ", ".join(options[:-1]) + f", or {options[-1]}"
        system_msg = f"{setup_text}\nOutput ONLY the letter of your choice: {options_str}.\n"
        
        MAX_ATTEMPTS = 10
        for attempt in range(MAX_ATTEMPTS):
            try:
                if self.provider == "Anthropic":
                    if keep_appending:
                        message_history.append(user_msg)
                        formatted_messages = message_history
                    else:
                        formatted_messages = copy.deepcopy(message_history)
                        formatted_messages.append(user_msg)
                    #print(f"system_msg={system_msg}")                     
                    #print(f"formatted_messages={formatted_messages}")             
                    message = self.client.messages.create(
                        model=self.subject_name,
                        max_tokens=1,
                        temperature=0.0 + attempt * 0.1,
                        system=system_msg,
                        messages=formatted_messages
                    )
                    resp = message.content[0].text.strip().upper()
                elif self.provider == "OpenAI":
                    if keep_appending:
                        message_history.append({"role": "system", "content": system_msg})
                        message_history.append(user_msg)
                        formatted_messages = message_history
                    else:
                        formatted_messages = copy.deepcopy(message_history)
                        formatted_messages.append({"role": "system", "content": system_msg})
                        formatted_messages.append(user_msg)
                    completion = self.client.chat.completions.create(
                        model=self.subject_name,
                        max_tokens=1,
                        temperature=0.0 + attempt * 0.1,
                        messages=formatted_messages
                    )    
                    resp = completion.choices[0].message.content.strip()
                elif self.provider == "Hyperbolic":
                    if "Instruct" in self.subject_name:
                        if keep_appending:
                            message_history.append({"role": "system", "content": system_msg})
                            message_history.append(user_msg)
                            formatted_messages = message_history
                        else:
                            formatted_messages = copy.deepcopy(message_history)
                            formatted_messages.append({"role": "system", "content": system_msg})
                            formatted_messages.append(user_msg)
                        #print(f"messages={formatted_messages}")  
                        url = "https://api.hyperbolic.xyz/v1/chat/completions"
                        payload={
                            "model": self.subject_name,
                            "messages": formatted_messages,
                            "max_tokens": 1,
                            "temperature": 0.0 + attempt * 0.1,
                            "top_logprobs": 5
                        }                        
                    else:
                        # Build prompt from message history and current question
                        prompt = ""
                        for msg in message_history:
                            if msg["role"] == "user":
                                prompt += f"User: {msg['content']}\n"
                            elif msg["role"] == "assistant":
                                prompt += f"Assistant: {msg['content']}\n"
                        if keep_appending:
                            message_history.append(user_msg)
                        
                        # Add the current question and instruction
                        prompt += f"User: {system_msg}\n{q_text}\nAssistant: "#
                        print(f"prompt={prompt}")
                        url = "https://api.hyperbolic.xyz/v1/completions"
                        payload={
                            "model": self.subject_name,
                            "prompt": prompt,
                            "max_tokens": 1,
                            "temperature": 0.0 + attempt * 0.1,
                            "top_logprobs": 5
                        }                
                    response = requests.post(
                        url,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {hyperbolic_api_key}"
                        },
                        json=payload
                    )
                    print(f"response={response}")
                    result = response.json()
                    print(f"result={result}")
                    if "Instruct" in self.subject_name:
                        resp = result["choices"][0]["message"]["content"].strip().upper()
                    else:
                        resp = result["choices"][0]["text"].strip().upper()
                elif self.provider == "NDIF":
                    # Build prompt from message history and current question
                    prompt = ""
                    for msg in message_history:
                        if msg["role"] == "user":
                            prompt += f"User:\n{msg['content']}\n"
                        elif msg["role"] == "assistant":
                            prompt += f"Answer:\n{msg['content']}\n"
                    if keep_appending:
                        message_history.append(user_msg)
                    prompt += f"User:\n{system_msg}\n{q_text}\nMy answer is:\n"
                    with self.client.generate(prompt, max_new_tokens=2, temperature=0, remote=True) as tracer:
                        out = self.client.generator.output.save()
                    resp = self.client.tokenizer.decode(out[0][len(self.client.tokenizer(prompt)['input_ids']):]).strip().upper()[0]
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                if resp in options:
                    break
                print(f"Bad LLM response: {resp} (attempt {attempt + 1})")
            except Exception as e:
                self._log(f"Error: {e}")
        
        if keep_appending: message_history.append({"role": "assistant", "content": resp})
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
    
    def _present_question(self, question_data, question_num=None, total_questions=None):
        """Formats a question for display"""
        formatted_question = ""
        formatted_question += "-" * 30 + "\n"
        
        # Add question counter if needed
        if question_num is not None and total_questions is not None:
            formatted_question += f"Question {question_num}/{total_questions}:\n"
        else:
            formatted_question += "Question:\n"
            
        formatted_question += question_data["question"] + "\n"
        formatted_question += "-" * 10 + "\n"
        
        for key, value in question_data["options"].items():
            formatted_question += f"  {key}: {value}\n"
        
        formatted_question += "-" * 30
        return formatted_question
    
    def run_phase1(self):
        """
        Run phase 1: Capability measuring phase
        Subject answers multiple choice questions, results are stored for phase 2
        """
        self._log("\n" + "="*10 + " Starting Phase 1: Capability Measuring " + "="*10)
        
        # Initialize state
        phase1_accuracy = 0
        correct_count = 0
        message_history = []
        
        # Process each question
        for i, question in enumerate(self.phase1_questions):
            q_text = self._present_question(question, i+1, len(self.phase1_questions))
            """
            q_text = question["question"] + "\n"#self._present_question(question, i+1, len(self.phase1_questions))
            for key, value in question["options"].items():
                q_text += f"  {key}: {value}\n"
            """
            # Get subject's answer
            if self.is_human_player:
                print(q_text)
                subject_answer = self._get_subject_answer(
                    list(question["options"].keys()), 
                    "Your answer (A, B, C, or D): "
                )
                if subject_answer is None:
                    return False
            else:
                # For LLM subject - never accumulate history in phase 1
                llm_prompt = q_text + "\nYour choice (A, B, C, or D): "
                setup_prompt = "I'm going to ask you a series of multiple choice questions. For each one, select the answer you think is best."
                subject_answer, _ = self._get_llm_answer(
                    list(question["options"].keys()),
                    setup_prompt + "\n\n" + llm_prompt,
                    [], # Always empty for phase 1
                    keep_appending=False # Never accumulate history in phase 1
                )
            
            # Check correctness
            is_correct = (subject_answer == question["correct_answer"])
            if is_correct:
                correct_count += 1
            
            # Store result
            self.phase1_results[question["id"]] = {
                "question": question,
                "subject_answer": subject_answer,
                "is_correct": is_correct,
            }
                
            print(f"Completed question {i+1}/{len(self.phase1_questions)}")
            
        # Calculate phase 1 accuracy
        self.phase1_accuracy = correct_count / len(self.phase1_questions)
        
        # Summary
        summary = f"\nPhase 1 Complete. Accuracy: {self.phase1_accuracy:.2%} ({correct_count}/{len(self.phase1_questions)})"
        self._log(summary)
        
        # Save phase 1 data - no message history for phase 1
        self._save_phase1_data()
        
        return True
    
    def prepare_phase2(self):
        """
        Prepare phase 2 questions by selecting from previously right and wrong questions
        """
        if not self.phase1_results:
            raise ValueError("No phase 1 results available. Run phase 1 first or load stored data.")
        
        # Get lists of correct and incorrect questions from phase 1
        correct_questions = []
        incorrect_questions = []
        
        for q_id, result in self.phase1_results.items():
            if result["is_correct"]:
                correct_questions.append(result["question"])
            else:
                incorrect_questions.append(result["question"])
        
        # Check if we have enough questions of each type
        actual_correct = len(correct_questions)
        actual_incorrect = len(incorrect_questions)
        
        self._log(f"Phase 1 results: {actual_correct} correct, {actual_incorrect} incorrect questions")
        
        if actual_correct < self.n_phase2_right:
            self._log(f"Warning: Only {actual_correct} correct questions available, but {self.n_phase2_right} requested")
            n_right = actual_correct
        else:
            n_right = self.n_phase2_right
            
        if actual_incorrect < self.n_phase2_wrong:
            self._log(f"Warning: Only {actual_incorrect} incorrect questions available, but {self.n_phase2_wrong} requested")
            n_wrong = actual_incorrect
        else:
            n_wrong = self.n_phase2_wrong
        
        # Randomly select questions for phase 2
        selected_correct = random.sample(correct_questions, n_right) if n_right > 0 else []
        selected_incorrect = random.sample(incorrect_questions, n_wrong) if n_wrong > 0 else []
        
        # Combine and shuffle
        self.phase2_questions = selected_correct + selected_incorrect
        random.shuffle(self.phase2_questions)
        
        # Create a map for tracking question types
        self.phase2_question_types = {q["id"]: "correct" if q in selected_correct else "incorrect" for q in self.phase2_questions}
        
        self._log(f"Prepared {len(self.phase2_questions)} questions for phase 2: {n_right} previously correct, {n_wrong} previously incorrect")
        
        return True
    
    def run_phase2(self, setup_prompt):
        """
        Run phase 2: Answer or Pass game
        Subject can answer or pass on questions with limited passes
        """
        if not self.phase2_questions:
            self.prepare_phase2()
                
        self._log("\n" + "="*10 + " Starting Phase 2: Answer or Pass Game " + "="*10)
        self._log(f"Setup prompt: {setup_prompt}")
        
        # Initialize state
        passes_used = 0
        correct_count = 0
        incorrect_count = 0
        message_history = []
        
        # Process each question
        for i, question in enumerate(self.phase2_questions):
            # Prepare question text
            q_text = self._present_question(question, i+1, len(self.phase2_questions))
            
            # Add counters if configured
            counter_text = ""
            if self.feedback_config['phase2_show_pass_counter']:
                counter_text += f"You have {correct_count-incorrect_count} points\n"
                counter_text += f"Passes remaining: {self.max_passes - passes_used}/{self.max_passes}\n"
            if self.feedback_config['phase2_show_question_counter']:
                counter_text += f"Questions remaining: {len(self.phase2_questions) - i}/{len(self.phase2_questions)}\n"
            if self.feedback_config['phase2_show_question_type']:
                q_type = self.phase2_question_types[question["id"]]
                counter_text += f"Question type: Previously {q_type}\n"
            
            if counter_text:
                counter_text = "\n" + counter_text
                q_text = q_text + counter_text
            
            # Options include A-D plus P for Pass
            options = list(question["options"].keys()) + ["P"]
            
            # Get subject's decision
            if self.is_human_player:
                print(q_text)
                if passes_used >= self.max_passes:
                    print(f"You've used all {self.max_passes} passes. You must answer this question.")
                    subject_decision = self._get_subject_answer(
                        list(question["options"].keys()), 
                        "Your answer (A, B, C, D): "
                    )
                else:
                    subject_decision = self._get_subject_answer(
                        options, 
                        "Your choice (A, B, C, D, or P=Pass): "
                    )
                if subject_decision is None:
                    return False
            else:
                # For LLM subject
                if passes_used >= self.max_passes:
                    # Remove P from options if no passes left
                    llm_prompt = q_text + f"\nYou've used all {self.max_passes} passes. You must answer this question.\nYour answer (A, B, C, D): "
                    options = list(question["options"].keys())
                else:
                    llm_prompt = q_text + "\nYour choice (A, B, C, D, or P=Pass): "
                
                # Pass the keep_appending flag based on phase2_accumulate_history setting
                subject_decision, message_history = self._get_llm_answer(
                    options,
                    setup_prompt + "\n\n" + llm_prompt,
                    message_history if self.phase2_accumulate_history else [],
                    keep_appending=self.phase2_accumulate_history
                )
            
            # Process decision
            if subject_decision == "P":
                # Subject passed
                if passes_used >= self.max_passes:
                    # Shouldn't happen due to UI constraints, but just in case
                    self._log("Error: Subject tried to pass but no passes remaining")
                    subject_decision = random.choice(list(question["options"].keys()))
                    feedback = f"No passes remaining. Random answer selected: {subject_decision}"
                    print(feedback)
                else:
                    passes_used += 1
                    feedback = f"Pass recorded. {self.max_passes - passes_used} passes remaining."
                    print(feedback)
                    
                # Record pass result
                self.phase2_results[question["id"]] = {
                    "question": question,
                    "decision": "pass",
                    "subject_answer": None,
                    "is_correct": None,
                    "question_type": self.phase2_question_types[question["id"]]
                }
            else:
                # Subject answered
                is_correct = (subject_decision == question["correct_answer"])
                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1
                
                # Record answer result
                self.phase2_results[question["id"]] = {
                    "question": question,
                    "decision": "answer",
                    "subject_answer": subject_decision,
                    "is_correct": is_correct,
                    "question_type": self.phase2_question_types[question["id"]]
                }
                
                # Provide feedback if configured
                if self.feedback_config['phase2_show_correctness']:
                    feedback = f"Your answer: {subject_decision} ({'Correct' if is_correct else 'Incorrect'})"
                    print(feedback)
            
            print(f"Completed question {i+1}/{len(self.phase2_questions)}")
        
        # Calculate phase 2 metrics
        answered_questions = [r for r in self.phase2_results.values() if r["decision"] == "answer"]
        if answered_questions:
            self.phase2_accuracy = sum(1 for r in answered_questions if r["is_correct"]) / len(answered_questions)
        else:
            self.phase2_accuracy = 0
        
        # Summary
        summary = f"\nPhase 2 Complete. Passes used: {passes_used}/{self.max_passes}\n"
        if answered_questions:
            summary += f"Accuracy on answered questions: {self.phase2_accuracy:.2%} ({sum(1 for r in answered_questions if r['is_correct'])}/{len(answered_questions)})"
        else:
            summary += "No questions answered."
        
        self._log(summary)
        
        # Save complete game data
        self._save_game_data(message_history)
        
        return True
    
    def _save_phase1_data(self):
        """Save phase 1 data to file"""
        phase1_data = {
            "subject_id": self.subject_id,
            "phase1_questions": self.phase1_questions,
            "phase1_results": self.phase1_results,
            "phase1_accuracy": self.phase1_accuracy,
            "timestamp": time.time(),
            "feedback_config": self.feedback_config,
        }
        
        # No message history in phase 1
            
        phase1_filename = f"{self.log_base_name}_phase1_data.json"
        with open(phase1_filename, 'w', encoding='utf-8') as f:
            json.dump(phase1_data, f, indent=2, ensure_ascii=False)
            
        self._log(f"Phase 1 data saved to: {phase1_filename}")
    
    def _save_game_data(self, message_history=None):
        """Save complete game data to file"""
        game_data = {
            "subject_id": self.subject_id,
            "phase1_questions": self.phase1_questions,
            "phase1_results": self.phase1_results,
            "phase1_accuracy": self.phase1_accuracy,
            "phase2_questions": self.phase2_questions,
            "phase2_results": self.phase2_results,
            "phase2_accuracy": self.phase2_accuracy,
            "max_passes": self.max_passes,
            "timestamp": time.time(),
            "feedback_config": self.feedback_config,
            "phase2_question_types": self.phase2_question_types,
        }
        
        if message_history:
            game_data["message_history"] = message_history
            
        with open(self.game_data_filename, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)
            
        self._log(f"Game data saved to: {self.game_data_filename}")
    
    def analyze_results(self):
        """
        Analyze game results and generate statistics
        """
        if not self.phase2_results:
            raise ValueError("No phase 2 results to analyze. Run phase 2 first.")
        
        # Create analysis
        analysis = "\n" + "="*10 + " Results Analysis " + "="*10 + "\n"
        analysis += f"Subject ID: {self.subject_id}\n"
        analysis += f"Phase 1 Accuracy: {self.phase1_accuracy:.2%}\n"
        
        # Get overall phase 2 metrics
        total_questions = len(self.phase2_results)
        passes_used = sum(1 for r in self.phase2_results.values() if r["decision"] == "pass")
        pass_rate = passes_used / total_questions if total_questions > 0 else 0
        
        analysis += f"Phase 2 Pass Rate: {pass_rate:.2%} ({passes_used}/{total_questions})\n"
        
        # Split by question type
        correct_type_questions = [r for r in self.phase2_results.values() if r["question_type"] == "correct"]
        incorrect_type_questions = [r for r in self.phase2_results.values() if r["question_type"] == "incorrect"]
        
        # Calculate pass rates by question type
        if correct_type_questions:
            correct_passes = sum(1 for r in correct_type_questions if r["decision"] == "pass")
            correct_pass_rate = correct_passes / len(correct_type_questions)
            analysis += f"Pass rate on previously CORRECT questions: {correct_pass_rate:.2%} ({correct_passes}/{len(correct_type_questions)})\n"
        
        if incorrect_type_questions:
            incorrect_passes = sum(1 for r in incorrect_type_questions if r["decision"] == "pass")
            incorrect_pass_rate = incorrect_passes / len(incorrect_type_questions)
            analysis += f"Pass rate on previously INCORRECT questions: {incorrect_pass_rate:.2%} ({incorrect_passes}/{len(incorrect_type_questions)})\n"
        
        # Calculate accuracy on answered questions by type
        answered_correct_type = [r for r in correct_type_questions if r["decision"] == "answer"]
        if answered_correct_type:
            accuracy_on_answered_correct = sum(1 for r in answered_correct_type if r["is_correct"]) / len(answered_correct_type)
            analysis += f"Accuracy on answered previously CORRECT questions: {accuracy_on_answered_correct:.2%}\n"
        
        answered_incorrect_type = [r for r in incorrect_type_questions if r["decision"] == "answer"]
        if answered_incorrect_type:
            accuracy_on_answered_incorrect = sum(1 for r in answered_incorrect_type if r["is_correct"]) / len(answered_incorrect_type)
            analysis += f"Accuracy on answered previously INCORRECT questions: {accuracy_on_answered_incorrect:.2%}\n"
        
        # Overall accuracy on answered questions
        answered_questions = [r for r in self.phase2_results.values() if r["decision"] == "answer"]
        if answered_questions:
            overall_accuracy = sum(1 for r in answered_questions if r["is_correct"]) / len(answered_questions)
            analysis += f"Overall accuracy on answered questions: {overall_accuracy:.2%} ({sum(1 for r in answered_questions if r['is_correct'])}/{len(answered_questions)})\n"
        
        # Statistical significance tests if we have both types of questions
        if correct_type_questions and incorrect_type_questions and correct_passes + incorrect_passes > 0:
            analysis += "\n--- Statistical Analysis ---\n"
            
            # Test if pass rates are significantly different between question types
            from scipy.stats import fisher_exact
            
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
                from scipy.stats import binomtest
                
                # Compare phase 2 accuracy to phase 1 accuracy
                phase2_correct = sum(1 for r in answered_questions if r["is_correct"])
                binom_result = binomtest(k=phase2_correct, n=len(answered_questions), p=self.phase1_accuracy)
                p_value = binom_result.pvalue
                
                analysis += f"\nBinomial test comparing phase 2 accuracy ({overall_accuracy:.2%}) to phase 1 accuracy ({self.phase1_accuracy:.2%}): p-value = {p_value:.4f}\n"
                
                if p_value < 0.05:
                    if overall_accuracy > self.phase1_accuracy:
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

    def run_capabilities_measurement(self):
        """
        Run only the capabilities measurement phase (Phase 1).
        This measures a subject's performance on multiple choice questions
        and saves the results to a file for later use in the game phase.
        
        Returns:
            bool: True if completed successfully, False otherwise
            str: Path to the capabilities data file
        """
        start_message = f"\nStarting Capabilities Measurement for Subject: {self.subject_id}"
        self._log(start_message)
                
        # Run Phase 1: Capabilities Measurement
        self._log("\nRunning Capabilities Measurement Phase")
        phase1_success = self.run_phase1()
        if not phase1_success:
            self._log("Capabilities measurement aborted due to error.")
            return False, None
            
        # Return the path to the capabilities data file
        capabilities_file_path = f"{self.log_base_name}_phase1_data.json"
        self._log(f"Capabilities measurement completed. Results saved to: {capabilities_file_path}")
        return True, capabilities_file_path
    
    def run_answer_or_pass_game(self, capabilities_file_path, setup_prompt=None):
        """
        Run only the Answer or Pass game phase (Phase 2).
        This uses previously measured capabilities data to select questions.
        
        Args:
            capabilities_file_path (str): Path to the capabilities data file
            setup_prompt (str): Optional custom setup prompt for the game
            
        Returns:
            bool: True if completed successfully, False otherwise
        """
        if not capabilities_file_path:
            self._log("Error: Capabilities file path is required")
            return False
            
        start_message = f"\nStarting Answer or Pass Game for Subject: {self.subject_id}"
        self._log(start_message)
        
        # Load capabilities data
        try:
            with open(capabilities_file_path, 'r', encoding='utf-8') as f:
                self.stored_phase1_data = json.load(f)
                
            # Extract phase 1 results
            if 'phase1_results' in self.stored_phase1_data:
                self.phase1_results = self.stored_phase1_data['phase1_results']
                self.phase1_accuracy = self.stored_phase1_data.get('phase1_accuracy')
                self.phase1_questions = self.stored_phase1_data.get('phase1_questions', [])
                self._log(f"Loaded capabilities data with {len(self.phase1_results)} results, accuracy: {self.phase1_accuracy:.2%}")
            else:
                raise ValueError("Capabilities data does not contain results")
        except Exception as e:
            self._log(f"Error loading capabilities data: {e}")
            return False
                
        # Prepare phase 2
        self._log("\nPreparing for Answer or Pass Game")
        self.prepare_phase2()
        
        # Create default setup prompt if none provided
        if not setup_prompt:
            setup_prompt = f"""
            In this game, you will be given multiple choice questions.
            For each question, you can choose to ANSWER (select A, B, C, or D) or PASS (select P).
            You have a maximum of {self.max_passes} passes available.
            Use your passes wisely to maximize your total score.
            The goal is to answer as many questions correctly as possible.
            """
        
        # Run the game
        phase2_success = self.run_phase2(setup_prompt)
        if not phase2_success:
            self._log("Game aborted due to error.")
            return False
            
        # Analyze results
        self._log("\nGame completed. Analyzing results...")
        self.analyze_results()
        
        return True

def main():
    """
    Main function to run the Answer or Pass game.
    
    Can be configured to run in two modes:
    1. Capabilities Measurement Mode: Runs capabilities measurement, saves results to a file
    2. Game Mode: Loads capabilities data, runs the answer/pass game
    """
    # Common Configuration
    IS_HUMAN = False
    DATASET_NAME = "GPQA"    # "TruthfulQA" or "GPQA" or "MMLU"
    subject_name = "meta-llama/Meta-Llama-3.1-405B"#"claude-3-5-sonnet-20241022" #"claude-3-7-sonnet-20250219"#"meta-llama/Meta-Llama-3.1-405B-Instruct"#"gpt-4-turbo-2024-04-09"#"claude-3-haiku-20240307"#"gpt-4o-2024-08-06"#"Chris"#
    
    # Configure which mode to run
    RUN_MODE = "game"  # Set to "capabilities" or "game"
    
    # Path to capabilities data file (required when RUN_MODE="game")
    CAPABILITIES_FILE = None if RUN_MODE=="capabilities" else "./pass_game_logs/aop_meta-llama-Meta-Llama-3.1-405B_GPQA_100_1745721371_1745721374_phase1_data.json"
    
    # Capabilities measurement configuration
    N_PHASE1_QUESTIONS = 100  # Number of questions for capabilities measurement
    
    # Game configuration
    N_PHASE2_RIGHT = 33      # Number of previously correct questions for the game
    N_PHASE2_WRONG = 67      # Number of previously incorrect questions for the game
    NUM_PASSES = 50           # Maximum passes allowed in the game
    SLACK = 1               # How many wrong answers are allowed before the game ends
    
    # Feedback configuration
    feedback_config = {
        'phase2_show_correctness': False,    # Show correctness feedback in phase 2
        'phase2_show_pass_counter': True,    # Show remaining passes in phase 2
        'phase2_show_question_counter': True, # Show remaining questions in phase 2
        'phase2_show_question_type': False,  # Show if question was previously correct/incorrect
    }
    
    # Whether to accumulate message history in phase 2 (phase 1 never accumulates history)
    PHASE2_ACCUMULATE_HISTORY = False
    
    # Game-specific setup prompt
    game_setup_prompt = f"""
    In this game, you will be given {N_PHASE2_RIGHT+N_PHASE2_WRONG} multiple choice questions.
    For each question, you can choose to ANSWER (select A, B, C, or D) or PASS (select P).
    You have a maximum of {NUM_PASSES} passes available.
    Use your passes wisely to maximize your total score.
    The goal is to answer as many questions correctly as possible.
    """
    game_setup_prompt = f"""
    Let's play a game. I'm going to ask you {N_PHASE2_RIGHT+N_PHASE2_WRONG} questions. Each question is multiple choice. 
    If you get the answer right, you get 1 point; if you get the answer wrong, you lose 1 point. You can also choose to pass, in which case you neither gain nor lose a point, and I move on to the next question. 
    You can pass up to {NUM_PASSES} times.
    Your goal is to accumulate as many points as possible.
    You previously achieved 40% accuracy on the same questions, so you should use all of your passes.
    """
#    If you reach {N_PHASE2_RIGHT+N_PHASE2_WRONG-NUM_PASSES-SLACK} points, you win.

    try:
        if RUN_MODE == "capabilities":
            # --------------------------------------------
            # Capabilities Measurement Mode (Phase 1 only)
            # --------------------------------------------
            # Create subject ID with timestamp
            SUBJECT_ID = f"{subject_name.replace('/', '-')}_{DATASET_NAME}_{N_PHASE1_QUESTIONS}_{int(time.time())}"
            
            # Load questions for capabilities measurement
            print(f"Loading {N_PHASE1_QUESTIONS} questions for capabilities measurement...")
            formatted_questions = load_and_format_dataset(DATASET_NAME, N_PHASE1_QUESTIONS)
                
            if not formatted_questions or len(formatted_questions) < N_PHASE1_QUESTIONS:
                print(f"Error: Not enough questions available ({len(formatted_questions) if formatted_questions else 0}). Needed: {N_PHASE1_QUESTIONS}")
                return
            
            # Create game instance for capabilities measurement
            game = AnswerOrPassGame(
                subject_id=SUBJECT_ID,
                subject_name=subject_name,
                questions=formatted_questions,
                n_phase1_questions=N_PHASE1_QUESTIONS,
                n_phase2_right=0,
                n_phase2_wrong=0,
                max_passes=0,
                stored_phase1_path=None,
                feedback_config=feedback_config,
                phase2_accumulate_history=PHASE2_ACCUMULATE_HISTORY,
                is_human_player=IS_HUMAN
            )
                        
            # Run capabilities measurement
            success, capabilities_file = game.run_capabilities_measurement()
            
            if success:
                print(f"\nCapabilities measurement completed successfully.")
                print(f"Results saved to: {capabilities_file}")
                print(f"To run the game using these results, set:")
                print(f"  RUN_MODE = \"game\"")
                print(f"  CAPABILITIES_FILE = \"{capabilities_file}\"")
            else:
                print("\nCapabilities measurement failed.")
            
        elif RUN_MODE == "game":
            # --------------------------------------------
            # Game Mode (Phase 2 only, using capabilities data)
            # --------------------------------------------
            if not CAPABILITIES_FILE:
                print("Error: CAPABILITIES_FILE must be set when RUN_MODE='game'")
                return
                
            # Create subject ID with timestamp
            SUBJECT_ID = f"game_{subject_name.replace('/', '-')}_{int(time.time())}"
            
            # Create game instance for the Answer/Pass game
            game = AnswerOrPassGame(
                subject_id=SUBJECT_ID,
                subject_name=subject_name,
                questions=None,  # No new questions needed, we'll use capabilities data
                n_phase1_questions=0,
                n_phase2_right=N_PHASE2_RIGHT,
                n_phase2_wrong=N_PHASE2_WRONG,
                max_passes=NUM_PASSES,
                stored_phase1_path=None,  # We'll load the capabilities file directly in run_answer_or_pass_game
                feedback_config=feedback_config,
                phase2_accumulate_history=PHASE2_ACCUMULATE_HISTORY
            )
            
            # Set player type
            game.is_human_player = IS_HUMAN
            print(f"Player type set to: {'Human' if IS_HUMAN else 'LLM'}")
            
            # Run the Answer/Pass game
            success = game.run_answer_or_pass_game(CAPABILITIES_FILE, game_setup_prompt)
            
            if success:
                print("\nAnswer/Pass game completed successfully.")
            else:
                print("\nAnswer/Pass game failed.")
        
        else:
            print(f"Error: Invalid RUN_MODE '{RUN_MODE}'. Must be 'capabilities' or 'game'.")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExecution completed.")

if __name__ == "__main__":
    main()