"""
PassGameFromCapabilities - A version of the pass game that uses completed results files

Features:
- Takes output from complete_model_results.py (completed_results_XX directory)
- Selects balanced question set
- Runs only Phase 2 (delegate game) with multiple choice or short answer questions
- Centralizes prompts and run parameters; records all important run-level parameters
- Prints/logs only summary stats
"""

import random
import time
import copy
import json
import os
import re
from base_game_class import BaseGameClass
from load_and_format_datasets import load_and_format_dataset
import string

ANSWER_TYPES = None  # e.g., ["Date", "Person"]

class AnswerOrPassGame(BaseGameClass):
    def __init__(
        self,
        subject_id,
        subject_name,
        is_human_player,
        completed_results_file=None,
        dataset="GPQA",
        all_questions=False,
        n_right=None,
        n_wrong=None,
        max_passes=None,
        feedback_config=None,
        accumulate_history=False,
        initial_setup_explanation="",
        seed=None,
        temperature=0.0,
        resume_from=None,
        include_question_num=False,
        include_total_questions=False
    ):
        super().__init__(subject_id, subject_name, is_human_player, "pass_game_logs")

        # Seed RNG
        self.seed = seed
        if self.seed is not None:
            self._log(f"Using random seed: {self.seed}")
            random.seed(self.seed)

        # Store configuration parameters
        self.temperature = temperature
        self.completed_results_file = completed_results_file
        self.dataset = dataset
        self.n_right = n_right
        self.n_wrong = n_wrong
        self.max_passes = max_passes
        self.accumulate_history = accumulate_history
        self.is_human_player = is_human_player
        self.include_question_num = include_question_num
        self.include_total_questions = include_total_questions

        self.questions = []
        self.results = []
        self.score = 0
        self.accuracy = None

        # Centralized prompt/wording strings (used everywhere; recorded later)
        self.prompts = {
            # Game setup template 
            "game_setup_prompt_template": initial_setup_explanation,

            # Human prompts (MC)
            "human_mc_choice_with_pass": "Your choice (A, B, C, D, or P=Pass): ",
            "human_mc_answer_no_pass": "Your answer (A, B, C, D): ",

            # Human prompts (SA)
            "human_sa_choice_with_pass": "Your choice (answer succinctly or P=Pass): ",
            "human_sa_answer_no_pass": "Your answer: ",

            # LLM rules and lines (match original punctuation/newlines)
            "llm_mc_choice_rule": "Respond only with the letter of your choice; do NOT output any other text.",
            "llm_sa_choice_rule": "Respond ONLY with your answer or 'P'.\n",
            "llm_sa_answer_rule": "Respond ONLY with your answer.\n",
            "llm_force_answer_line": "You've used all {MAX_PASSES} passes. You must answer this question.",

            # LLM prompt suffixes (exact original text)
            "llm_mc_choice_with_pass_suffix": "Your choice (A, B, C, D, or P=Pass): ",
            "llm_mc_answer_no_pass_suffix": "Your answer (A, B, C, D): ",
            "llm_sa_choice_with_pass_suffix": "Your choice (answer succinctly or P=Pass): ",
            "llm_sa_answer_no_pass_suffix": "Your answer: ",

            # On-screen counters
            "counter_points_line": "You have {POINTS} points",
            "counter_passes_line": "Passes remaining: {REMAIN}/{TOTAL}",
            "counter_questions_line": "Questions remaining: {REMAIN}/{TOTAL}",
            "counter_type_line": "Question type: Previously {QTYPE}",

            # Displayed feedback lines
            "feedback_pass_recorded": "Pass recorded. {REMAIN} passes remaining.",
            "feedback_different_answer": "Different answer to question {QID} from phase 1: {CUR} != {ORIG}",
        }

        # Default feedback configuration; override with provided config
        self.feedback_config = {
            'show_correctness': False,
            'show_pass_counter': True,
            'show_point_counter': True,
            'show_question_counter': True,
            'show_question_type': False,
        }
        if feedback_config:
            self.feedback_config.update(feedback_config)

        # Load completed results data and select questions
        self._load_completed_results(all_questions)

        # Finalize the game setup prompt using selected counts
        try:
            phase1_accuracy_pct = round(self.n_right / (len(self.questions)) * 100)
        except Exception:
            phase1_accuracy_pct = 0
        self.initial_setup_explanation = self.prompts["game_setup_prompt_template"].format(
            N_QUESTIONS=len(self.questions),
            ACCURACY=phase1_accuracy_pct,
            NUM_PASSES=self.max_passes
        )

        # Resume behavior: load completed_results (phase 1) if provided via resume
        if resume_from:
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

        # Compute static get_llm_answer args (non-per-question) and store run parameters
        # MAX_TOKENS rule: None for 'opus-4'/'sonnet-4' or short-answer; else 1
        max_tokens_used = None if ('opus-4' in self.subject_name or 'sonnet-4' in self.subject_name or getattr(self, "is_short_answer", False)) else 1
        self.get_llm_answer_static_args = {
            "keep_appending": self.accumulate_history,
            "message_history": [],
            "MAX_TOKENS": max_tokens_used,
            "temp": self.temperature,
            "accept_any": True
        }

        # Record run-level parameters (only non-per-question items)
        self.run_parameters = {
            "dataset": self.dataset,
            "completed_results_file": self.completed_results_file,
            "all_questions": all_questions,
            "n_right": self.n_right,
            "n_wrong": self.n_wrong,
            "max_passes": self.max_passes,
            "feedback_config": self.feedback_config,
            "accumulate_history": self.accumulate_history,
            "is_human_player": self.is_human_player,
            "temperature": self.temperature,
            "seed": self.seed,
            "is_short_answer": self.is_short_answer,
            "present_question_args": {
                "include_question_num": self.include_question_num,
                "include_total_questions": self.include_total_questions
            },
            "get_llm_answer_static_args": self.get_llm_answer_static_args,
            "prompts_used": {
                "game_setup_prompt_template": self.prompts["game_setup_prompt_template"],
                "game_setup_prompt_resolved": self.initial_setup_explanation,
                "human_mc_choice_with_pass": self.prompts["human_mc_choice_with_pass"],
                "human_mc_answer_no_pass": self.prompts["human_mc_answer_no_pass"],
                "human_sa_choice_with_pass": self.prompts["human_sa_choice_with_pass"],
                "human_sa_answer_no_pass": self.prompts["human_sa_answer_no_pass"],
                "llm_mc_choice_rule": self.prompts["llm_mc_choice_rule"],
                "llm_sa_choice_rule": self.prompts["llm_sa_choice_rule"],
                "llm_sa_answer_rule": self.prompts["llm_sa_answer_rule"],
                "llm_force_answer_line": self.prompts["llm_force_answer_line"],
                "llm_mc_choice_with_pass_suffix": self.prompts["llm_mc_choice_with_pass_suffix"],
                "llm_mc_answer_no_pass_suffix": self.prompts["llm_mc_answer_no_pass_suffix"],
                "llm_sa_choice_with_pass_suffix": self.prompts["llm_sa_choice_with_pass_suffix"],
                "llm_sa_answer_no_pass_suffix": self.prompts["llm_sa_answer_no_pass_suffix"],
                "counter_points_line": self.prompts["counter_points_line"],
                "counter_passes_line": self.prompts["counter_passes_line"],
                "counter_questions_line": self.prompts["counter_questions_line"],
                "counter_type_line": self.prompts["counter_type_line"],
                "feedback_pass_recorded": self.prompts["feedback_pass_recorded"],
                "feedback_different_answer": self.prompts["feedback_different_answer"],
            }
        }

    def _load_completed_results(self, all_questions):
        """Load completed results data from the specified file and select questions."""
        if not self.completed_results_file or not os.path.exists(self.completed_results_file):
            raise ValueError(f"Completed results file not found: {self.completed_results_file}")

        try:
            self._log(f"Loading completed results from: {self.completed_results_file}")
            with open(self.completed_results_file, 'r', encoding='utf-8') as f:
                self.completed_data = json.load(f)

            if "results" not in self.completed_data or not isinstance(self.completed_data["results"], dict):
                raise ValueError("Invalid completed results file: missing or invalid 'results' field")

            self._determine_question_type()
            self._separate_questions_by_correctness(all_questions)

            if self.max_passes is None:
                self.max_passes = len(self.all_incorrect_questions)

            self._log(f"Loaded completed results with {len(self.completed_data['results'])} questions")
            self._log(f"Selected {len(self.questions)} questions")
            self._log(f"Question type: {'Short Answer' if self.is_short_answer else 'Multiple Choice'}")

        except Exception as e:
            raise ValueError(f"Error loading completed results data: {e}")

    def _determine_question_type(self):
        """Determine if the dataset is multiple choice or short answer."""
        result = next(iter(self.completed_data["results"].values()))
        first_result = result['question'] if isinstance(result['question'], dict) else result
        self.is_short_answer = not ("options" in first_result and isinstance(first_result["options"], dict) and len(first_result["options"]) > 0)

    def _separate_questions_by_correctness(self, all_questions):
        """Separate questions into correct and incorrect lists."""
        self.all_correct_questions = []
        self.all_incorrect_questions = []
        
        if not self.completed_data or "results" not in self.completed_data:
            self._log("Error: Completed data is missing or has no 'results' field in _separate_questions_by_correctness.")
            return

        if self.dataset == "GPQA":
            gpqa_questions_with_features = load_and_format_dataset("GPQA")
            feature_lookup = {
                item['id']: {
                    'difficulty': item.get('difficulty_score'),
                    'overlap_ratio': item.get('overlap_ratio', 0),
                    'domain': item.get('high_level_domain'),
                    'question_text': item.get('question')
                }
                for item in gpqa_questions_with_features if item.get('id')
            }
        else:
            feature_lookup = {}

        for q_id, result_item in self.completed_data["results"].items():
            # Filter GPQA by subject suffix if needed
            if self.dataset == "GPQA":
                domain = feature_lookup.get(q_id, {}).get("domain")
                if "_nobio" in self.subject_id and domain and str(domain).lower() == "biology":
                    continue
                difficulty = feature_lookup.get(q_id, {}).get("difficulty", 0)
                if "_noeasy" in self.subject_id and difficulty and difficulty < 2:
                    continue

            question_data_for_list = {"id": q_id}
            current_is_correct = result_item.get("is_correct")

            # Skip questions where correctness cannot be determined
            if current_is_correct is not True and current_is_correct is not False:
                self._log(f"Question {q_id} has 'is_correct' as '{current_is_correct}'. Skipping for correct/incorrect separation.")
                continue

            question_data_for_list["is_correct"] = current_is_correct
            question_data_for_list["subject_answer"] = result_item.get("subject_answer", "N/A")
            question_data_for_list["probs"] = result_item.get("probs")

            resq = result_item['question'] if isinstance(result_item['question'], dict) else result_item
            question_data_for_list["question"] = resq.get("question", "N/A")
            question_data_for_list["options"] = resq.get("options", {})
            question_data_for_list["correct_answer"] = resq.get("correct_answer_label", "N/A") if "correct_answer_label" in resq else resq.get("correct_answer", "N/A")

            # Add to appropriate list
            if question_data_for_list["is_correct"]:
                self.all_correct_questions.append(question_data_for_list)
            else:
                self.all_incorrect_questions.append(question_data_for_list)
        
        self._log(f"Separated questions: {len(self.all_correct_questions)} correct, {len(self.all_incorrect_questions)} incorrect")
        
        # Shuffle both lists to ensure random selection
        if self.all_correct_questions:
            random.shuffle(self.all_correct_questions)
        if self.all_incorrect_questions:
            random.shuffle(self.all_incorrect_questions)

        if ANSWER_TYPES and (self.dataset == "SimpleQA" or self.dataset == "SimpleMC"):
            sqa_all_questions = load_and_format_dataset(self.dataset)
            sqa_feature_lookup = {
                item['id']: {
                    'answer_type': item.get('answer_type'),
                    'topic': item.get('topic'),
                    'q_text': item.get('question')
                } for item in sqa_all_questions
            }
            self.all_correct_questions = [q for q in self.all_correct_questions if sqa_feature_lookup.get(q["id"], {}).get("answer_type") in ANSWER_TYPES]
            self.all_incorrect_questions = [q for q in self.all_incorrect_questions if sqa_feature_lookup.get(q["id"], {}).get("answer_type") in ANSWER_TYPES]

        if self.n_right is not None and self.n_wrong is not None:
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

    def _present_question_with_indices(self, question, i, total):
        """Helper to call present_question with the configured indices."""
        if self.include_question_num and self.include_total_questions:
            return self._present_question(question, i, total)
        elif self.include_question_num:
            return self._present_question(question, i)
        else:
            return self._present_question(question)

    def _check_short_answer(self, subject_answer, correct_answer):
        """Simple string-matching check for short answer correctness."""
        subject_normalized = self._normalize_text(subject_answer)
        correct_normalized = self._normalize_text(correct_answer)
        if subject_normalized == correct_normalized:
            return True
        if len(subject_normalized) > 4 and len(correct_normalized) > 4:
            if subject_normalized in correct_normalized or correct_normalized in subject_normalized:
                return True
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
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _parse_subject_decision(self, resp, options):
        """Normalize subject response to a choice token if possible."""
        if len(resp.rstrip(string.whitespace + string.punctuation)) == 0:
            return resp
        arr = resp.upper().rstrip(string.whitespace + string.punctuation)
        if arr and arr[0] in options:
            return arr[0]
        if arr and arr[-1] in options:
            return arr[-1]
        return resp

    def _save_game_data(self, message_history=None):
        """Save complete game data to file"""
        game_data = {
            "subject_id": self.subject_id,
            "questions": self.questions,
            "results": self.results,
            "accuracy": self.accuracy,
            "score": self.score,
            "subject_accuracy_phase1": self.n_right / (self.n_right + self.n_wrong) if (self.n_right is not None and self.n_wrong is not None and (self.n_right + self.n_wrong) > 0) else None,
            "max_passes": self.max_passes,
            "feedback_config": self.feedback_config,
            "initial_setup_explanation": self.initial_setup_explanation,
            "capabilities_file": self.completed_results_file,
            "run_parameters": self.run_parameters
        }
        if message_history:
            game_data["message_history"] = message_history
            
        with open(self.game_data_filename, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)
        self._log(f"Game data saved to: {self.game_data_filename}")

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
            # Present question; optionally include indices if configured (defaults preserve original behavior)
            q_text = self._present_question_with_indices(question, i+1, len(self.questions)) if (self.include_question_num or self.include_total_questions) else self._present_question(question)

            # Build counter_text (exact original semantics)
            counter_text = ""
            if self.feedback_config['show_point_counter']:
                counter_text += self.prompts["counter_points_line"].format(POINTS=correct_count - incorrect_count) + "\n"
            if self.feedback_config['show_pass_counter']:
                counter_text += self.prompts["counter_passes_line"].format(REMAIN=self.max_passes - passes_used, TOTAL=self.max_passes) + "\n"
            if self.feedback_config['show_question_counter']:
                counter_text += self.prompts["counter_questions_line"].format(REMAIN=len(self.questions) - i, TOTAL=len(self.questions)) + "\n"
            if self.feedback_config['show_question_type']:
                q_type = "Correct" if question["is_correct"] else "Incorrect"
                counter_text += self.prompts["counter_type_line"].format(QTYPE=q_type) + "\n"
            
            if counter_text:
                q_text = q_text + "\n" + counter_text
            
            # Options include A-D plus P for Pass
            if self.is_short_answer:
                options = ["P"]
            else:
                options = list(question["options"].keys()) + ["P"]
            if passes_used >= self.max_passes and "P" in options:
                options.remove("P")
            
            # Get subject's decision
            if self.is_human_player:
                print(q_text)
                if self.is_short_answer:
                    if passes_used >= self.max_passes:
                        print(self.prompts["llm_force_answer_line"].format(MAX_PASSES=self.max_passes))
                        resp = self._get_subject_answer([], self.prompts["human_sa_answer_no_pass"])
                    else:
                        resp = self._get_subject_answer([], self.prompts["human_sa_choice_with_pass"])
                else:
                    if passes_used >= self.max_passes:
                        print(self.prompts["llm_force_answer_line"].format(MAX_PASSES=self.max_passes))
                        resp = self._get_subject_answer(list(question["options"].keys()), self.prompts["human_mc_answer_no_pass"])
                    else:
                        resp = self._get_subject_answer(options, self.prompts["human_mc_choice_with_pass"])
                if resp is None:
                    return False
            else:
                # For LLM subject (use only centralized prompts)
                if self.is_short_answer:
                    if passes_used >= self.max_passes:
                        llm_prompt = q_text + f"\n{self.prompts['llm_force_answer_line'].format(MAX_PASSES=self.max_passes)}\n{self.prompts['llm_sa_answer_no_pass_suffix']}"
                        setup_text = self.prompts["llm_sa_answer_rule"]  # includes trailing newline
                    else:
                        llm_prompt = q_text + "\n" + self.prompts["llm_sa_choice_with_pass_suffix"]
                        setup_text = self.prompts["llm_sa_choice_rule"]  # includes trailing newline
                else:
                    setup_text = None
                    if passes_used >= self.max_passes:
                        llm_prompt = q_text + f"\n{self.prompts['llm_force_answer_line'].format(MAX_PASSES=self.max_passes)} {self.prompts['llm_mc_choice_rule']}\n{self.prompts['llm_mc_answer_no_pass_suffix']}"
                    else:
                        llm_prompt = q_text + f"\n{self.prompts['llm_mc_choice_rule']}\n{self.prompts['llm_mc_choice_with_pass_suffix']}"
                
                gla = self.get_llm_answer_static_args
                resp, message_history, probs = self._get_llm_answer(
                    options if not self.is_short_answer else None,
                    self.initial_setup_explanation + "\n\n" + llm_prompt,
                    message_history if gla["keep_appending"] else [],
                    keep_appending=gla["keep_appending"],
                    setup_text=setup_text,
                    MAX_TOKENS=gla["MAX_TOKENS"],
                    temp=gla["temp"],
                    accept_any=gla["accept_any"]
                )
            
            # Parse decision
            subject_decision = self._parse_subject_decision(resp, options)

            # Process decision
            if subject_decision == "P":
                passes_used += 1
                print(self.prompts["feedback_pass_recorded"].format(REMAIN=self.max_passes - passes_used))
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
                    print(self.prompts["feedback_different_answer"].format(
                        QID=question["id"], CUR=subject_decision, ORIG=question["subject_answer"]
                    ))
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
            if (i+1) % log_interval == 0:
                self._save_game_data(message_history)
        
        # Summary stats
        answered = correct_count + incorrect_count
        self.accuracy = (correct_count / answered) if answered > 0 else None
        pass_rate = (passes_used / len(self.questions)) if self.questions else 0.0
        
        summary = "\n" + "="*10 + " Game Summary " + "="*10 + "\n"
        summary += f"Subject ID: {self.subject_id}\n"
        summary += f"Passes used: {passes_used}/{self.max_passes}\n"
        summary += f"Answered: {answered}/{len(self.questions)}\n"
        summary += f"Accuracy on answered questions: {self.accuracy:.2%}" if self.accuracy is not None else "Accuracy on answered questions: N/A"
        summary += f"\nPass rate: {pass_rate:.2%} ({passes_used}/{len(self.questions)})\n"
        summary += f"Different-from-phase1 answers: {dif_answer_cnt}\n"
        self._log(summary)
        
        self._save_game_data(message_history)
        return True

    def get_results(self):
        """Returns the recorded trial data."""
        return copy.deepcopy(self.results)


def real_main(SUBJECT_NAME, DATASET):
    TEMPERATURE = 0.0
    SEED = 999
    
    ALL_QUESTIONS = True
    N_RIGHT = None    
    N_WRONG = None  
    NUM_PASSES = 500
    
    # Feedback configuration
    feedback_config = {
        'show_correctness': False,    # Show correctness feedback in phase 2
        'show_pass_counter': False,   # Show remaining passes in phase 2
        'show_point_counter': False,  # Show score in phase 2
        'show_question_counter': False, # Show remaining questions in phase 2
        'show_question_type': False,  # Show if question was previously correct/incorrect
    }
    
    ACCUMULATE_HISTORY = False
    resume_from = None
    IS_HUMAN = False

    # Centralized setup prompt template (keep exact whitespace/newlines)
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

    # Determine capabilities file path
    if DATASET == "SimpleQA":
        CAPABILITES_TEST_FILE = f"./compiled_results_sqa/{SUBJECT_NAME.replace('/','-')}_phase1_compiled.json"
    elif DATASET == "GPSA":
        CAPABILITES_TEST_FILE = f"./compiled_results_gpsa/{SUBJECT_NAME.replace('/','-')}_phase1_compiled.json"
    elif DATASET == "SimpleMC":
        CAPABILITES_TEST_FILE = f"./compiled_results_smc/{SUBJECT_NAME.replace('/','-')}_phase1_compiled.json"
    else:
        CAPABILITES_TEST_FILE = f"./completed_results_{DATASET.lower()}/{SUBJECT_NAME.replace('/','-')}_phase1_completed.json"

    # Optional: control passing indices into present_question (defaults keep original behavior)
    INCLUDE_QNUM = False
    INCLUDE_TOTAL = False
        
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
            resume_from=resume_from,
            include_question_num=INCLUDE_QNUM,
            include_total_questions=INCLUDE_TOTAL
        )
        
        # Run the game
        success = game.run_pass_game()
        if success:
            print(f"\nGame completed. Results saved to: {game.game_data_filename}")
        else:
            print("\nGame failed.")
        
    except Exception as e:
        print(f"Error during game execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExecution completed.")

def main():
    """Main function to run the delegate game from completed results"""
    DATASETS = ["SimpleMC"]  # One of: GPQA, SimpleQA, SimpleMC, MMLU, TruthfulQA, GPSA
    models = ["llama-3.1-405b"]
    for model in models:
        for DATASET in DATASETS:
            real_main(model, DATASET)

if __name__ == "__main__":
    main()