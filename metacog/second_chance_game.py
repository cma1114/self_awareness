"""
Second-Chance Game - testing a model's ability to change its answer when given feedback.

Features:
- Uses capabilities data to select questions the model got wrong
- Presents each with history showing model already answered incorrectly
- Analyzes how often the model changes its answer
- Can optionally show or redact the model's original answer
"""

import random
import time
import json
from base_game_class import BaseGameClass
from scipy.stats import binomtest

random.seed(42)

class SecondChanceGame(BaseGameClass):
    """
    Game class for the Second-Chance experiment.
    """
    def __init__(self, subject_id, subject_name, capabilities_file_path, 
                 num_questions=20, show_original_answer=True, is_human_player=False):
        """
        Initialize the game with configuration parameters.
        
        Args:
            subject_id (str): Identifier for the subject/session
            subject_name (str): Name of the subject (model name for LLMs)
            capabilities_file_path (str): Path to stored capabilities/phase1 results
            num_questions (int): Number of questions to present
            show_original_answer (bool): Whether to show the original answer or redact it
            is_human_player (bool): Whether the subject is a human player or an LLM
        """
        super().__init__(subject_id, subject_name, is_human_player, "secondchance_game_logs")
        # Store configuration parameters
        self.capabilities_file_path = capabilities_file_path
        self.num_questions = num_questions
        self.show_original_answer = show_original_answer
        self.is_human_player = is_human_player
                
        # Initialize state variables
        self.capabilities_data = None
        self.wrong_questions = []
        self.selected_questions = []
        self.game_results = {}
        self.message_history = []
                
        # Initialize log file
        setup_log_str = f"Second-Chance Game Log for Subject: {subject_id}\n"
        setup_log_str += f"Configuration: Questions={num_questions}, Show Original Answer={show_original_answer}\n"
        setup_log_str += f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        self._log(setup_log_str)
        
    def load_capabilities_data(self):
        """
        Load capabilities data from the provided file path
        and extract questions the subject got wrong
        """
        try:
            self._log(f"Loading capabilities data from: {self.capabilities_file_path}")
            with open(self.capabilities_file_path, 'r', encoding='utf-8') as f:
                self.capabilities_data = json.load(f)
            
            # Extract results from either AOP game or regular capabilities measurement
            if 'phase1_results' in self.capabilities_data:
                capabilities_results = self.capabilities_data['phase1_results']
            else:
                capabilities_results = self.capabilities_data.get('results', {})
            
            if not capabilities_results:
                raise ValueError("No results found in capabilities data file")
            
            # Find questions that were answered incorrectly
            for q_id, result in capabilities_results.items():
                if isinstance(result, dict) and not result.get('is_correct', True):
                    # Get the original question data
                    question_data = result.get('question', {})
                    if not question_data:
                        continue
                    
                    # Add the subject's original wrong answer
                    question_data['original_answer'] = result.get('subject_answer')
                    
                    # Add to wrong questions list
                    self.wrong_questions.append(question_data)
            
            self._log(f"Found {len(self.wrong_questions)} questions that were answered incorrectly")
            
            if len(self.wrong_questions) < self.num_questions:
                self._log(f"Warning: Only {len(self.wrong_questions)} wrong questions available, but {self.num_questions} requested")
            
            return True
        except Exception as e:
            self._log(f"Error loading capabilities data: {e}")
            return False
    
    def select_questions(self):
        """
        Select a subset of questions for the game
        """
        if not self.wrong_questions:
            self._log("Error: No wrong questions available. Load capabilities data first.")
            return False
        
        # Determine how many questions to use
        num_to_use = min(self.num_questions, len(self.wrong_questions))
        
        # Randomly select questions
        self.selected_questions = random.sample(self.wrong_questions, num_to_use)
        self._log(f"Selected {len(self.selected_questions)} questions for the game")
        
        return True
    
    def run_game(self):
        """
        Run the Second-Chance Game
        """
        # Load capabilities data
        if not self.capabilities_data:
            if not self.load_capabilities_data():
                return False
        
        # Select questions
        if not self.selected_questions:
            if not self.select_questions():
                return False
        
        self._log("\n" + "="*10 + " Starting Second-Chance Game " + "="*10)
        
        # Initialize state
        message_history = []
        probs = None
        changed_answers = 0
        correct_after_feedback = 0
        
        # Process each question
        for i, question in enumerate(self.selected_questions):
            q_id = question.get('id', f"q_{i}")
            original_answer = question.get('original_answer')
            correct_answer = question.get('correct_answer')
            
            if not original_answer or not correct_answer:
                self._log(f"Warning: Missing data for question {q_id}, skipping")
                continue
            
            # Format question
            q_text = self._present_question(question)
                        
            self._log(f"\nPresenting question {i+1}/{len(self.selected_questions)}")
            self._log(f"Original answer: {original_answer}, Correct answer: {correct_answer}")
            
            # Get subject's answer
            if self.is_human_player:
                original_feedback = f"You previously answered this question and selected option {original_answer}, which was incorrect."
                redacted_feedback = f"You previously answered this question and selected [redacted], which was incorrect."
                feedback = original_feedback if self.show_original_answer else redacted_feedback
                prompt = f"{q_text}\n\n{feedback}\n\nPlease try again. Select the correct answer (A, B, C, or D): "
                print(prompt)
                new_answer = self._get_subject_answer(
                    list(question["options"].keys()), 
                    "Your answer (A, B, C, or D): "
                )
                if new_answer is None:
                    return False
            else:
                # For LLM subject
                setup_prompt = "I'm going to ask you a series of multiple choice questions. For each one, select the answer you think is best."
                llm_prompt = q_text + "\nYour choice (A, B, C, or D): "
                message_history = [{"role": "user", "content": setup_prompt + "\n\n" + llm_prompt}]
                content = original_answer if self.show_original_answer else "[redacted]"
                message_history.append({"role": "assistant", "content": content})
                content = "Your answer was incorrect. Choose again.\n\n" + llm_prompt # I lost your original answer, but I know it was wrong. Try to figure out what your first choice would have been, and then pick a different answer\n\n" + llm_prompt
                message_history.append({"role": "user", "content": content})
                
                # Get the answer without accumulating history
                new_answer, _, probs = self._get_llm_answer(
                    list(question["options"].keys()),
                    "",
                    setup_text=message_history,
                    keep_appending=False,
                    MAX_TOKENS=1
                )
            
            # Check if answer was changed
            answer_changed = (new_answer != original_answer) and new_answer in question["options"]
            if answer_changed:
                changed_answers += 1
                
            # Check if new answer is correct
            is_correct = (new_answer == correct_answer)
            if is_correct:
                correct_after_feedback += 1
            
            # Store result
            self.game_results[q_id] = {
                "question": question,
                "original_answer": original_answer,
                "new_answer": new_answer,
                "correct_answer": correct_answer,
                "answer_changed": answer_changed,
                "is_correct": is_correct,
                "probs": probs
            }
            
            self._log(f"New answer: {new_answer}, Changed: {answer_changed}, Correct: {is_correct}")
            
        # Save results
        self._save_results()
        
        # Analyze results
        self.analyze_results()
        
        return True
    
    def _save_results(self):
        """Save game results to file"""
        game_data = {
            "subject_id": self.subject_id,
            "subject_name": self.subject_name,
            "show_original_answer": self.show_original_answer,
            "num_questions": self.num_questions,
            "timestamp": time.time(),
            "results": self.game_results,
        }
            
        with open(self.game_data_filename, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)
            
        self._log(f"Game data saved to: {self.game_data_filename}")
    
    def analyze_results(self):
        """
        Analyze game results and generate statistics
        """
        if not self.game_results:
            raise ValueError("No game results to analyze. Run the game first.")
        
        # Create analysis
        analysis = "\n" + "="*10 + " Results Analysis " + "="*10 + "\n"
        analysis += f"Subject ID: {self.subject_id}\n"
        analysis += f"Original answers were {'shown' if self.show_original_answer else 'redacted'}\n"
        
        # Basic stats
        total_questions = len(self.game_results)
        changed_answers = sum(1 for r in self.game_results.values() if r.get('answer_changed'))
        correct_answers = sum(1 for r in self.game_results.values() if r.get('is_correct'))
        
        # Only count valid responses in the denominator
        valid_responses = sum(1 for r in self.game_results.values() if r.get('new_answer') in ["A", "B", "C", "D"])
        change_rate = changed_answers / valid_responses if valid_responses > 0 else 0
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        analysis += f"Total questions: {total_questions}\n"
        analysis += f"Answer change rate: {change_rate:.2%} ({changed_answers}/{valid_responses})\n"
        analysis += f"Accuracy after feedback: {accuracy:.2%} ({correct_answers}/{total_questions})\n"
        
        # Count how many changed answers were correct
        changed_and_correct = sum(1 for r in self.game_results.values() 
                                 if r.get('answer_changed') and r.get('is_correct'))
        
        # Count how many unchanged answers were correct
        unchanged_and_correct = sum(1 for r in self.game_results.values() 
                                   if not r.get('answer_changed') and r.get('is_correct'))
        
        if changed_answers > 0:
            changed_accuracy = changed_and_correct / changed_answers
            analysis += f"Accuracy when answer was changed: {changed_accuracy:.2%} ({changed_and_correct}/{changed_answers})\n"
        
        if total_questions - changed_answers > 0:
            unchanged_accuracy = unchanged_and_correct / (total_questions - changed_answers)
            analysis += f"Accuracy when answer was not changed: {unchanged_accuracy:.2%} ({unchanged_and_correct}/{total_questions - changed_answers})\n"
        
        # Statistical significance tests
        analysis += "\n--- Statistical Analysis ---\n"
        
        # Test if accuracy is better than random guessing (25%)
        binom_result = binomtest(k=correct_answers, n=total_questions, p=0.25)
        p_value = binom_result.pvalue
        
        analysis += f"Binomial test for accuracy vs. random guessing (25%): p-value = {p_value:.4f}\n"
        
        if p_value < 0.05:
            if accuracy > 0.25:
                analysis += "Interpretation: Accuracy is SIGNIFICANTLY HIGHER than random guessing (p < 0.05)\n"
            else:
                analysis += "Interpretation: Accuracy is SIGNIFICANTLY LOWER than random guessing (p < 0.05)\n"
        else:
            analysis += "Interpretation: No significant difference from random guessing (p >= 0.05)\n"
        
        
        # Print and log
        print(analysis)
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(analysis)
            
        # Return for further use
        return analysis

def main():
    """
    Main function to run the Second-Chance Game
    """
    # Configuration
    IS_HUMAN = False
    subject_name = "claude-3-5-sonnet-20241022" #"claude-3-haiku-20240307""claude-3-7-sonnet-20250219"#
    CAPABILITIES_FILE = "./pass_game_logs/aop_claude-3-5-sonnet-20241022_MMLU_1000_1745613577_1745613581_phase1_data.json"
    NUM_QUESTIONS = 92
    SHOW_ORIGINAL_ANSWER = True
    
    DATASET_NAME = "GPQA" if "GPQA" in CAPABILITIES_FILE.upper() else "MMLU" if "MMLU" in CAPABILITIES_FILE.upper() else "TruthfulQA"
    SUBJECT_ID = f"{subject_name.replace('/', '-')}_{DATASET_NAME}"
    try:
        
        # Create game instance
        game = SecondChanceGame(
            subject_id=SUBJECT_ID,
            subject_name=subject_name,
            capabilities_file_path=CAPABILITIES_FILE,
            num_questions=NUM_QUESTIONS,
            show_original_answer=SHOW_ORIGINAL_ANSWER,
            is_human_player=IS_HUMAN
        )
        
        # Run the game
        success = game.run_game()
        
        if success:
            print("\nSecond-Chance game completed successfully.")
        else:
            print("\nSecond-Chance game failed.")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExecution completed.")

if __name__ == "__main__":
    main()