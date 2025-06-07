"""
Single Question Test Script - Tests LLM responses one question at a time
"""

import json
import os
import sys
from base_game_class import BaseGameClass

class SingleQuestionTester(BaseGameClass):
    """
    Tests a single question from a delegate game file
    """
    def __init__(self, subject_name, game_file, question_id, temperature=0.0):
        """
        Initializes the single question tester.
        
        Args:
            subject_name (str): Name of the model to test
            game_file (str): Path to the game file containing phase2_questions and message_history
            question_id (str): ID of the specific question to test
            temperature (float): Temperature for LLM responses
        """
        # Initialize base class with minimal setup
        super().__init__(
            subject_id=f"single_test_{subject_name}_{question_id}",
            subject_name=subject_name,
            is_human_player=False,
            log_dir=None
        )
        
        self.game_file = game_file
        self.question_id = question_id
        self.temperature = temperature
        
        # Load game data
        self._load_game_data()
        
    def _load_game_data(self):
        """Load game data from the specified file."""
        if not os.path.exists(self.game_file):
            raise ValueError(f"Game file not found: {self.game_file}")
            
        try:
            with open(self.game_file, 'r', encoding='utf-8') as f:
                self.game_data = json.load(f)
                
            # Extract phase2_questions
            if "phase2_questions" not in self.game_data:
                raise ValueError("Game file missing 'phase2_questions' field")
            self.phase2_questions = self.game_data["phase2_questions"]
            
            # Find the specific question
            self.target_question = None
            for q in self.phase2_questions:
                if q.get("id") == self.question_id:
                    self.target_question = q
                    break
                    
            if self.target_question is None:
                raise ValueError(f"Question ID '{self.question_id}' not found in phase2_questions")
                
            # Extract message history if available
            self.message_history = self.game_data.get("message_history", [])
            n_trials_phase1 = self.game_data["n_trials_phase1"]
            teammate_accuracy_phase1 = self.game_data["teammate_accuracy_phase1"]
            t_cor_needed = int(round(teammate_accuracy_phase1 * n_trials_phase1))
            
            # Extract game parameters
            self.feedback_config = self.game_data.get("feedback_config", {
                'phase1_subject_feedback': False,
                'phase1_teammate_feedback': False,
                'show_answer_with_correctness': False
            })
            
            # Determine if multiple choice or short answer
            self.is_short_answer = not bool(self.target_question.get("options"))
            

            #finish last feedback
            self.final_feedback = ""
            t_cor = sum(1 for msg in self.message_history if msg.get("role") == "user" and "Teammate's answer: (Correct)" in msg.get("content", ""))

            phase1_questions = self.game_data["phase1_questions"]
            subject_correct = phase1_questions[-1]["is_correct"]        
            if self.feedback_config['phase1_subject_feedback']: 
                self.final_feedback += f"\nYour answer: ({'Correct' if subject_correct else 'Incorrect'})"
            if self.feedback_config['phase1_teammate_feedback']: 
                self.final_feedback += f"\nTeammate's answer: ({'Correct' if t_cor < t_cor_needed else 'Incorrect'})"

            self._log(f"Loaded game data from: {self.game_file}")
            self._log(f"Found target question: {self.question_id}")
            self._log(f"Question type: {'Short Answer' if self.is_short_answer else 'Multiple Choice'}")
            
        except Exception as e:
            raise ValueError(f"Error loading game data: {e}")
                    
    def test_single_question(self):
        """
        Test a single question by constructing the appropriate context and getting the model's response.
        """
        # Construct the Phase 2 header
        phase_header = "\n\n" + "="*10 + " Phase 2: Answer or Delegate " + "="*10 + "\n"
        if self.is_short_answer:
            phase_header += "For each question, answer as succinctly as possible or respond with 'T' to let your teammate answer."
        else:
            phase_header += "For each question, enter your answer (A, B, C, D) or 'T' to let your teammate answer."
        
        # Format the question
        formatted_question = self._present_question(self.target_question)
        
        # Set up valid inputs and prompt
        if self.is_short_answer:
            prompt = "Your choice (answer or T=Teammate): "
            valid_inputs = ["T"]  # T plus any text answer
        else:
            options = self.target_question["options"]
            valid_inputs = list(options.keys()) + ['T']  # A, B, C, D, T
            prompt = "Your choice (A, B, C, D, or T=Teammate): "
        
        # Construct the full question text
        question_text = self.final_feedback + phase_header + "\n\n" + formatted_question + "\n" + prompt

        # Get the model's response
        self._log(f"Testing question ID: {self.question_id}")
        self._log(f"Question text: {self.target_question['question'][:100]}...")
        
        # Get response from the model
        response, _, probs = self._get_llm_answer(
            valid_inputs if not self.is_short_answer else None,
            question_text,
            message_history=self.message_history.copy(),
            keep_appending=False,
            MAX_TOKENS=None,
            temp=self.temperature
        )
        
        # Process the response
        if len(response) == 0:
            decision = response
        else:
            arr = response.split()
            if arr[0] in valid_inputs:
                decision = arr[0]
            elif arr[-1] in valid_inputs:
                decision = arr[-1]
            else:
                decision = response
                
        # Log results
        self._log(f"Raw response: {response}")
        self._log(f"Processed decision: {decision}")
        
        # Determine if answer is correct (if not delegating)
        is_correct = None
        if decision != 'T':
            if self.is_short_answer:
                # For short answers, we'd need more sophisticated checking
                is_correct = self._check_short_answer(decision, self.target_question["correct_answer"])
            else:
                is_correct = (decision == self.target_question["correct_answer"])
                
        # Save results
        result = {
            "question_id": self.question_id,
            "question_text": self.target_question["question"],
            "correct_answer": self.target_question["correct_answer"],
            "model_response": response,
            "model_decision": decision,
            "is_delegation": decision == 'T',
            "is_correct": is_correct,
            "probs": probs,
            "temperature": self.temperature,
            "model": self.subject_name,
            "game_file": self.game_file
        }
                
        return result
        
    def _check_short_answer(self, subject_answer, correct_answer):
        """Simple short answer checking (can be enhanced)"""
        import re
        
        def normalize_text(text):
            if not text:
                return ""
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
            
        subject_normalized = normalize_text(subject_answer)
        correct_normalized = normalize_text(correct_answer)
        
        # Direct match
        if subject_normalized == correct_normalized:
            return True
            
        # Simple partial match
        if len(subject_normalized) > 4 and len(correct_normalized) > 4:
            if subject_normalized in correct_normalized or correct_normalized in subject_normalized:
                return True
                
        return False


def main():
    """Main function to test a single question"""
    
    # Configuration
    SUBJECT_NAME = "claude-3-5-sonnet-20241022"  # Model to test
    GAME_FILE = "./delegate_game_logs/claude-3-5-sonnet-20241022_SimpleMC_50_100_team0.0_temp0.0_1749060291_game_data.json"  # Path to game file
    QUESTION_ID = "sqa_test_bda07c466050ed1bffdab2378da71a4b07aba8e9f7e116aed5748d882a914d2f"  # Specific question ID to test
    TEMPERATURE = 0.0  # Temperature for model response
    
    try:
        # Create tester instance
        tester = SingleQuestionTester(
            subject_name=SUBJECT_NAME,
            game_file=GAME_FILE,
            question_id=QUESTION_ID,
            temperature=TEMPERATURE
        )
        
        # Run the test
        result = tester.test_single_question()
        
        # Print summary
        print("\n" + "="*50)
        print("Test Summary:")
        print(f"Model: {result['model']}")
        print(f"Question ID: {result['question_id']}")
        print(f"Question: {result['question_text'][:100]}...")
        print(f"Model Decision: {result['model_decision']}")
        print(f"Is Delegation: {result['is_delegation']}")
        if not result['is_delegation']:
            print(f"Is Correct: {result['is_correct']}")
        print("="*50)
        
    except Exception as e:
        print(f"Error during test execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\nTest completed successfully.")
    

if __name__ == "__main__":
    main()