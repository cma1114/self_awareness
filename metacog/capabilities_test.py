import time
import json
from load_and_format_datasets import load_and_format_dataset
from base_game_class import BaseGameClass

class CapabilitiesTest(BaseGameClass):
    """
    Just ask independent multiple-choice or short answer questions and record responses.
    """
    def __init__(self, subject_id, subject_name, questions, n_questions=None, is_human_player=False, resume_from=None, temperature=0.0):
        """
        Args:
            subject_id (str): Identifier for the subject/session
            subject_name (str): Name of the subject (model name for LLMs)
            questions (list): Formatted questions to use
            n_questions (int): How many questions to use
            is_human_player (bool): Whether the subject is a human player or an LLM
            resume_from (string): Filename to resume from (in case game got interrupted)
        """
        super().__init__(subject_id, subject_name, is_human_player, "capabilities_test_logs")
        self.n_questions = len(questions) if not n_questions else n_questions

        # Set up state variables
        self.results = {}
        self.questions = []
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = None
        self.temperature = temperature
        self.log_suffix = "_test_data"            

        if len(questions) < self.n_questions:
            raise ValueError(f"Not enough questions provided ({len(questions)}); ({self.n_questions} needed)")
        
        # Take the first n_questions
        self.questions = questions[:self.n_questions]
        self._log(f"Using {len(self.questions)} provided questions")

        if resume_from and resume_from != "":
            try:
                with open(resume_from, "r") as f:
                    prev_data = json.load(f)
            except Exception as e:
                print(f"Error opening resume file: {str(e)}")
                return False
            self.results = prev_data["results"]
            self._log(f"Resuming from {resume_from} holding {len(self.results)} questions")
            for rdict in self.results.values():
                if rdict["is_correct"] == True: self.correct_count +=1
                self.total_count += 1
            self.questions = [q for q in self.questions if q["id"] not in self.results]

    def _save_data(self):
        """Save data to file"""
        data = {
            "subject_id": self.subject_id,
            "timestamp": time.time(),
            "accuracy": self.accuracy,
            "results": self.results,
        }
                    
        filename = f"{self.log_base_name}{self.log_suffix}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        self._log(f"Data saved to: {filename}")
    
    def run_capabilities_measurement(self):
        """
        This measures a subject's performance on multiple choice questions and saves the results to a file.
        
        Returns:
            bool: True if completed successfully, False otherwise
            str: Path to the capabilities data file
        """
        start_message = f"\nStarting Capabilities Measurement for Subject: {self.subject_id}"
        self._log(start_message)
        self._log(f"Configuration: Questions={self.n_questions}")
        self._log("\n" + "="*10 + " Starting Capability Measuring " + "="*10)
        
        # Initialize state
        probs = None
        log_interval = 10
        
        # Process each question
        for i, question in enumerate(self.questions):
            q_text = self._present_question(question)

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
                # For LLM subject
                llm_prompt = q_text + "\nYour choice (A, B, C, or D): "
                setup_prompt = "I'm going to ask you a series of multiple choice questions. For each one, select the answer you think is best."
                if False:#question['id'] == "gpqa_train_rec42yAdAZnQgoibP" or question['id'] == "gpqa_train_recnTTKdBzfuoZ7w7": 
                    subject_answer = ""
                    probs = None
                else:
                    subject_answer, _, probs = self._get_llm_answer(
                        list(question["options"].keys()),
                        setup_prompt + "\n\n" + llm_prompt,
                        [], # no history
                        keep_appending=False,
                        MAX_TOKENS=None,#1,
                        temp=self.temperature
                    )
            
            # Check correctness
            if len(subject_answer) == 0:
                subject_decision = subject_answer
            else:
                arr=subject_answer.split()
                if arr[0] in list(question["options"].keys()):
                    subject_decision = arr[0]
                elif arr[-1] in list(question["options"].keys()):
                    subject_decision = arr[-1]
                else:
                    subject_decision = subject_answer

            is_correct = (subject_decision == question["correct_answer"])
            if is_correct:
                self.correct_count += 1
            
            # Store result
            if subject_decision != "":
                self.results[question["id"]] = {
                    "question": question,
                    "subject_answer": subject_decision,
                    "is_correct": is_correct,
                    "probs": probs 
                }
            self.total_count += 1
            print(f"Completed question {self.total_count}/{len(self.questions)}")
            self.accuracy = self.correct_count / self.total_count 
            if (i+1)%log_interval == 0: self._save_data()
            
        # Calculate accuracy
        self.accuracy = self.correct_count / len(self.questions)
        
        # Summary
        summary = f"\nCapabilities Test Complete. Accuracy: {self.accuracy:.2%} ({self.correct_count}/{len(self.questions)})"
        self._log(summary)
        
        self._save_data()
                    
        # Return the path to the capabilities data file
        capabilities_file_path = f"{self.log_base_name}{self.log_suffix}.json"
        self._log(f"Capabilities measurement completed. Results saved to: {capabilities_file_path}")
        return True, capabilities_file_path

    def run_capabilities_measurement_sa(self):
        """
        This measures a subject's performance on short answer questions and saves the results to a file.
        
        Returns:
            bool: True if completed successfully, False otherwise
            str: Path to the capabilities data file
        """
        start_message = f"\nStarting Capabilities Measurement for Subject: {self.subject_id}"
        self._log(start_message)
        self._log(f"Configuration: Questions={self.n_questions}")
        self._log("\n" + "="*10 + " Starting Capability Measuring " + "="*10)
        
        # Initialize state
        probs = None
        log_interval = 10
        self.accuracy = None
        
        # Process each question
        for i, question in enumerate(self.questions):
            q_text = self._present_question(question)

            # Get subject's answer
            if self.is_human_player:
                print(q_text)
                subject_answer = self._get_subject_answer(
                    list(question["options"].keys()), 
                    "Your answer: "
                )
                if subject_answer is None:
                    return False
            else:
                # For LLM subject
                llm_prompt = q_text + "\nYour answer: "
                setup_prompt = "I'm going to ask you a series of short answer questions. For each one, respond as succinctly as possible."
                if False:#question['id'] == "gpqa_train_rec42yAdAZnQgoibP" or question['id'] == "gpqa_train_recnTTKdBzfuoZ7w7": 
                    subject_answer = ""
                    probs = None
                else:
                    subject_answer, _, probs = self._get_llm_answer(
                        None,
                        setup_prompt + "\n\n" + llm_prompt,
                        [], # no history
                        keep_appending=False,
                        MAX_TOKENS=None,
                        temp=self.temperature
                    )
                        
            # Store result
            if subject_answer != "":
                self.results[question["id"]] = {
                    "question": question,
                    "subject_answer": subject_answer,
                    "is_correct": None,
                    "probs": probs 
                }
            self.total_count += 1
            print(f"Completed question {self.total_count}/{len(self.questions)}")
            if (i+1)%log_interval == 0: self._save_data()
            
        # Summary
        summary = f"\nCapabilities Test Complete."
        self._log(summary)
        
        self._save_data()
                    
        # Return the path to the capabilities data file
        capabilities_file_path = f"{self.log_base_name}{self.log_suffix}.json"
        self._log(f"Capabilities measurement completed. Results saved to: {capabilities_file_path}")
        return True, capabilities_file_path

def main():
    IS_HUMAN = False
    DATASET_NAME = "GPSA"    # "TruthfulQA" or "GPQA" or "MMLU or SimpleQA" or "SimpleMC" or "GPSA"
    subject_name = "claude-3-sonnet-20240229"#"claude-sonnet-4-20250514"#"deepseek-chat"#"gpt-4o-2024-08-06"#"grok-3-latest"#'gemini-2.0-flash-001'#"claude-3-5-sonnet-20241022" #"gemini-2.5-flash-preview-04-17"#"meta-llama/Meta-Llama-3.1-405B-Instruct"#"meta-llama/Meta-Llama-3.1-405B"#"gemini-2.5-pro-exp-03-25"#"claude-3-7-sonnet-20250219"#"gpt-4-turbo-2024-04-09"#"claude-3-haiku-20240307"#"Chris"#
    resume_from = None#"./capabilities_test_logs/meta-llama-Meta-Llama-3.1-405B-Instruct_GPQA_447_1746367623_test_data.json" 
    N_QUESTIONS = 447#500#   # Number of questions for capabilities measurement
    temp = 0.0
    
    SUBJECT_ID = f"{subject_name.replace('/', '-')}_{DATASET_NAME}_{N_QUESTIONS}"
    try:
        # Load questions for capabilities measurement
        print(f"Loading {N_QUESTIONS} questions for capabilities measurement...")
        formatted_questions = load_and_format_dataset(DATASET_NAME, N_QUESTIONS)
            
        if not formatted_questions or len(formatted_questions) < N_QUESTIONS:
            print(f"Error: Not enough questions available ({len(formatted_questions) if formatted_questions else 0}). Needed: {N_QUESTIONS}")
            return
        
        # Create game instance for capabilities measurement
        game = CapabilitiesTest(
            subject_id=SUBJECT_ID,
            subject_name=subject_name,
            questions=formatted_questions,
            n_questions=N_QUESTIONS,
            is_human_player=IS_HUMAN,
            resume_from=resume_from,
            temperature=temp
        )
                    
        # Run capabilities measurement
        if DATASET_NAME == "SimpleQA" or DATASET_NAME == "GPSA": success, capabilities_file = game.run_capabilities_measurement_sa()
        else: success, capabilities_file = game.run_capabilities_measurement()
        
        if success:
            print(f"\nCapabilities measurement completed successfully.")
            print(f"Results saved to: {capabilities_file}")
        else:
            print("\nCapabilities measurement failed.")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExecution completed.")

if __name__ == "__main__":
    main()