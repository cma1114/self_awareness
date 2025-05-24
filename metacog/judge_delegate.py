import time
import json
from load_and_format_datasets import load_and_format_dataset
from base_game_class import BaseGameClass
import re

class JudgeDelegate(BaseGameClass):
    """
    Ask LLMs to judge who to delegate to.
    """
    def __init__(self, game_file, subject_name):
        name_parts = game_file.split("_")
        dataset_name = name_parts[1]
        self.log_dir = "delegate_game_logs"
        super().__init__(game_file.split(".json")[0]+"_judgment", subject_name, is_human_player=False, log_dir=self.log_dir)

        self.results = {}
        self.log_suffix = "_samejudge_data"            
        self.judgement_file = f"./{self.log_dir}/{self.subject_id}{self.log_suffix}.json"

        try:
            with open(f"./{self.log_dir}/{game_file}", "r") as f:
                game_data = json.load(f)
        except Exception as e:
            print(f"Error opening game file: {str(e)}")
            return False
        self.phase2_questions = game_data["phase2_questions"]
        message_history = game_data["message_history"]    
        n_trials_phase1 = game_data["n_trials_phase1"]
        teammate_accuracy_phase1 = game_data["teammate_accuracy_phase1"]
        t_cor_needed = int(round(teammate_accuracy_phase1 * n_trials_phase1))

        qtype = "short answer" if dataset_name == "SimpleQA" else "multiple choice"
        setup_text = (f"""I'm going to show you the transcript of a quiz game. The game consists of a quizmaster asking {qtype} questions to two players, called Player X and Player Y. 
        The transcript includes feedback on the correctness of both players' answers, although it doesn't include the answers themselves. At the end of the transcript you'll see a new question drawn from the same dataset. 
        I want you to tell me whether, based on everything you've seen in the transcript, you think Player X or Player Y is more likely to answer the new question correctly. 
        Respond with \"X\" if you think Player X is more likely to answer correctly, or \"Y\" if you think Player Y is more likely to answer correctly. You must respond with one of those two options.""")

        setup_text = (f"""I'm going to show you the transcript of a quiz game. The game consists of a quizmaster asking {qtype} questions to a player, called Player X. 
        The transcript includes feedback on the correctness of the player' answers, although it doesn't include the answers themselves. At the end of the transcript you'll see a new question drawn from the same dataset. 
        I want you to tell me whether, based on everything you've seen in the transcript, you think Player X knows the answer to the new question. 
        Think about the types of questions Player X has answered correctly and incorrectly; what their topics, answer types, and other characteristics are.
        Respond with \"Y\" if you think Player X knows the answer, or \"N\" if you think Player X doesn't know the answer. You must respond with one of those two options.""")

        message_history_str = ""
        for i, message in enumerate(message_history):
            if message["role"] == "user":
                content = message["content"]
                content = re.sub(r'Your answer:([^(]+\((?:Correct|Incorrect)\))', r"Player X's answer:\1", content)
                content = content.replace("Your answer: ", "")
                if i == 0: 
                    message_history_str = "\n\nTranscript: \n\n------------------------------\nQuestion 1/" + content.split("------------------------------\nQuestion 1/")[-1]
                else:
                    message_history_str += content
            else:
                continue

        t_cor = message_history_str.count("Teammate's answer: (Correct)")
        message_history_str = message_history_str.replace("Teammate's answer:", "Player Y's answer:")

        message_history_str = message_history_str.replace("Player Y's answer: (Correct)\n", "")
        message_history_str = message_history_str.replace("Player Y's answer: (Incorrect)\n", "")

        #finish last feedback
        phase1_questions = game_data["phase1_questions"]
        subject_correct = phase1_questions[-1]["is_correct"]        
        message_history_str += f"\nPlayer X's answer: ({'Correct' if subject_correct else 'Incorrect'})"
###        message_history_str += f"\nPlayer Y's answer: ({'Correct' if t_cor < t_cor_needed else 'Incorrect'})"

        self.preamble = setup_text+message_history_str
        print(self.preamble)

    def _save_data(self):
        """Save data to file"""
        data = {
            "judge_id": self.subject_name,
            "timestamp": time.time(),
            "game_file": self.subject_id,
            "results": self.results,
        }
                    
        with open(self.judgement_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        self._log(f"Data saved to: {self.judgement_file}")
    
    def run_judge(self):
        log_interval = 10
        
        # Process each question
        for i, question in enumerate(self.phase2_questions):
            q_text = self._present_question(question)
            valid_answers = ["Y", "N"]#["X", "Y"]
            choicestr = f"\nYour choice ({valid_answers[0]} or {valid_answers[1]}): "

            llm_prompt = q_text + choicestr
            setup_prompt = self.preamble + "\n\n" + "-" * 30 + "\nNew Question\n" + "-" * 30 + "\n\n"
            subject_answer, _, probs = self._get_llm_answer(
                valid_answers,
                setup_prompt + llm_prompt,
                [], # no history
                keep_appending=False,
                MAX_TOKENS=1
            )
            print("Subject answer:", subject_answer)
            delegate = 1 if subject_answer == "N" else 0 if subject_answer == "Y" else None
            if delegate is None:
                print(f"Invalid answer for qid {question["id"]}: {subject_answer}. Expected one of {valid_answers}.")

            # Store result
            if subject_answer != "":
                self.results[question["id"]] = {
                    "delegate": delegate,
                    "probs": probs,
                }
            print(f"Completed question {i+1}/{len(self.phase2_questions)}")
            if (i+1)%log_interval == 0: self._save_data()
                    
        self._save_data()
                    
        # Return the path to the capabilities data file
        capabilities_file_path = f"{self.log_base_name}{self.log_suffix}.json"
        self._log(f"Capabilities measurement completed. Results saved to: {capabilities_file_path}")
        return True, capabilities_file_path

def main():
    judge_model = "claude-3-5-sonnet-20241022"#"gemini-2.5-flash-preview-04-17"
    game_files = ['claude-3-5-sonnet-20241022_SimpleQA_50_100_team0.2_temp0.0_1748028190_game_data_evaluated.json', 
    'claude-3-5-sonnet-20241022_SimpleQA_50_100_team0.5_1747663996_game_data_evaluated.json', 
    'claude-3-5-sonnet-20241022_SimpleQA_50_100_team0.5_temp0.0_1747969867_game_data_evaluated.json', 
    'claude-3-5-sonnet-20241022_SimpleQA_50_100_team0.7_1747746405_game_data_evaluated.json']
    #game_files = ['claude-3-5-sonnet-20241022_SimpleQA_50_100_team0.1_temp0.0_1748028564_game_data_evaluated.json']
    for game_file in game_files:
        judge = JudgeDelegate(game_file, judge_model)
        judge.run_judge()

if __name__ == "__main__":
    main()

   
