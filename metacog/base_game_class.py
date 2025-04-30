import torch
import os
import time
import re
import copy
import anthropic
from openai import OpenAI
from nnsight import LanguageModel
from nnsight import CONFIG
from google import genai
from google.genai import types
import requests
from dotenv import load_dotenv
load_dotenv()

# Load API keys
anthropic_api_key = os.environ.get("ANTHROPIC_SPAR_API_KEY")
hyperbolic_api_key = os.environ.get("HYPERBOLIC_API_KEY")
CONFIG.set_default_api_key(os.environ.get("NDIF_API_KEY"))
gemini_api_key = os.environ.get("GEMINI_API_KEY")

class BaseGameClass:
    """Base class for all games with common functionality."""

    def __init__(self, subject_id, subject_name, questions=None, is_human_player=False, log_dir="game_logs"):
        """Initialize with common parameters."""
        self.subject_id = subject_id
        self.subject_name = subject_name
        self.is_human_player = is_human_player

        self._setup_provider()
        self._setup_logging(log_dir)

    def _setup_provider(self):
        """Determine provider based on model name."""
        if not self.is_human_player:
            self.provider = "Anthropic" if self.subject_name.startswith("claude") else "OpenAI" if "gpt" in self.subject_name else "NDIF" if re.match(r"meta-llama/Meta-Llama-3\.1-\d+B", self.subject_name) else "Google" if self.subject_name.startswith("gemini") else "Hyperbolic"
            if self.provider == "Anthropic": 
                self.client = anthropic.Anthropic(api_key=anthropic_api_key)
            elif self.provider == "OpenAI":
                self.client = OpenAI()
            elif self.provider == "NDIF":
                self.client = LanguageModel(self.subject_name, device_map="auto")
            elif self.provider == "Google":
                self.client = genai.Client(api_key=gemini_api_key)

    def _setup_logging(self, log_dir):
        """Set up logging files and directories."""
        os.makedirs('./pass_game_logs', exist_ok=True)
        timestamp = int(time.time())
        self.log_base_name = f"./pass_game_logs/aop_{self.subject_id}_{timestamp}"
        self.log_filename = f"{self.log_base_name}.log"
        self.game_data_filename = f"{self.log_base_name}_game_data.json"

    def _log(self, message):
        """Write to log file and console."""
        print(message)
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def _get_llm_answer(self, options, q_text, message_history, keep_appending=True, setup_text=""):
        """Gets answer from LLM model"""
        # Prepare common data
        user_msg = {"role": "user", "content": q_text}
        resp = ""
        token_probs = None
        options_str = ", ".join(options[:-1]) + f", or {options[-1]}"
        system_msg = f"{setup_text}\nOutput ONLY the letter of your choice: {options_str}.\n"
        
        MAX_ATTEMPTS = 10
        delay = 1.0
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
                        messages=formatted_messages,
                        logprobs=True,
                        top_logprobs=len(options)                     
                    )    
                    resp = completion.choices[0].message.content.strip()
                    entry = completion.choices[0].logprobs.content[0]
                    tokens = [tl.token for tl in entry.top_logprobs]
                    logprob_tensor = torch.tensor([tl.logprob for tl in entry.top_logprobs])
                    prob_tensor = torch.nn.functional.softmax(logprob_tensor, dim=0)
                    token_probs = dict(zip(tokens, prob_tensor.tolist()))
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
                            prompt += f"My answer is:\n{msg['content']}\n"
                    if keep_appending:
                        message_history.append(user_msg)
                    prompt += f"User:\n{system_msg}\nYou are an Assistant.\n{q_text}\nThe Assistant responds only with {options_str}\nMy answer is:\n"
                    print(f"prompt={prompt}")
                    with self.client.generate(prompt, max_new_tokens=2, temperature=0, remote=True) as tracer:
                        out = self.client.generator.output.save()
                    resp = self.client.tokenizer.decode(out[0][len(self.client.tokenizer(prompt)['input_ids']):]).strip().upper()[0]
                elif self.provider == "Google":
                    formatted_messages = []
                    for msg in message_history:
                        if msg["role"] == "user":
                            formatted_messages.append(types.Content(role='user', parts=[types.Part.from_text(text=msg['content'])]))
                        elif msg["role"] == "assistant":
                            formatted_messages.append(types.Content(role='model', parts=[types.Part.from_text(text=msg['content'])]))
                    formatted_messages.append(types.Content(role='user', parts=[types.Part.from_text(text=user_msg['content'])]))
                    if keep_appending:
                        message_history.append(user_msg)
                    #print(f"system_msg={system_msg}")                     
                    #print(f"formatted_messages={formatted_messages}")             
                    message = self.client.models.generate_content(
                        model=self.subject_name,
                        contents=formatted_messages,
                        config=types.GenerateContentConfig(
                            system_instruction=system_msg,
                            max_output_tokens=1,
                            temperature=0.0 + attempt * 0.1,
                        ), 
                    )
                    resp = message.text.strip().upper()
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                if resp in options:
                    break
                print(f"Bad LLM response: {resp} (attempt {attempt + 1})")
            except Exception as e:
                self._log(f"Error: {e}")
                if "429" in str(e):
                    # Rate limit error, wait and retry
                    time.sleep(delay)
                    delay = min(delay*2,15)
                    attempt -= 1 #don't increase temperature
        
        if keep_appending: message_history.append({"role": "assistant", "content": resp})
        if resp not in options:
            self._log(f"Failed to get valid response for text: {q_text}; response: {resp}")

        return resp, message_history, token_probs

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
        elif question_num is not None and total_questions is None:
            formatted_question += f"Question {question_num}:\n"
        else:
            formatted_question += "Question:\n"
            
        formatted_question += question_data["question"] + "\n"
        formatted_question += "-" * 10 + "\n"
        
        for key, value in question_data["options"].items():
            formatted_question += f"  {key}: {value}\n"
        
        formatted_question += "-" * 30
        return formatted_question
