import torch
import os
import time
import re
import math
import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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
anthropic_api_key = os.environ.get("ANTHROPIC_SPAR_API_KEY")##os.environ.get("ANTHROPIC_API_KEY")##
hyperbolic_api_key = os.environ.get("HYPERBOLIC_API_KEY")
CONFIG.set_default_api_key(os.environ.get("NDIF_API_KEY"))
gemini_api_key = os.environ.get("GEMINI_API_KEY")
xai_api_key = os.environ.get("XAI_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")    

class BaseGameClass:
    """Base class for all games with common functionality."""

    def __init__(self, subject_id, subject_name, is_human_player=False, log_dir="game_logs"):
        """Initialize with common parameters."""
        self.subject_id = subject_id
        self.subject_name = subject_name
        self.is_human_player = is_human_player

        self._setup_logging(log_dir)
        self._setup_provider()

    def _setup_provider(self):
        """Determine provider based on model name."""
        if not self.is_human_player:
            if self.subject_name.startswith("claude"):
                self.provider = "Anthropic"
            elif "gpt" in self.subject_name or self.subject_name.startswith("o3") or self.subject_name.startswith("o1"):
                self.provider = "OpenAI"
            elif self.subject_name.startswith("gemini"):
                self.provider = "Google"
            elif self.subject_name.startswith("grok"):
                self.provider = "xAI"
            elif re.match(r"meta-llama/Meta-Llama-3\.1-\d+B", self.subject_name):
                self.provider = "NDIF"###"Hyperbolic"###
            elif "deepseek" in self.subject_name:
                self.provider = "DeepSeek"
            else:
                self.provider = "Hyperbolic"

            if self.provider == "Anthropic": 
                self.client = anthropic.Anthropic(api_key=anthropic_api_key)
            elif self.provider == "OpenAI":
                self.client = OpenAI()
            elif self.provider == "Google":
                self.client = genai.Client(vertexai=True, project="gen-lang-client-0693193232", location="us-central1") if 'gemini-1.5' not in self.subject_name else genai.Client(api_key=gemini_api_key)
            elif self.provider == "xAI":
                self.client = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1",)
            elif self.provider == "NDIF":
                self.client = LanguageModel(self.subject_name, device_map="auto")
            elif self.provider == "DeepSeek":
                self.client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

            self._log(f"Provider: {self.provider}")

    def _setup_logging(self, log_dir):
        """Set up logging files and directories."""
        if log_dir:
            os.makedirs(f"./{log_dir}", exist_ok=True)
            timestamp = int(time.time())
            self.log_base_name = f"./{log_dir}/{self.subject_id}_{timestamp}"
            self.log_filename = f"{self.log_base_name}.log"
            self.game_data_filename = f"{self.log_base_name}_game_data.json"
        else:
            self.log_filename = None

    def _log(self, message):
        """Write to log file and console."""
        print(message)
        if self.log_filename:
            with open(self.log_filename, 'a', encoding='utf-8') as f:
                f.write(message + "\n")

    def _call_with_timeout(self, fn, timeout=60):
        """
        Run `fn()` in a worker thread and return its result.
        Raises TimeoutError if fn() doesn't finish in `timeout` seconds.
        
        Note: This implementation avoids context managers which can hang on timeout.
        """
        # Create executor and submit the task
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn)
        
        try:
            # Wait for the result with timeout
            result = future.result(timeout=timeout)
            # Only if we get here, shutdown with wait=True is safe
            executor.shutdown(wait=True)
            return result
        except (TimeoutError, Exception) as e:
            # For any exception, attempt to cancel future and shutdown without waiting
            future.cancel()
            executor.shutdown(wait=False)
            # Log the error for debugging
            self._log(f"Thread execution error: {type(e).__name__}: {e}")
            # Re-raise for retry handling
            raise

    def _get_llm_answer(self, options, q_text, message_history, keep_appending=True, setup_text="", MAX_TOKENS=1, temp=0.0, accept_any=True):
        """Gets answer from LLM model"""
        # Prepare common data
        user_msg = {"role": "user", "content": q_text}
        if options: 
            options_str = " or ".join(options) if len(options) == 2 else ", ".join(options[:-1]) + f", or {options[-1]}"
            system_msg = f"{setup_text}\nOutput ONLY the letter of your choice: {options_str}.\n"
        else:
            system_msg = f"{setup_text}"
            options = " " #just to have len(options) be 1 for number of logprobs to return in short answer case
        
        MAX_ATTEMPTS = 10 #for bad resp format
        MAX_CALL_ATTEMPTS = 4 #for rate limit/timeout/server errors
        delay = 1.0
        attempt = 0
        temp_inc = -0.05 if temp > 0.5 else 0.05
        resp = ""
        token_probs = None
        for callctr in range(MAX_CALL_ATTEMPTS):
            def model_call():
                self._log(f"In model_call, provider={self.provider}, attempt={attempt + 1}")
                resp = ""
                if self.provider == "Anthropic":
                    if keep_appending:
                        message_history.append(user_msg)
                        formatted_messages = message_history
                    else:
                        formatted_messages = copy.deepcopy(message_history)
                        formatted_messages.append(user_msg)
                    #print(f"\nsystem_msg={system_msg}")                     
                    #print(f"\nformatted_messages={formatted_messages}\n")             
                    message = self.client.messages.create(
                        model=self.subject_name,
                        max_tokens=(MAX_TOKENS if MAX_TOKENS else 1024),
                        temperature=temp + attempt * temp_inc,
                        **({"system": system_msg} if system_msg != "" else {}),
                        messages=formatted_messages
                    )
                    #print(f"message={message}")
                    resp = message.content[0].text.strip()
                    return resp, None
                elif self.provider == "OpenAI" or self.provider == "xAI" or self.provider == "DeepSeek":
                    if keep_appending:
                        if system_msg != "": message_history.append({"role": "system", "content": system_msg})
                        message_history.append(user_msg)
                        formatted_messages = message_history
                    else:
                        formatted_messages = copy.deepcopy(message_history)
                        if system_msg != "": formatted_messages.append({"role": "system", "content": system_msg})
                        formatted_messages.append(user_msg)
                    #print(f"formatted_messages={formatted_messages}")
                    completion = self.client.chat.completions.create(
                        model=self.subject_name,
                        **({"max_completion_tokens": MAX_TOKENS} if self.subject_name.startswith("o") else {"max_tokens": MAX_TOKENS}),
                        **({"temperature": temp + attempt * temp_inc} if not self.subject_name.startswith("o") else {}),
                        messages=formatted_messages,
                        **({"logprobs": True} if not self.subject_name.startswith("o") else {}),
                        **({"top_logprobs": len(options)} if not self.subject_name.startswith("o") else {})
                    )   
                    #print(f"completion={completion}") 
                    resp = completion.choices[0].message.content.strip()
                    if 'o3' in self.subject_name: return resp, None
                    if len(options) == 1: #short answer, just average
                        token_logprobs = completion.choices[0].logprobs.content    
                        top_probs = []
                        for token_logprob in token_logprobs:
                            if token_logprob.top_logprobs is None or len(token_logprob.top_logprobs) == 0:
                                top_logprob_value = 0.0
                            else:
                                top_logprob_value = token_logprob.top_logprobs[0].logprob
                            top_prob = top_logprob_value
                            top_probs.append(top_prob)
                        token_probs = {resp: math.exp(sum(top_probs) / len(top_probs))}
                    else:
                        entry = completion.choices[0].logprobs.content[0]
                        if len(entry.top_logprobs) < len(options) and callctr < MAX_CALL_ATTEMPTS - 1:  
                            raise ValueError("full logprobs not returned")
                        try:
                            tokens = [tl.token for tl in entry.top_logprobs]
                            probs = [math.exp(tl.logprob) for tl in entry.top_logprobs]
                            token_probs = dict(zip(tokens, probs))
                        #logprob_tensor = torch.tensor([tl.logprob for tl in entry.top_logprobs])
                        #prob_tensor = torch.nn.functional.softmax(logprob_tensor, dim=0)
                        #token_probs = dict(zip(tokens, prob_tensor.tolist()))
                        except Exception as e:
                            if callctr < MAX_CALL_ATTEMPTS - 1: raise ValueError(f"Error processing logprobs: {e}")
                            else: return resp, None
                    #print(f"resp={resp}, token_probs={token_probs}")
                    return resp, token_probs
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
                            "max_tokens": MAX_TOKENS,
                            "temperature": temp + attempt * temp_inc,
                            "logprobs": True,
                            "top_logprobs": len(options)
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
                            "max_tokens": MAX_TOKENS,
                            "temperature": temp + attempt * temp_inc,
                            "logprobs": True,
                            "top_logprobs": len(options)
                        }                
                    response = requests.post(
                        url,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {hyperbolic_api_key}"
                        },
                        json=payload
                    )
                    result = response.json()
                    if not result["choices"][0]['logprobs']: raise ValueError("logprobs not returned")
                    resp = result['choices'][0]['message']['content'].strip()

                    if len(options) == 1:                     # ---------- short‑answer ----------
                        token_logprobs = result['choices'][0]['logprobs']['content']
                        top_probs = []

                        for tok in token_logprobs:
                            top_list = tok.get('top_logprobs') or []      # [] if None
                            if top_list:
                                top_logprob_value = top_list[0]['logprob']
                            else:
                                top_logprob_value = tok['logprob']        # <-- fallback
                            top_probs.append(top_logprob_value)

                        token_probs = {resp: math.exp(sum(top_probs) / len(top_probs))}

                    else:                                      # ---------- multiple choice ----------
                        entry   = result['choices'][0]['logprobs']['content'][0]
                        tokens  = [alt['token'].strip() for alt in entry['top_logprobs']]
                        probs   = [math.exp(alt['logprob'])     for alt in entry['top_logprobs']]
                        token_probs = dict(zip(tokens, probs))
                    return resp, token_probs
                elif self.provider == "NDIF":
                    prompt = ""
                    # Build prompt from message history and current question
                    if "Instruct" in self.subject_name:
                        if len(system_msg) > 0:
                            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
                        for msg in message_history:
                            if msg["role"] == "user":
                                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
                            elif msg["role"] == "assistant":
                                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
                        if keep_appending:
                            message_history.append(user_msg)
                        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{q_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    else:
                        for msg in message_history:
                            if msg["role"] == "user":
                                prompt += f"User: {msg['content']}\n"
                            elif msg["role"] == "assistant":
                                prompt += f"Assistant:\n{msg['content']}\n"
                        if keep_appending:
                            message_history.append(user_msg)
                        prompt += f"User:\n{system_msg}\nYou are an Assistant.\n{q_text}\nThe Assistant responds only with {options_str}\nAssistant:\n"
                        prompt = prompt.replace("\nYour choice (A, B, C, or D): ", "")
                    #print(f"prompt={prompt}")
                    #with self.client.generate(prompt, max_new_tokens=2, temperature=0, remote=True) as tracer:
                    #    out = self.client.generator.output.save()
                    #resp = self.client.tokenizer.decode(out[0][len(self.client.tokenizer(prompt)['input_ids']):]).strip().upper()[0]
                    with self.client.trace(prompt, remote=True):
                        output = self.client.output.save()
                    probs = torch.nn.functional.softmax(output["logits"][0,-1,:],dim=-1)
                    values,indices=torch.torch.topk(probs,k=len(options))
                    tokens = [self.client.tokenizer.decode(i) for i in indices]
                    token_probs = dict(sorted(zip(tokens,values.tolist())))
                    print(f"tokens[0]={tokens[0]}, token_probs={token_probs}")
                    return tokens[0], token_probs
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
                            **({"system_instruction": system_msg} if system_msg != "" else {}),
                            max_output_tokens=(None if "2.5" in self.subject_name else MAX_TOKENS),
                            temperature=temp + attempt * temp_inc,
                            candidate_count=1,
                            **({"response_logprobs": True} if '1.5' not in self.subject_name else {}),
                            **({"logprobs": len(options)} if '1.5' not in self.subject_name else {})
                        ), 
                    )
                    if '1.5' in self.subject_name: return message.text.strip(), None
                    cand = message.candidates[0]
                    resp = cand.content.parts[0].text.strip()
                    logres = cand.logprobs_result  
                    if len(options) == 1:                   # short answer – average over all tokens
                        # chosen_candidates = one entry per generated token
                        top_probs = [c.log_probability for c in logres.chosen_candidates]
                        token_probs = {resp: math.exp(sum(top_probs) / len(top_probs))}

                    else:                                   # multiple-choice – inspect 1st token only
                        # top_candidates[0].candidates = k alternatives for the 1st token
                        first_step = logres.top_candidates[0].candidates
                        tokens = [alt.token for alt in first_step]
                        probs  = [math.exp(alt.log_probability) for alt in first_step]
                        token_probs = dict(zip(tokens, probs))
                    return resp, token_probs
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            try:
                resp, token_probs = self._call_with_timeout(model_call, timeout=150)
            except TimeoutError:
                self._log(f"Timeout on attempt {callctr+1}, retrying…")
                attempt += 1
                continue
            except Exception as e:
                attempt += 1
                self._log(f"Error in llm processing: {e}")
                if "429" in str(e) or "503" in str(e) or "not returned" in str(e) or "[Errno 8]" in str(e):
                    # Rate limit error, wait and retry
                    time.sleep(delay)
                    delay = min(delay*2,15)
                    attempt -= 1 #don't increase temperature
                continue
            if accept_any or resp.upper() in options or options == " ":
                if token_probs: print(token_probs)
                break
            attempt += 1
            print(f"Bad LLM response: {resp} (attempt {attempt + 1})")
            if attempt == MAX_ATTEMPTS: break

        if keep_appending: message_history.append({"role": "assistant", "content": resp})
        if resp.upper() not in options and options != " ":
            self._log(f"Failed to get valid response for text: {q_text}; response: {resp}")
        return resp, message_history, token_probs

    def _get_subject_answer(self, options, prompt):
        """Gets the human subject's response."""
        if options: opts_msg = f", ".join(options[:-1]) + f", or {options[-1]}.\n"
        while True:
            try:
                answer = input(prompt).strip().upper()
                if options:
                    if answer in options:
                        return answer
                    else:
                        print(f"Invalid input. Please enter {opts_msg}.")
                else: return answer
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
        if "options" in question_data:
            formatted_question += "-" * 10 + "\n"        
            for key, value in question_data["options"].items():
                formatted_question += f"  {key}: {value}\n"
        
        formatted_question += "-" * 30
        return formatted_question
