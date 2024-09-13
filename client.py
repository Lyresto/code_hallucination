import threading
import time
import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

openai.api_key = ''  # Your OpenAI Key


class Client:
    def __init__(self, model):
        self.lock = threading.Lock()

        if model == 'deepseek-coder-1.3b':
            path = './models/deepseek-coder-1.3b/'
        elif model == 'deepseek-coder-7b':
            path = './models/deepseek-coder-7b'
        elif model == 'codellama-7b':
            path = './models/codellama-7b'
        elif 'gpt' in model:
            path = None
        else:
            raise NotImplementedError()
        if path is None:
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if '7b' in model:
            max_memory = {0: "10000MiB", 1: "20000MiB", 2: "20000MiB", 3: "20000MiB"}
            self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, device_map='auto',
                                                              max_memory=max_memory)
        elif 'gpt' in model:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).cuda()

    def generate(self, msg, **kwargs):
        self.lock.acquire(timeout=120)
        time.sleep(1)

        if isinstance(msg, str):
            msg = [
                {'role': 'user', 'content': msg}
            ]
        if self.tokenizer is None:
            max_call = 20
            while True:
                try:
                    response = openai.ChatCompletion.create(model=self.model, messages=msg, temperature=0.0, top_p=0.95)
                    break
                except Exception as e:
                    print('[ERROR]', e)
                    print("[ERROR] fail to call gpt, trying again...")
                    max_call -= 1
                    if max_call == 0:
                        raise RuntimeError()
                    time.sleep(2)
            return response['choices'][0]['message']['content']
        inputs = self.tokenizer.apply_chat_template(
            msg, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(self.model.device)
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else 512
        temperature = kwargs['temperature'] if 'temperature' in kwargs else 0.0
        do_sample = temperature > 0.0
        args = {'temperature': temperature, 'top_p': 0.95} if temperature > 0.0 else {}
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=50,
                                      num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id,
                                      pad_token_id=self.tokenizer.eos_token_id, attention_mask=attention_mask, **args)
        result = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        self.lock.release()
        return result
