import threading
import time
from http.client import HTTPException

import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

deepseek_client = OpenAI(
    api_key="<KEY>",
    base_url="https://api.deepseek.com"
)

openai_client = OpenAI(
    api_key="<KEY>",
    base_url="https://platform.openai.com"
)


class Client:
    tokenizer = None
    model = None
    model_name = None

    def __init__(self, model_name):
        self.lock = threading.Lock()
        self.messages = []
        if Client.model is not None and Client.model_name == model_name:
            return
        Client.model_name = model_name
        if model_name == 'deepseek-coder-1.3b':
            path = 'path/to/model'
        elif model_name == 'deepseek-coder-7b':
            path = 'path/to/model'
        elif model_name == 'codellama-7b':
            path = 'path/to/model'
        elif model_name in ['gpt-4', 'deepseek-r1']:
            path = None
            if model_name == 'deepseek-r1':
                model_name = 'deepseek-reasoner'
        else:
            raise NotImplementedError()
        if path is not None:
            Client.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if '7b' in model_name:
            max_memory = {0: "20000MiB", 1: "20000MiB"}
            Client.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, device_map='auto',
                                                                max_memory=max_memory)
        elif model_name in ['gpt-4', 'deepseek-r1']:
            Client.model = model_name
        else:
            Client.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).cuda()

    def generate(self, user_request, **kwargs):
        if isinstance(user_request, str):
            self.messages.append({'role': 'user', 'content': user_request})
        else:
            self.messages = user_request
        if self.tokenizer is None:
            if self.model == 'gpt-4':
                client = openai_client
            elif self.model == 'deepseek-reasoner':
                client = deepseek_client
            else:
                raise NotImplementedError()
            max_call = 20
            while True:
                try:
                    completion = client.chat.completions.create(
                        model=self.model,
                        temperature=0.0,
                        messages=self.messages,
                    )
                    break
                except HTTPException:
                    max_call -= 1
                    if max_call == 0:
                        raise RuntimeError('max retries exceeded')
                    time.sleep(2)
            return (
                completion.choices[0].message.content,
                completion.choices[0].message.model_extra['reasoning_content']
                if self.model == 'deepseek-reasoner' else None
            )

        inputs = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(self.model.device)
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else 512
        temperature = kwargs['temperature'] if 'temperature' in kwargs else 0.0
        do_sample = temperature > 0.0
        args = {'temperature': temperature, 'top_p': 0.95} if temperature > 0.0 else {}
        self.lock.acquire(timeout=120)
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=50,
                                      num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id,
                                      pad_token_id=self.tokenizer.eos_token_id, attention_mask=attention_mask, **args)
        self.lock.release()
        result = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        self.messages.append({'role': 'assistant', 'content': result})
        return result, None
