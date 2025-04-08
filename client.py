import threading
import time
import openai
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
            path = '/home/shared_models/workspace/deepseek-coder-1.3b-instruct/'
        elif model_name == 'deepseek-coder-7b':
            path = '/home/shared_models/workspace/deepseek-coder-7b-instruct-v1.5/'
        elif model_name == 'codellama-7b':
            path = './models/codellama-7b'
        elif 'gpt' in model_name or 'r1' in model_name:
            path = None
            if 'r1' in model_name:
                model_name = 'deepseek-reasoner'
        else:
            raise NotImplementedError()
        if path is not None:
            Client.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if '7b' in model_name:
            max_memory = {0: "20000MiB", 1: "20000MiB"}
            Client.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, device_map='auto',
                                                                max_memory=max_memory)
        elif 'gpt' in model_name or 'deepseek-reasoner' in model_name:
            Client.model = model_name
        else:
            Client.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).cuda()

    def generate(self, msg, **kwargs):
        # time.sleep(1)
        if isinstance(msg, str):
            self.messages.append({'role': 'user', 'content': msg})
        else:
            self.messages = msg
        if self.tokenizer is None:
            if 'gpt' in self.model:
                client = openai_client
            elif 'deepseek-reasoner' in self.model:
                client = deepseek_client
            else:
                raise NotImplementedError()
            max_call = 20
            while True:
                try:
                    completion = client.chat.completions.create(
                        model=self.model,
                        temperature=0.0,
                        top_p=0.95,
                        messages=self.messages,
                        # response_format={
                        #     'type': 'json_object'
                        # }
                    )
                    break
                except Exception as e:
                    print('[ERROR]', e)
                    print(f'[ERROR] fail to call {self.model}, trying again...')
                    max_call -= 1
                    if max_call == 0:
                        raise RuntimeError()
                    time.sleep(2)
            return (
                completion.choices[0].message.content,
                completion.choices[0].message.model_extra['reasoning_content']
                if 'deepseek-reasoner' in self.model else None
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
