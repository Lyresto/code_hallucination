import json
import os
from tqdm import tqdm
from client import Client

if __name__ == '__main__':
    for llm in ['deepseek-coder-1.3b', 'deepseek-coder-7b', 'codellama-7b', 'gpt-4']:
        if 'gpt' not in llm:
            continue
        llm_client = Client(llm)
        for idx, dataset in enumerate(['CEJavaRaw', 'CEPythonRaw', 'HumanEval']):
            bar = tqdm(desc=f'{llm}/{dataset}', total=[239, 230, 164][idx])
            if os.path.exists(f'result/{llm}_{dataset}_result.jsonl'):
                with open(f'result/{llm}_{dataset}_result.jsonl') as f:
                    result = f.read()
            else:
                result = ""
            with open(f'data/{dataset}.jsonl') as f:
                for line in f:
                    data_item = json.loads(line)
                    if dataset.startswith('CE'):
                        item_id_key = "question_id"
                        prompt_key = "input"
                    else:
                        item_id_key = "task_id"
                        prompt_key = "prompt"
                    if f'"{data_item[item_id_key]}"' not in result:
                        start_prompt = \
                            "Please try to directly implement this function according to its description:\n\n"
                        response = llm_client.generate(start_prompt + data_item[prompt_key])
                        result += json.dumps({item_id_key: data_item[item_id_key], "generation": response}) + "\n"
                        with open(f'result/{llm}_{dataset}_result.jsonl', 'w+') as f2:
                            f2.write(result)
                    bar.update(1)
            bar.close()


