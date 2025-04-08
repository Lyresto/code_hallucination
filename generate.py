import json
import os
import re
from threading import Thread

from tqdm import tqdm
from client import Client


def generate_for_dataset(llm, dataset, mode=''):
    print(f'{llm}/{dataset} start generating, mode = {mode}')
    if mode != '':
        _mode = f'_{mode}'
    else:
        _mode = ''
    result_path = f'./result/{llm}/{dataset}{_mode}/generation.jsonl'
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = f.read()
    else:
        result = ""
    with open(f'raw/{dataset}.jsonl') as f:
        for line in tqdm(f, desc=f'{llm}/{dataset}'):
            data_item = json.loads(line)
            if dataset.startswith('CE'):
                item_id_key = "question_id"
                prompt_key = "input"
            else:
                item_id_key = "task_id"
                prompt_key = "prompt"
            task_id = data_item[item_id_key]
            if f'"{task_id}"' not in result:
                start_prompt = end_prompt = ''
                if mode == '':
                    start_prompt = \
                        'Please try to directly implement this function according to its description:\n\n'
                elif mode == 'refine':
                    start_prompt = 'Please explain the requirements of this function in your own words:\n\n'
                elif mode == 'cot':
                    end_prompt = ('\n\nThis is a Python function that needs to be completed. '
                                  'Please provide your steps to solve it. '
                                  '\nLet\'s think step by step:\n1.')
                llm_client = Client(llm)
                response, thinking = llm_client.generate(f'{start_prompt}{data_item[prompt_key]}{end_prompt}')
                if mode == 'refine' or \
                        (mode == 'cot' and len(re.findall(r'```.*```', response, re.DOTALL)) == 0):
                    response, thinking = llm_client.generate('Now please implement this function.')
                result += json.dumps({
                    item_id_key: task_id,
                    "generation": response,
                    "reasoning_content": thinking,
                    "conversation": llm_client.messages
                }) + "\n"
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                with open(result_path, 'w+') as result_file:
                    result_file.write(result)


def main(mode=''):
    """
    :param mode: empty for baseline, `refine` for self-refine, `cot` for CoT
    :return:
    """
    for llm in ['deepseek-coder-1.3b', 'deepseek-coder-7b', 'codellama-7b', 'gpt-4', 'deepseek-r1']:
        threads = []
        for dataset in ['CEJavaRaw', 'CEPythonRaw', 'HumanEval']:
            thread = Thread(target=generate_for_dataset, args=(llm, dataset, mode))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    print('Done!')


if __name__ == '__main__':
    main()
