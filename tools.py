import json
import os
import re
import subprocess
import tokenize
from io import BytesIO
import random

import tiktoken
from tqdm import tqdm

from client import Client
from constants import datasets, models_simple_name, type2desc, factor2desc, affection2desc, text_complexity_mapper


def remove_unused_task_in_ce_java():
    with open('data/CEJavaRaw.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    used_task_ids = set()
    for task in data['RECORDS']:
        used_task_ids.add(task['_id'])
    filtered_tasks = []
    with open('data/CEJavaRaw.jsonl', 'r') as f:
        for line in f:
            parsed_line = json.loads(line)
            if parsed_line['question_id'] in used_task_ids:
                filtered_tasks.append(line)
    with open('data/CEJavaRaw.jsonl', 'w') as f:
        f.write(''.join(filtered_tasks))
    print(len(filtered_tasks))


def load_json(path, encoding='utf-8'):
    with open(os.path.abspath(path), encoding=encoding) as __f:
        return json.load(__f)


def load_ce_json(path):
    with open(os.path.abspath(path)) as __f:
        __data = json.load(__f)
    __res = {}
    for item in __data['RECORDS']:
        __res[item['_id']] = item
    return __res


# context = load_ce_json('data/CEJavaRaw.json')['636767581a6d9265ec017fb4']['file_content']
# print(context)
# exit(-1)


def load_jsonl_with_all_models(path) -> dict[str, dict]:
    with open(os.path.abspath(path)) as f:
        length = len(f.read().split('\n'))
    with open(os.path.abspath(path)) as f:
        data = dict()
        for i, line in enumerate(f):
            line = json.loads(line)
            data[f'{line["_id"]}-{i // (length // len(models_simple_name))}'] = line
        return data


def load_jsonl_with_single_model(path) -> list[dict]:
    with open(os.path.abspath(path)) as __f:
        data = []
        for line in __f:
            data.append(json.loads(line))
        return data


def jsonl_to_dict(data, key) -> dict[str, dict]:
    new_data = dict()
    for line in data:
        new_data[line[key]] = line
    return new_data


def parse_tokens(__code: str):
    try:
        return list(tokenize.tokenize(BytesIO(__code.encode('utf-8')).readline))[1:]
    except Exception as e:
        if len(__code.split('\n')) == 1:
            print(f'{e}')
        return parse_tokens('\n'.join(__code.split('\n')[:-1]))


def extract_function(__content, __func_name):
    tokens = parse_tokens(__content)
    start = [(0, 0)]
    end = [(0, 0)]
    indents = 0
    find_func = False
    for i, token in enumerate(tokens[1:]):
        if token.type == tokenize.NAME and token.string == __func_name and tokens[i].string == 'def':
            find_func = True
            start.append(tokens[i].start)
        elif find_func and token.type == tokenize.INDENT:
            indents += 4
        elif find_func and token.type == tokenize.DEDENT:
            indents -= 4
            if indents == 0:
                end.append(token.start)
                find_func = False
    lines = __content.split('\n')[start[len(end) - 1][0] - 1: end[-1][0] - 1]

    return '\n'.join(lines[1:])


def remove_space(s):
    return ''.join(s.strip().split(' '))


def extract_completed_code(_raw_code, _info, _extract_imports=True):
    func_sign_prefix = _info["func_sign_prefix"]
    filtered_lines = []
    spaces = 0
    for line in _raw_code.split('\n')[::-1]:
        filtered_lines.append(line)
        if remove_space(func_sign_prefix) in remove_space(line) and '`' not in line:
            spaces = (len(line) - len(line.lstrip())) * (1 if line[0] == ' ' else 4)
            func_sign_mark = ':' if _info['language'] == 'python' else '{'
            while not filtered_lines[-1].strip().endswith(func_sign_mark):
                del filtered_lines[-1]
            del filtered_lines[-1]
            break

    filtered_lines = filtered_lines[::-1]
    filtered_lines = list(map(lambda li: li[spaces:], filtered_lines))
    if _info["language"] == 'python':
        code = _info["func_sign"] + '\n' + '\n'.join(filtered_lines)
        import_lines = list(map(lambda _li: ' ' * 4 + _li,
                                filter(lambda _li:
                                       (_li.startswith('import ') or _li.startswith('from ')) and 'typing ' not in _li,
                                       _raw_code.split('\n'))
                                ))
        # import_lines = []
        func_content = extract_function(code, _info["entrypoint"])
        return ('\n'.join(import_lines) + '\n' + func_content).strip('\n')
    else:
        left_braces = 1
        right_braces = 0
        final_lines = []
        for line in filtered_lines:
            final_lines.append(line)
            left_braces += line.count('{')
            right_braces += line.count('}')
            if left_braces == right_braces:
                break
        return '\n'.join(final_lines)


def list2chat(lst):
    chat = ''
    for i, (user_msg, bot_msg) in enumerate(lst):
        if i == 0:
            sys_msg = '<<SYS>>\n{you are a coding assistant.}\n<</SYS>>'
        else:
            sys_msg = ''
        chat += f'<s>[INST]{sys_msg}{{{user_msg}}}[/INST]'
        if i != len(lst) - 1:
            chat += f'{{{bot_msg}}}</s>'
    return chat


def extract_result(mode):
    if len(mode) > 0:
        mode = f'_{mode}'
    for dataset in datasets[2::]:
        for llm in tqdm(models_simple_name[:1:]):
            print(f'extracting {llm}/{dataset}')
            base_dir = f'result/{llm}/{dataset}{mode}'
            if dataset.startswith('CE'):
                item_id_key = 'question_id'
            else:
                item_id_key = 'task_id'
            if 'Java' in dataset:
                language = 'java'
            else:
                language = 'python'
            extracted_result = ""
            if os.path.exists(f'{base_dir}/generation_filtered.jsonl'):
                raw_data = load_jsonl_with_single_model(f'{base_dir}/generation_filtered.jsonl')
            else:
                raw_data = load_jsonl_with_single_model(f'{base_dir}/generation.jsonl')
            source_data = jsonl_to_dict(load_jsonl_with_single_model(f'data/{dataset}.jsonl'), item_id_key)
            for item in tqdm(raw_data):
                task_id = item[item_id_key]
                if dataset.startswith('CE'):
                    sign_prefix = source_data[task_id]['signature']
                    start = max(sign_prefix.find('def '), sign_prefix.find('public '),
                                sign_prefix.find('protected '), sign_prefix.find('private '), 0)
                    sign_prefix = sign_prefix[start:].split('(')[0].strip()
                    for prefix in ['public', 'protected', 'private', 'final', 'static']:
                        sign_prefix = sign_prefix.removeprefix(prefix).strip()
                    entrypoint = sign_prefix.split(' ')[-1]
                else:
                    entrypoint = source_data[task_id]['entry_point']
                    sign_prefix = f'def {entrypoint}'
                # extracted_generation = extract_completed_code(item['generation'], {
                #     'language': language,
                #     'func_sign_prefix': sign_prefix + '(',
                #     'entrypoint': entrypoint,
                #     'func_sign': f'{sign_prefix}():'
                # })
                extracted_generation = max(re.findall(r'```.*?\n(.*?)```', item['generation'], re.DOTALL),
                                           key=lambda c: len(c))
                if len(extracted_generation) == 0:
                    print('no code detected!!!')
                if dataset.startswith('CE'):
                    if 'Java' in dataset:
                        # import_lines = list(
                        #     filter(lambda l: l.startswith('import ') or l.startswith('from '),
                        #            item['generation'].split('\n')))
                        # print(import_lines)
                        import_lines = []
                    else:
                        import_lines = []
                    extracted_result += json.dumps({
                        '_id': task_id,
                        'generate_results': [
                            '\n'.join(import_lines +
                                      ['\n', source_data[task_id]['signature'], extracted_generation]).strip()
                        ]
                    }) + '\n'
                else:
                    extracted_result += json.dumps({
                        'task_id': task_id,
                        'generation': extracted_generation
                    }) + '\n'
            with open(f'{base_dir}/generation_extracted.jsonl', 'w+') as f:
                f.write(extracted_result)


def merge_result(mode=''):
    if len(mode) > 0:
        mode = f'_{mode}'
    for dataset in datasets[2::]:
        merged_results = ''
        for llm in tqdm(models_simple_name):
            print(f'merging {llm}/{dataset} for mode: {mode}')
            base_dir = f'./result/{llm}/{dataset}{mode}'
            if not os.path.exists(base_dir):
                print('no such dir, skip')
                continue
            if dataset.startswith('CE'):
                item_id_key = 'question_id'
                results = jsonl_to_dict(load_jsonl_with_single_model(f'{base_dir}/generation_results.jsonl'), '_id')
                ref_data = jsonl_to_dict(load_json(f'data/{dataset}.json')['RECORDS'], '_id')
            else:
                item_id_key = 'task_id'
                results = load_json(f'{base_dir}/generation_extracted.eval_results.json')['eval']
                ref_data = dict()
            if os.path.exists(f'{base_dir}/generation_filtered.jsonl'):
                raw_generation = load_jsonl_with_single_model(f'{base_dir}/generation_filtered.jsonl')
            else:
                raw_generation = load_jsonl_with_single_model(f'{base_dir}/generation.jsonl')
            raw_generation = jsonl_to_dict(raw_generation, item_id_key)
            generation = load_jsonl_with_single_model(f'{base_dir}/generation_extracted.jsonl')
            source_data = jsonl_to_dict(load_jsonl_with_single_model(f'data/{dataset}.jsonl'), item_id_key)
            random.seed(42)
            for index, item in enumerate(generation):
                _id = item[item_id_key] if item_id_key in item else item['_id']
                common_dict = {
                    "_id": _id,
                    "model": llm,
                    "index_within_model": index,
                    "raw_generation": raw_generation[_id]['generation'],
                    "reasoning_content": raw_generation[_id]['reasoning_content']
                    if 'reasoning_content' in raw_generation[_id] else None,
                }
                if dataset.startswith('CE'):
                    merged_results += json.dumps({
                        **common_dict,
                        "prompt": source_data[_id]['input'],
                        "generation": item['generate_results'][0],
                        "evaluation_result": results[_id]['generate_results'][0],
                        "reference_solution": ref_data[_id]['code'],
                        "human_label": ref_data[_id]['human_label'],
                        "level": ref_data[_id]['level'],
                        "project": ref_data[_id]['project']
                    }) + '\n'
                else:
                    merged_results += json.dumps({
                        **common_dict,
                        "prompt": source_data[_id]['prompt'],
                        "generation": item['generation'],
                        "evaluation_result": {
                            "status": results[_id][0]['plus_status'],
                            "example_fail_tests": results[_id][0]['plus_fail_tests'].replace(' = ', ' == '),
                            "base_status": results[_id][0]['base_status'],
                        },
                        "reference_solution": source_data[_id]['canonical_solution']
                    }) + '\n'
        with open(f'result/{dataset.removesuffix("Raw")}{mode}.jsonl', 'w+') as f:
            f.write(merged_results)


def calculate_passk():
    for llm in models_simple_name:
        passk_by_model = []
        for dataset in datasets:
            base_dir = f'result/{llm}/{dataset}'
            if dataset.startswith('CE'):
                results = jsonl_to_dict(load_jsonl_with_single_model(f'{base_dir}/generation_results.jsonl'), '_id')
            else:
                results = load_json(f'{base_dir}/generation_extracted.eval_results.json')['eval']
            for result in results.values():
                if dataset.startswith('CE'):
                    passk_by_model.append(result['generate_results'][0]['is_pass'])
                else:
                    passk_by_model.append(result[0]['plus_status'] == 'pass')
        print(f'{llm}: {round(sum(passk_by_model) / len(passk_by_model) * 100, 2)}')


def get_prompt_length(prompt):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt))


def merge_results_with_labels():
    all_data = []
    length_cache = {}
    types_cnt = {v: 0 for v in type2desc.values()}
    bar = tqdm(total=(164 + 230 + 230) * 5)
    comp3 = set()
    for dataset, dataset_labels in label_map.items():
        for task_id, task_labels in dataset_labels.items():
            for user, user_labels in task_labels.items():
                if user not in schedule or 'prefer' not in schedule[user]:
                    continue
                prompt_label = user_labels['prompt']
                prompt_label['logic-complexity'] = text_complexity_mapper[prompt_label['logic-complexity']]
                if dataset == 'HumanEval' and prompt_label['logic-complexity'] >= 2:
                    comp3.add(task_id)
                for inner_index, code_labels in sorted(user_labels['code'].items()):
                    if int(inner_index) >= 5:
                        continue
                    mapped_labels = []
                    for code_label in code_labels:
                        tp = code_label['hallucination-type']
                        if tp == 'R1':
                            if len(re.findall(r'\W\d+\W', f" {code_label['conflict-content']} ")) == 0:
                                tp = 'R11'
                            else:
                                tp = 'R12'
                        if tp == 'other' and '无法' in code_label['hallucination-type-other']:
                            tp = 'C5'
                        if tp in type2desc:
                            code_label['hallucination-type'] = type2desc[tp]
                            types_cnt[type2desc[tp]] += 1
                        else:
                            continue
                        mapped_factors = set()
                        for factor in code_label['factors']:
                            if factor in factor2desc:
                                mapped_factors.add(factor2desc[factor])
                        code_label['factors'] = list(mapped_factors)
                        mapped_affections = set()
                        for affection in code_label['affections']:
                            if affection in affection2desc:
                                if tp == 'C2' and affection == 'A1':
                                    continue
                                mapped_affections.add(affection2desc[affection])
                        if len(mapped_affections) == 0:
                            mapped_affections.add(affection2desc['A2'])
                        code_label['affections'] = list(mapped_affections)
                        mapped_labels.append(code_label)
                    task_model_data = data_map[dataset][f'{task_id}-{inner_index}']
                    task_model_data['dataset'] = dataset
                    assert 'labeler' not in task_model_data
                    task_model_data['labeler'] = user
                    if f'{dataset}-{task_id}' not in length_cache:
                        length_cache[f'{dataset}-{task_id}'] = get_prompt_length(task_model_data['prompt'])
                    task_model_data['prompt_length'] = length_cache[f'{dataset}-{task_id}']
                    task_model_data['prompt_label'] = prompt_label
                    task_model_data['code_labels'] = mapped_labels
                    all_data.append(json.dumps(task_model_data))
                    bar.update(1)
    # print(json.dumps(types_cnt, indent=4))
    # print(comp3)
    with open('./result/merged.jsonl', 'w+') as f:
        f.write('\n'.join(all_data))


def iter_data():
    with open(f'{__file__}/../result/merged.jsonl', 'r+') as f:
        for line in f:
            yield json.loads(line)


label_map = {
    'CEJava': load_json(f'{__file__}/../label_result/CEJava.json'),
    'CEPython': load_json(f'{__file__}/../label_result/CEPython.json'),
    'HumanEval': load_json(f'{__file__}/../label_result/HumanEval.json')
}

data_map = {
    'CEJava': load_jsonl_with_all_models(f'{__file__}/../result/CEJava.jsonl'),
    'CEPython': load_jsonl_with_all_models(f'{__file__}/../result/CEPython.jsonl'),
    'HumanEval': load_jsonl_with_all_models(f'{__file__}/../result/HumanEval.jsonl')
}

schedule = load_json(f'{__file__}/../label_result/schedule.json')


# if __name__ == '__main__':
#     # llm_client = Client('deepseek-coder-1.3b')
#     llm_client = Client('deepseek-coder-7b')
#     conversation = []
#     while True:
#         req = ''
#         while True:
#             ipt = input()
#             if ipt == '|END':
#                 break
#             req += ipt + '\n'
#         conversation.append({'role': 'user', 'content': req})
#         # conversation.append((req, ''))
#         # chat = list2chat(conversation)
#         resp = llm_client.generate(conversation)
#         print(resp)
#         conversation.append({'role': 'assistant', 'content': resp})
#     # llm_client = Client('gpt-3.5-turbo')
#     # print(llm_client.generate(input()))
#     # a = parse_tokens('def p()\n    return 1\n```\nefocunehqo\nf3f')
#     # language = 'python'


if __name__ == '__main__':
    # remove_unused_task_in_ce_java()
    # extract_result('cot2')
    # merge_result()
    # calculate_passk()
    merge_results_with_labels()
