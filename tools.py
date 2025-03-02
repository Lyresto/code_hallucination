import json
import os
import tokenize
from io import BytesIO

from tqdm import tqdm

from client import Client
from constants import datasets, models_simple_name


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


def load_json(path):
    with open(path) as __f:
        return json.load(__f)


def load_ce_json(path):
    with open(path) as __f:
        __data = json.load(__f)
    __res = {}
    for item in __data['RECORDS']:
        __res[item['_id']] = item
    return __res


# context = load_ce_json('data/CEJavaRaw.json')['636767581a6d9265ec017fb4']['file_content']
# print(context)
# exit(-1)


def load_jsonl_with_all_models(path) -> dict[str, dict]:
    with open(path) as f:
        length = len(f.read().split('\n'))
    with open(path) as f:
        data = dict()
        for i, line in enumerate(f):
            line = json.loads(line)
            data[f'{line["_id"]}-{i // (length // len(models_simple_name))}'] = line
        return data


def load_jsonl_with_single_model(path) -> list[dict]:
    with open(path) as __f:
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


def extract_function(__content, __func_name, keep_intact=False):
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
    if keep_intact:
        return '\n'.join(lines)
    # lines = list(filter(lambda l: not l.strip().startswith('print('), lines))
    import_lines = list(filter(lambda l: l.startswith('import ') or l.startswith('from '), __content.split('\n')))
    return '\n'.join(import_lines + lines)


def remove_space(s):
    return ''.join(s.strip().split(' '))


def extract_completed_code(_raw_code, _info, _language):
    func_sign_prefix = _info["func_sign_prefix"]
    filtered_lines = []
    spaces = 0
    for line in _raw_code.split('\n')[::-1]:
        filtered_lines.append(line)
        if remove_space(func_sign_prefix) in remove_space(line) and '`' not in line:
            spaces = (len(line) - len(line.lstrip())) * (1 if line[0] == ' ' else 4)
            func_sign_mark = ':' if _language == 'python' else '{'
            while not filtered_lines[-1].strip().endswith(func_sign_mark):
                del filtered_lines[-1]
            del filtered_lines[-1]
            break

    filtered_lines = filtered_lines[::-1]
    filtered_lines = list(map(lambda li: li[spaces:], filtered_lines))
    if _info["language"] == 'python':
        code = _info["func_sign"] + '\n' + '\n'.join(filtered_lines)
        completed_code_with_func_sign = extract_function(code, _info["entrypoint"], True)
        return '\n'.join(completed_code_with_func_sign.split('\n')[1:])
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


def extract_result():
    for idx, dataset in enumerate(datasets):
        for llm in tqdm(models_simple_name):
            base_dir = f'result/{llm}/{dataset}'
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
                extracted_generation = extract_completed_code(item['generation'], {
                    'language': language,
                    'func_sign_prefix': sign_prefix + '(',
                    'entrypoint': entrypoint,
                    'func_sign': f'{sign_prefix}():'
                })
                if len(extracted_generation) == 0:
                    print(11111)
                if dataset.startswith('CE'):
                    import_lines = list(
                        filter(lambda l: l.startswith('import ') or l.startswith('from '),
                               item['generation'].split('\n')))
                    print(import_lines)
                    extracted_result += json.dumps({
                        '_id': task_id,
                        'generate_results': [
                            '\n'.join(import_lines + ['\n', source_data[task_id]['signature'], extracted_generation])]
                    }) + '\n'
                else:
                    extracted_result += json.dumps({
                        'task_id': task_id,
                        'generation': extracted_generation
                    }) + '\n'
            with open(f'{base_dir}/generation_extracted.jsonl', 'w+') as f:
                f.write(extracted_result)


def merge_result():
    for idx, dataset in enumerate(datasets):
        merged_results = ""
        for llm in tqdm(models_simple_name):
            base_dir = f'result/{llm}/{dataset}'
            if dataset.startswith('CE'):
                item_id_key = 'question_id'
                results = jsonl_to_dict(load_jsonl_with_single_model(f'{base_dir}/generation_results.jsonl'), '_id')
                with open(f'data/{dataset}.json', encoding='utf-8') as f:
                    ref_data = jsonl_to_dict(json.load(f)['RECORDS'], '_id')
            else:
                item_id_key = 'task_id'
                results = load_json(f'{base_dir}/generation_extracted_python_result.json')
                ref_data = dict()
            if os.path.exists(f'{base_dir}/generation_filtered.jsonl'):
                raw_generation = load_jsonl_with_single_model(f'{base_dir}/generation_filtered.jsonl')
            else:
                raw_generation = load_jsonl_with_single_model(f'{base_dir}/generation.jsonl')
            raw_generation = jsonl_to_dict(raw_generation, item_id_key)
            generation = load_jsonl_with_single_model(f'{base_dir}/generation_extracted.jsonl')
            source_data = jsonl_to_dict(load_jsonl_with_single_model(f'data/{dataset}.jsonl'), item_id_key)
            for index, item in enumerate(generation):
                _id = item[item_id_key] if item_id_key in item else item['_id']
                common_dict = {
                    "_id": _id,
                    "model": llm,
                    "index_within_model": index
                }
                if dataset.startswith('CE'):
                    merged_results += json.dumps({
                        **common_dict,
                        "prompt": source_data[_id]['input'],
                        "generation": item['generate_results'][0],
                        "evaluation_result": results[_id]['generate_results'][0],
                        "raw_generation": raw_generation[_id]['generation'],
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
                        "evaluation_result": results[_id][0],
                        "raw_generation": raw_generation[_id]['generation'],
                        "reference_solution": source_data[_id]['canonical_solution']
                    }) + '\n'
        with open(f'result/{dataset.removesuffix("Raw")}.jsonl', 'w+') as f:
            f.write(merged_results)


if __name__ == '__main__':
    # llm_client = Client('deepseek-coder-1.3b')
    llm_client = Client('deepseek-coder-7b')
    conversation = []
    while True:
        req = ''
        while True:
            ipt = input()
            if ipt == '|END':
                break
            req += ipt + '\n'
        conversation.append({'role': 'user', 'content': req})
        # conversation.append((req, ''))
        # chat = list2chat(conversation)
        resp = llm_client.generate(conversation)
        print(resp)
        conversation.append({'role': 'assistant', 'content': resp})
    # llm_client = Client('gpt-3.5-turbo')
    # print(llm_client.generate(input()))
    # a = parse_tokens('def p()\n    return 1\n```\nefocunehqo\nf3f')
    # language = 'python'


if __name__ == '__main__':
    remove_unused_task_in_ce_java()
