import json
import random
import matplotlib.pyplot as plt
import tiktoken


def load_json(path):
    with open(path) as __f:
        return json.load(__f)


def load_jsonl(path) -> dict[str, dict]:
    with open(path) as f:
        _length = len(f.read().split('\n'))
    with open(path) as f:
        data = dict()
        for i, line in enumerate(f):
            line = json.loads(line)
            data[f'{line["_id"]}-{i // (_length // 4)}'] = line
        return data


def get_prompt_length(prompt_):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt_))


label_map = {
    'CEJava': load_json('../label_result/CEJava.json'),
    'CEPython': load_json('../label_result/CEPython.json'),
    'HumanEval': load_json('../label_result/HumanEval.json')
}

data_map = {
    'CEJava': load_jsonl('../result/CEJava.jsonl'),
    'CEPython': load_jsonl('../result/CEPython.jsonl'),
    'HumanEval': load_jsonl('../result/HumanEval.jsonl')
}

schedule = load_json('../label_result/schedule.json')


user2types = {}
user_pair = set()
factors = ['P1', 'P2', 'P3', 'P4', 'M1', 'M2']
types = ['R1', 'R21', 'R22', 'R23', 'K1', 'K2', 'K31', 'K32', 'K33', 'C1', 'C2', 'C3', 'C4', 'C5']
affections = ['A1', 'A2', 'A3', 'A4', 'A5']


type2desc = {
    'R1': 'Functional\nConflicting',
    'R21': 'Non-functional\nConflicting',
    'R22': 'Non-functional\nConflicting',
    'R23': 'Non-functional\nConflicting',
    'C1': 'Undefined\nVariables',
    'C2': 'Useless\nStatements\n(executed\nwithout effect)',
    'C3': 'Fragmented\nLogics',
    'C4': 'Inconsistent\nLibraries',
    'C5': 'Useless\nStatements\n(unexecuted)',
    'K1': 'Common\nSense',
    'K2': 'Mathematics\n& Natural\nScience',
    'K31': 'Algorithm',
    'K32': 'Library/Project',
    'K33': 'Computer\nTheory'
}

factor2desc = {
    'P1': 'Ambiguous',
    'P2': 'Incomplete',
    'P3': 'Overly complex',
    'P4': 'Lengthy',
    'M1': 'Poor Reasoning Ability',
    'M2': 'Lack of Relevant Knowledge'
}


def desc2type(__desc, mp):
    for __k, __v in mp.items():
        if __v == __desc:
            return __k


models = ['DeepSeek-Coder-1.3b', 'DeepSeek-Coder-7b', 'CodeLlama-7b', 'GPT-4']

factors_cnt = {x: [] for x in types}
types_cnt = {x: [0, 0, 0, 0] for x in types}
affections_cnt = {x: 0 for x in affections}
length = []
hallu_length = []
complexity = []
hallu_complexity = []
incomplete = [0, 0]
hallu_incomplete = [0, 0]
fuzzy = [0, 0]
hallu_fuzzy = [0, 0]
code_cnt = 0
label_cnt = 0

for dataset, dataset_labels in label_map.items():
    items = list(dataset_labels.items())
    random.shuffle(items)
    for task_id, task_labels in items:
        for user, user_labels in task_labels.items():
            if user not in schedule or 'prefer' not in schedule[user]:
                continue
            prompt_label = user_labels['prompt']
            prompt = data_map[dataset][f'{task_id}-0']['prompt']
            le = get_prompt_length(prompt)
            comp = int(prompt_label['logic-complexity'])
            fuz = int('fuzzy' in prompt_label)
            inc = int('incomplete' in prompt_label)
            for inner_index, code_labels in sorted(user_labels['code'].items()):
                if len(code_labels) > 0:
                    hallu_length.append(le)
                    hallu_complexity.append(comp)
                    hallu_fuzzy[fuz] += 1
                    hallu_incomplete[inc] += 1
                length.append(le)
                complexity.append(comp)
                fuzzy[fuz] += 1
                incomplete[inc] += 1
                for code_label in code_labels:
                    tp = code_label['hallucination-type']
                    if tp in types_cnt:
                        types_cnt[tp][int(inner_index)] += 1
                    elif tp == 'other' and '无法' in code_label['hallucination-type-other']:
                        types_cnt['C5'][int(inner_index)] += 1
                        tp = 'C5'
                    else:
                        continue
                    factor_01 = [0] * 6
                    for factor in code_label['factors']:
                        if factor in factors:
                            factor_01[factors.index(factor)] += 1
                    factors_cnt[tp].append(factor_01)
                    for affection in code_label['affections']:
                        if affection in affections_cnt:
                            affections_cnt[affection] += 1
                    label_cnt += 1
                if len(code_labels) > 0:
                    code_cnt += 1

"""
Length Analyze
"""
# bins = np.linspace(min(min(length), min(hallu_length)), max(max(length), max(hallu_length)), 21)
# hist_len, _ = np.histogram(length, bins)
# hist_hallu_len, _ = np.histogram(hallu_length, bins)
# print(hist_len)
# print(hist_hallu_len)
# ratios = hist_hallu_len / hist_len * 100
#
# plt.bar(bins[:-1], ratios, width=np.diff(bins), edgecolor='black', alpha=0.5, color='g', zorder=3)
#
#
# plt.grid(axis='y', zorder=1)
# plt.xlabel('Prompt Tokens (#)', fontdict={'fontsize': 14})
# plt.ylabel('Proportion (%)', fontdict={'fontsize': 14})

"""
Complexity Analyze
"""
# bins = np.linspace(1, 6, 6)
# print(bins)
# hist_comp, _ = np.histogram(complexity, bins)
# hist_hallu_comp, _ = np.histogram(hallu_complexity, bins)
# print(hist_comp)
# print(hist_hallu_comp)
# ratios = hist_hallu_comp / hist_comp * 100
# print(ratios)
# plt.bar(bins[:-1], ratios, width=0.3 * np.diff(bins), edgecolor='black', alpha=0.5, color='orange', zorder=3)
# plt.plot([1, 2, 3, 4, 5], ratios, marker='v', markersize=12, zorder=5)
#
# plt.grid(axis='y', zorder=1)
# plt.xlabel('Logic Complexity', fontdict={'fontsize': 14})
# plt.ylabel('Proportion (%)', fontdict={'fontsize': 14})


plt.tight_layout()

# plt.savefig('prompt_complexity.pdf', format='pdf')
# plt.show()
