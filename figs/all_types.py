import json
import random
import matplotlib.pyplot as plt


def load_json(path):
    with open(path) as __f:
        return json.load(__f)


def load_jsonl(path) -> dict[str, dict]:
    with open(path) as f:
        length = len(f.read().split('\n'))
    with open(path) as f:
        data = dict()
        for i, line in enumerate(f):
            line = json.loads(line)
            data[f'{line["_id"]}-{i // (length // 4)}'] = line
        return data


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
    'R1': 'Functional Conflicting',
    'R21': 'Non-functional Conflicting',
    'R22': 'Non-functional Conflicting',
    'R23': 'Non-functional Conflicting',
    'C1': 'Undefined Variables',
    'C2': 'Useless Statements (executed without effect)',
    'C3': 'Fragmented Logics',
    'C4': 'Inconsistent Libraries',
    'C5': 'Useless Statements (unexecuted)',
    'K1': 'Common Sense',
    'K2': 'Mathematics & Natural Science',
    'K31': 'Algorithm',
    'K32': 'Library/Project',
    'K33': 'Computer Theory'
}


def desc2type(__desc):
    for __k, __v in type2desc.items():
        if __v == __desc:
            return __k


models = ['DeepSeek-Coder-1.3B', 'DeepSeek-Coder-7B', 'CodeLlama-7B', 'GPT-4']

factors_cnt = {x: 0 for x in factors}
types_cnt = {x: [0, 0, 0, 0] for x in types}
affections_cnt = {x: 0 for x in affections}
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
            for inner_index, code_labels in sorted(user_labels['code'].items()):
                for code_label in code_labels:
                    tp = code_label['hallucination-type']
                    if tp in types_cnt:
                        types_cnt[tp][int(inner_index)] += 1
                    elif tp == 'other' and '无法' in code_label['hallucination-type-other']:
                        types_cnt['C5'][int(inner_index)] += 1
                    else:
                        continue
                    for factor in code_label['factors']:
                        if factor in factors_cnt:
                            factors_cnt[factor] += 1
                    for affection in code_label['affections']:
                        if affection in affections_cnt:
                            affections_cnt[affection] += 1
                    label_cnt += 1
                if len(code_labels) > 0:
                    code_cnt += 1

n_type_cnt = {}
for k, v in types_cnt.items():
    desc = type2desc[k]
    desc = '(' + ['I', 'II', 'III'][['R', 'C', 'K'].index(k[0])] + ') ' + desc
    if desc not in n_type_cnt:
        n_type_cnt[desc] = [0, 0, 0, 0]
    n_type_cnt[desc] = [e1 + e2 for e1, e2 in zip(n_type_cnt[desc], v)]
types_cnt = n_type_cnt

sums = [sum(v[i] for v in types_cnt.values()) for i in range(4)]
print(sums)

colors = ['#EE6666', '#73C0DE', '#3BA272', '#FC8452']
type_colors = {
    'R': '#e68a00',
    'C': '#b2b300',
    'K': 'blue'
}
fig, ax = plt.subplots(figsize=(14, 6))
for i0, (desc, value) in enumerate(sorted(types_cnt.items(), key=lambda x: sum(x[1]))):
    left = 0
    for i, (v, c) in enumerate(zip(value, colors)):
        v = v / sums[i] * 100
        ax.barh(desc, v, left=left, color=c, alpha=0.9, edgecolor='grey', zorder=3,
                label=f'{models[i]} ({sums[i]})' if desc == '(III) Library/Project' else '')
        left += v
    ax.text(left + 5.0, i0, sum(value), ha='center', va='center', fontsize=12, color='black')

ax.set_xlabel('Frequency (%)', fontdict={'fontsize': 18})
ax.set_ylabel('Hallucination Categories', fontdict={'fontsize': 18})
ax.tick_params(axis='x', which='major', labelsize=12)
ax.tick_params(axis='y', which='major', labelsize=14)
ax.xaxis.grid(True, zorder=1)
ax.legend(loc='lower right', fontsize=16)

plt.tight_layout()
# plt.show()
# plt.savefig('all_hallu_types.pdf', format='pdf')
