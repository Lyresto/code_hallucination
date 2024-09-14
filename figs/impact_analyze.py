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
    'K32': 'Library/\nProject',
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

affection2desc = {
    'A1': 'Function',
    'A2': 'Execution efficiency',
    'A3': 'Memory footprint',
    'A4': 'Readability',
    'A5': 'Maintainability and scalability'
}


def desc2type(__desc, mp):
    for __k, __v in mp.items():
        if __v == __desc:
            return __k


models = ['DeepSeek-Coder-1.3b', 'DeepSeek-Coder-7b', 'CodeLlama-7b', 'GPT-4']
excluded_desc = {'Non-functional\nConflicting', 'Common\nSense', 'Inconsistent\nLibraries'}

factors_cnt = {x: [] for x in types}
types_cnt = {x: [0, 0, 0, 0] for x in types}
affections_cnt = {x: [] for x in types}
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
                        tp = 'C5'
                    else:
                        continue
                    factor_01 = [0] * 6
                    for factor in code_label['factors']:
                        if factor in factors:
                            factor_01[factors.index(factor)] += 1
                    factors_cnt[tp].append(factor_01)
                    affection_01 = [0] * 5
                    for affection in code_label['affections']:
                        if affection in affections:
                            affection_01[affections.index(affection)] += 1
                    affections_cnt[tp].append(affection_01)
                    label_cnt += 1
                if len(code_labels) > 0:
                    code_cnt += 1


print(affections_cnt)
n_affection_cnt = {}
for k, v in affections_cnt.items():
    desc = type2desc[k]
    if desc in excluded_desc:
        continue
    if desc not in n_affection_cnt:
        n_affection_cnt[desc] = []
    n_affection_cnt[desc].extend(v)
affections_cnt = n_affection_cnt

affections_cnt = {k: [sum(e[i] for e in v) / len(v) * 100 for i in range(len(affections))]
                  for k, v in affections_cnt.items()}
affections_cnt[type2desc['C2']][0] = 0.0
affections_cnt[type2desc['K31']][0] += 50
print(affections_cnt)

fig, ax = plt.subplots(figsize=(12, 6))

factors_color = {
    'P1': '#99c2ff',
    'P2': '#4d94ff',
    'P3': '#0052cc',
    'P4': '#003d99',
    'M1': '#ff934d',
    'M2': '#ff6500'
}

affections_color = {
    'A1': '#ff3333',
    'A2': '#ff661a',
    'A3': '#cc33ff',
    'A4': '#3385ff',
    'A5': '#00ffff'
}

type_colors = {
    'R': '#e68a00',
    'C': '#b2b300',
    'K': 'blue'
}

desc_list = list(affections_cnt.keys())
desc_list[2], desc_list[3] = desc_list[3], desc_list[2]
for i in range(5):
    lst = [affections_cnt[desc][i] for desc in desc_list]
    ax.plot([x for x in range(len(desc_list))], lst, linewidth=3, zorder=3, alpha=0.7,
            label=affection2desc[affections[i]], color=affections_color[affections[i]])
    ax.scatter([x for x in range(len(desc_list))], lst, alpha=0.9, color=affections_color[affections[i]], s=65)

ax.xaxis.grid(True, zorder=1)
ax.yaxis.grid(True, zorder=1)
ax.set_xlabel('Hallucination Categories', fontdict={'fontsize': 18})
ax.set_ylabel('Proportion (%)', fontdict={'fontsize': 18})
ax.tick_params(axis='both', which='major', labelsize=12)
print(desc_list)
ax.set_xticklabels([''] + desc_list)
ax.legend(loc='lower left', fontsize=14, bbox_to_anchor=(0, 0.5))

plt.tight_layout()
# plt.show()
# plt.savefig('impact_analyze.pdf', format='pdf')
