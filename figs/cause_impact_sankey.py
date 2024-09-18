import json
import random
import plotly.graph_objects as go


def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return f'rgba({r}, {g}, {b}, {alpha})'


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
    'C2': 'Useless\nStatements\n<br>(executed\nwithout effect)',
    'C3': 'Fragmented\nLogics',
    'C4': 'Inconsistent\nLibraries',
    'C5': 'Useless\nStatements\n<br>(unexecuted)',
    'K1': 'Common\nSense',
    'K2': 'Mathematics\n&<br>Natural\nScience',
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
    'A1': 'Incorrect Functionality',
    'A2': 'Low Execution Efficiency',
    'A3': 'Extra Memory Footprint',
    'A4': 'Poor Readability',
    'A5': 'Poor Maintainability<br>and Scalability'
}


def desc2type(__desc, mp):
    for __k, __v in mp.items():
        if __v == __desc:
            return __k


models = ['DeepSeek-Coder-1.3b', 'DeepSeek-Coder-7b', 'CodeLlama-7b', 'GPT-4']
excluded_desc = {'Non-functional\nConflicting', 'Common\nSense', 'Inconsistent\nLibraries'}

fta_desc = [factor2desc[k] for k in sorted(factor2desc.keys())] + \
           [type2desc[k] for k in sorted(type2desc.keys())] + \
           [affection2desc[k] for k in sorted(affection2desc.keys())]
fta_desc = [desc for desc in fta_desc if desc not in excluded_desc]

factors_cnt = {x: [] for x in types}
types_cnt = {x: [0, 0, 0, 0] for x in types}
affections_cnt = {x: [] for x in types}
code_cnt = 0
label_cnt = 0
factor_affection = {x: [] for x in types}
random.seed(42)

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
                    index = random.randint(0, len(code_label['factors']) - 1)
                    factor = code_label['factors'][index]
                    if factor not in factors:
                        continue
                    index = random.randint(0, len(code_label['affections']) - 1)
                    affection = code_label['affections'][index]
                    if affection not in affections:
                        continue
                    if type2desc[tp] in excluded_desc:
                        continue
                    if tp == 'C2' and affection == 'A1':
                        continue
                    factor_affection[tp].append((factor, affection))
                    label_cnt += 1
                if len(code_labels) > 0:
                    code_cnt += 1


factors_color = {
    'P1': '#99c2ff',
    'P2': '#4d94ff',
    'P3': '#0052cc',
    'P4': '#003d99',
    'M1': '#ff934d',
    'M2': '#e65c00'
}
factors_color = {factor2desc[k]: v for k, v in factors_color.items()}

affections_color = {
    'A1': '#ff3333',
    'A2': '#ff661a',
    'A3': '#cc33ff',
    'A4': '#3385ff',
    'A5': '#00ffff'
}
affections_color = {affection2desc[k]: v for k, v in affections_color.items()}

type_colors = {
    'R1': '#ffff00',
    'C1': '#aaff80',
    'C2': '#77ff33',
    'C3': '#4ce600',
    'C5': '#339900',
    'K2': '#99ffff',
    'K31': '#33ffff',
    'K32': '#00e5e6',
    'K33': '#009899'
}
type_colors = {type2desc[k]: v for k, v in type_colors.items()}

colors = {}
colors.update(factors_color)
colors.update(type_colors)
colors.update(affections_color)


link_cnt = {}
for t, lst in factor_affection.items():
    for f, a in lst:
        f_idx = fta_desc.index(factor2desc[f])
        t_idx = fta_desc.index(type2desc[t])
        a_idx = fta_desc.index(affection2desc[a])
        if (f_idx, t_idx) not in link_cnt:
            link_cnt[(f_idx, t_idx)] = 0
        if (t_idx, a_idx) not in link_cnt:
            link_cnt[(t_idx, a_idx)] = 0
        link_cnt[(f_idx, t_idx)] += 1
        link_cnt[(t_idx, a_idx)] += 1


source = []
target = []
value = []

for (s, t), v in link_cnt.items():
    source.append(s)
    target.append(t)
    value.append(v)


nodes = dict(
    pad=24,
    thickness=15,
    line=dict(color="rgba(0,0,0,0)", width=0),
    label=fta_desc,
    color=[colors[desc] for desc in fta_desc]
)

links = dict(
    source=source,
    target=target,
    value=value,
    color=[hex_to_rgba(colors[fta_desc[s]], alpha=0.2) for s in source]
)

annotations = [
    dict(
        x=-0.03,
        y=-0.15,
        text="Causes",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14)
    ),
    dict(
        x=0.51,
        y=-0.15,
        text="Code Hallucinations",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14)
    ),
    dict(
        x=1.04,
        y=-0.15,
        text="Impacts",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14)
    )
]

fig = go.Figure(data=[go.Sankey(
    node=nodes,
    link=links
)])

fig.update_layout(font_size=10, annotations=annotations)
# fig.show()
# fig.write_image('cause_type_impact.svg', format='svg')
