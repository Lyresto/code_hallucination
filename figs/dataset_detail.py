import json
import random
import matplotlib.pyplot as plt
import numpy as np

from constants import datasets2desc, factors, types, affections, label_map, schedule, type2desc, excluded_desc

datasets = sorted(list(datasets2desc.keys()))

factors_cnt = {x: 0 for x in factors}
types_cnt = {x: [0, 0, 0] for x in types}
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
                        types_cnt[tp][datasets.index(dataset)] += 1
                    elif tp == 'other' and '无法' in code_label['hallucination-type-other']:
                        types_cnt['C5'][datasets.index(dataset)] += 1
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

sums = [sum(v[i] for v in types_cnt.values()) for i in range(3)]
print(sums)

n_type_cnt = {}
for k, v in types_cnt.items():
    desc = type2desc[k]
    if desc in excluded_desc:
        continue
    if desc not in n_type_cnt:
        n_type_cnt[desc] = [0, 0, 0]
    n_type_cnt[desc] = [e1 + e2 for e1, e2 in zip(n_type_cnt[desc], v)]
types_cnt = n_type_cnt
print(types_cnt)

angles = np.linspace(0, 2 * np.pi, len(types_cnt), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

types_desc = ['Functional Conflicting', 'Algorithm', 'Undefined Variables', 'Fragmented Logics',
              'Mathematics &\nNatural Science', 'Computer Theory', 'Library/Project',
              'Useless Statements\n(executed without effect)', 'Useless Statements\n(unexecuted)']

colors = ['orange', 'green', 'blue']

print(list(zip(datasets, colors)))
for i in range(3):
    values = [types_cnt[k][i] / sum(types_cnt[k]) * 100 for k in types_desc]
    values += values[:1]
    ax.fill(angles, values, color=colors[i], alpha=0.25, label=f'{datasets2desc[datasets[i]]} ({sums[i]})')
    ax.plot(angles, values, color=colors[i], linewidth=1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(types_desc)
ax.set_xlabel('Proportion (%)', fontdict={'fontsize': 18})
ax.tick_params(axis='x', which='major', labelsize=12)

ax.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.5, 1.1))
plt.tight_layout()
# plt.show()
# plt.savefig('dataset_detail.pdf', format='pdf')
