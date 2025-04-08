import json
import random
import matplotlib.pyplot as plt
import numpy as np

from constants import datasets2desc, type2desc, excluded_desc
from tools import iter_data


def map_desc(_desc):
    if _desc == 'Mathematics & Natural Science':
        return 'Mathematics &\nNatural Science'
    if _desc == 'Useless Statements (executed without effect)':
        return 'Useless Statements\n(executed without effect)'
    if _desc == 'Useless Statements (unexecuted)':
        return 'Useless Statements\n(unexecuted)'
    return _desc


datasets = sorted(list(datasets2desc.keys()))

types_cnt = {desc: [0, 0, 0] for desc in type2desc.values() if desc not in excluded_desc}

for item in iter_data():
    for code_label in item['code_labels']:
        tp = code_label['hallucination-type']
        if tp in excluded_desc:
            continue
        types_cnt[tp][datasets.index(item['dataset'])] += 1

sums = [sum(v[i] for v in types_cnt.values()) for i in range(3)]
print(sums)

angles = np.linspace(0, 2 * np.pi, len(types_cnt), endpoint=False).tolist()
angles += angles[:1]

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(16, 12), subplot_kw=dict(polar=True))

types_desc = ['Behavior Conflicting', 'Algorithm', 'Undefined Variables', 'Fragmented Logics',
              'Data Conflicting', 'Mathematics & Natural Science', 'Computer Theory', 'Library/Project',
              'Useless Statements (executed without effect)', 'Useless Statements (unexecuted)']

colors = ['orange', 'green', 'blue']

print(list(zip(datasets, colors)))
for i in range(3):
    values = [types_cnt[k][i] / sum(types_cnt[k]) * 100 for k in types_desc]
    values += values[:1]
    ax.fill(angles, values, color=colors[i], alpha=0.25, label=f'{datasets2desc[datasets[i]]} ({sums[i]})')
    ax.plot(angles, values, color=colors[i], linewidth=1.5)

types_desc = [map_desc(desc) for desc in types_desc]
ax.set_xticks(angles[:-1])
ax.set_xticklabels(types_desc)
plt.xlabel('Proportion (%)', fontsize=40, labelpad=25)
ax.tick_params(axis='x', which='major', labelsize=26)
ax.tick_params(axis='y', which='major', labelsize=20)

ax.legend(loc='upper right', fontsize=28, bbox_to_anchor=(1.45, 1.1))
plt.tight_layout()
# plt.show()
plt.savefig('dataset_detail.pdf', format='pdf')
