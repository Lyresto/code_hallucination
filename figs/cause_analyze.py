import json
import random
import matplotlib.pyplot as plt

from constants import types, affections, label_map, schedule, factors, type2desc, excluded_desc, factor2desc

factors_cnt = {x: [] for x in types}
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


n_factor_cnt = {}
for k, v in factors_cnt.items():
    desc = type2desc[k]
    if desc in excluded_desc:
        continue
    if desc not in n_factor_cnt:
        n_factor_cnt[desc] = []
    n_factor_cnt[desc].extend(v)
factors_cnt = n_factor_cnt

factors_cnt = {k: [sum(e[i] for e in v) / len(v) * 100 for i in range(len(factors))] for k, v in factors_cnt.items()}
print(factors_cnt)

fig, ax = plt.subplots(figsize=(12, 6))

factors_color = {
    'P1': '#99c2ff',
    'P2': '#4d94ff',
    'P3': '#0052cc',
    'P4': '#003d99',
    'M1': '#ff934d',
    'M2': '#e65c00'
}

factors_marker = {
    'P1': '^',
    'P2': 's',
    'P3': 'p',
    'P4': 'x',
    'M1': '^',
    'M2': 's'
}

desc_list = list(factors_cnt.keys())
desc_list[2], desc_list[3] = desc_list[3], desc_list[2]
for i in range(6):
    lst = [factors_cnt[desc][i] for desc in desc_list]
    ax.plot([x for x in range(len(desc_list))], lst, linewidth=3, zorder=3, alpha=0.7, color=factors_color[factors[i]])
    ax.scatter([x for x in range(len(desc_list))], lst, alpha=0.9, marker=factors_marker[factors[i]],
               color=factors_color[factors[i]], s=65, label=factor2desc[factors[i]])

ax.xaxis.grid(True, zorder=1)
ax.yaxis.grid(True, zorder=1)
ax.set_xlabel('Hallucination Categories', fontdict={'fontsize': 18})
ax.set_ylabel('Proportion (%)', fontdict={'fontsize': 18})
ax.tick_params(axis='both', which='major', labelsize=12)
print(desc_list)
ax.set_xticklabels([''] + desc_list)
ax.legend(loc='lower right', fontsize=14, bbox_to_anchor=(1, 0.35))

plt.tight_layout()
# plt.show()
# plt.savefig('cause_analyze.pdf', format='pdf')
