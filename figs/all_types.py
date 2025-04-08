import json
import random
import matplotlib.pyplot as plt

from constants import type2desc, models_formal_name, models_simple_name
from tools import iter_data

desc2complete_desc = {desc: f"({['I', 'II', 'III']['RCK'.index(tp[0])]}) {desc}" for tp, desc in type2desc.items()}
types_cnt = {v: [0, 0, 0, 0, 0] for v in desc2complete_desc.values()}

for item in iter_data():
    for code_label in item['code_labels']:
        desc = code_label['hallucination-type']
        types_cnt[desc2complete_desc[desc]][models_simple_name.index(item['model'])] += 1


sums = [sum(cnt[i] for cnt in types_cnt.values()) for i in range(5)]
print(sums)

colors = ['#EC7063', '#A569BD', '#5DADE2', '#45B39D', '#58D68D']

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(16, 6))
for i0, (desc, value) in enumerate(sorted(types_cnt.items(), key=lambda x: sum(x[1]))):
    left = 0
    for i, (v, c) in enumerate(zip(value, colors)):
        v = v / sums[i] * 100
        ax.barh(desc, v, left=left, color=c, alpha=0.7, zorder=3,
                label=f'{models_formal_name[i]} ({sums[i]})' if desc == list(desc2complete_desc.values())[0] else '')
        left += v
    ax.text(left + 5.0, i0, str(sum(value)), ha='center', va='center', fontsize=20, color='black')

plt.xlabel('LLM Internal Frequency (%)', fontsize=25, labelpad=25)
plt.ylabel('Hallucination Categories', fontsize=25, labelpad=25)
ax.tick_params(axis='x', which='major', labelsize=20)
ax.tick_params(axis='y', which='major', labelsize=18)
ax.xaxis.grid(True, zorder=1)
ax.legend(loc='lower right', fontsize=20)

plt.tight_layout()
# plt.show()
plt.savefig('all_hallu_types.pdf', format='pdf')
