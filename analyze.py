import json
import re
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MultiLabelBinarizer

from constants import data_map, label_map, schedule, type2desc, types, factors, affections


def calculate_consistency(data1, data2):
    if len(data1) != len(data2):
        raise ValueError("两个数据集的长度必须相同")

    correlations = []
    for list1, list2 in zip(data1, data2):
        correlation = []
        for line1, line2 in zip(list1, list2):
            correlation.append(cohen_kappa_score(line1, line2))
        correlations.append(np.mean(correlation))
        # correlations.append(cohen_kappa_score(list1, list2))

    return np.mean(correlations)


# source_data_map = {
#     'CEJava': load_ce_json('data/CEJavaRaw.json')
# }


models2pass = {'all': []}
for dataset, dataset_data in data_map.items():
    for task_id, line in dataset_data.items():
        eval_res = line['evaluation_result']
        if dataset == 'HumanEval':
            passed = eval_res == 'passed'
        else:
            passed = eval_res['is_pass']
        model = line['model']
        if model not in models2pass:
            models2pass[model] = []
        models2pass[model].append(passed)
        models2pass['all'].append(passed)


# models2pass = {k: sum(v) / len(v) * 100 for k, v in models2pass.items()}
# print(models2pass)
# exit(-1)

# id2idx = {}
# idx2id = {}
# for data_set in ['CEJava', 'CEPython', 'HumanEval']:
#     result = load_jsonl(f'result/{data_set}.jsonl')
#     if data_set not in idx2id:
#         idx2id[data_set] = {}
#     for item in result:
#         id2idx[item['_id']] = item['index_within_model']
#         idx2id[data_set][item['index_within_model']] = item['_id']


# extra_types = set()
# for dataset, dataset_labels in label_map.items():
#     for task_id, task_labels in dataset_labels.items():
#         for user, user_labels in task_labels.items():
#             prompt_label = user_labels['prompt']
#             for inner_index, code_labels in user_labels['code'].items():
#                 for code_label in code_labels:
#                     for key in ['hallucination-type-other', 'factors', 'affections']:
#                         try:
#                             types = code_label[key]
#                         except KeyError:
#                             continue
#                         types = [types] if isinstance(types, str) else types
#                         for tp in types:
#                             if not re.match(r'[A-Z]\d+', tp):
#                                 extra_types.add((user, key, tp))
#
# for extra_type in extra_types:
#     print(extra_type)


user2types = {}
user_pair = set()

for dataset, dataset_labels in label_map.items():
    for task_id, task_labels in dataset_labels.items():
        user_pair.add(tuple(sorted(task_labels.keys())))
        for user, user_labels in task_labels.items():
            if user not in user2types:
                user2types[user] = []
            prompt_label = user_labels['prompt']
            for inner_index, code_labels in sorted(user_labels['code'].items()):
                raw_code = data_map[dataset][f'{task_id}-{inner_index}']['raw_generation']
                label_list = []
                for _ in range(len(raw_code.split('\n'))):
                    # label_list.append([1] + [0] * len(types))
                    label_list.append(0)
                for code_label in code_labels:
                    tp = code_label['hallucination-type']
                    # tps = code_label['affections']
                    # for tp in tps:
                    if tp in types:
                        tp = types.index(tp) + 1
                        for lineno in range(int(code_label['start-lineno']) - 1, int(code_label['end-lineno'])):
                            # label_list[lineno][0] = 0
                            # label_list[lineno][tp] = 1
                            label_list[lineno] = tp
                user2types[user].append(np.array(label_list))


for pair in user_pair:
    print(pair, len(user2types[pair[0]]), len(user2types[pair[1]]))
    consistency = calculate_consistency(user2types[pair[0]], user2types[pair[1]])
    print(pair, consistency)


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
                        # if tp.startswith('R2'):
                        #     print(tp)
                        data = data_map[dataset][f'{task_id}-{inner_index}']
                        if True:  # tp.startswith('K2') and dataset.startswith('H') and user != 'wuhaotian':
                            # context = source_data_map[dataset][task_id]['']
                            ref = data['reference_solution']
                            if 'aeiou' in ref.lower():
                                print(data)
                            # if 'import' in ref:
                            #     libs = re.findall(r'import (.*?)[\n\s]', ref)
                            #     for lib in libs:
                            #         if ref.count(lib) >= 3:
                            #             print(data)
                            print(data)
                    elif tp == 'other' and '无法' in code_label['hallucination-type-other']:
                        types_cnt['C5'][int(inner_index)] += 1
                        # data = data_map[dataset][f'{task_id}-{inner_index}']
                        # print(data)
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

types_cnt = {type2desc[k]: v for k, v in types_cnt.items()}
sums = [sum(v[i] for v in types_cnt.values()) for i in range(4)]
print(sums)

colors = ['red', 'green', 'blue', 'orange']
fig, ax = plt.subplots(figsize=(14, 6))
for desc, value in sorted(types_cnt.items(), key=lambda x: sum(x[1])):
    left = 0
    for i, (v, c) in enumerate(zip(value, colors)):
        v = v / sums[i] * 100
        ax.barh(desc, v, left=left, color=c, alpha=0.6, edgecolor='grey', zorder=3)
        left += v

ax.set_xlabel('Frequency (%)', fontdict={'fontsize': 14})
ax.set_ylabel('Hallucination Categories', fontdict={'fontsize': 14})
ax.tick_params(axis='both', which='major', labelsize=12)
ax.xaxis.grid(True, zorder=1)

plt.tight_layout()
plt.show()


# print(label_cnt)
# label_cnt = sum(map(lambda x: x[1], types_cnt.items()))
# print(code_cnt)
# print(label_cnt)
# print(factors_cnt)
# factors_cnt = {x[0]: round(x[1] / label_cnt * 100, 4) for x in factors_cnt.items()}
# for l in range(1, 4):
#     st = set(map(lambda x: x[:l], factors_cnt.keys()))
#     mp = {k: 0.0 for k in st}
#     for k, v in factors_cnt.items():
#         mp[k[:l]] += v
#     print(mp)
# print(types_cnt)
# types_cnt = {x[0]: round(x[1] / label_cnt * 100, 4) for x in types_cnt.items()}
#
# print(affections_cnt)
# print({x[0]: f'{round(x[1] / label_cnt * 100, 4)}%' for x in affections_cnt.items()})
