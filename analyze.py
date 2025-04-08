import json
import re
import random
import copy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import statsmodels.formula.api as smf

from constants import desc2top_type, type2desc, models_simple_name, datasets, datasets2desc
from tools import iter_data, load_json, label_map, schedule


#
# from constants import data_map, label_map, schedule, type2desc, types, factors, affections
#
#
# def calculate_consistency(data1, data2):
#     if len(data1) != len(data2):
#         raise ValueError("两个数据集的长度必须相同")
#
#     correlations = []
#     for list1, list2 in zip(data1, data2):
#         correlation = []
#         for line1, line2 in zip(list1, list2):
#             correlation.append(cohen_kappa_score(line1, line2))
#         correlations.append(np.mean(correlation))
#         # correlations.append(cohen_kappa_score(list1, list2))
#
#     return np.mean(correlations)
#
#
# # source_data_map = {
# #     'CEJava': load_ce_json('data/CEJavaRaw.json')
# # }
#
#
# models2pass = {'all': []}
# for dataset, dataset_data in data_map.items():
#     for task_id, line in dataset_data.items():
#         eval_res = line['evaluation_result']
#         if dataset == 'HumanEval':
#             passed = eval_res == 'passed'
#         else:
#             passed = eval_res['is_pass']
#         model = line['model']
#         if model not in models2pass:
#             models2pass[model] = []
#         models2pass[model].append(passed)
#         models2pass['all'].append(passed)
#
#
# # models2pass = {k: sum(v) / len(v) * 100 for k, v in models2pass.items()}
# # print(models2pass)
# # exit(-1)
#
# # id2idx = {}
# # idx2id = {}
# # for data_set in ['CEJava', 'CEPython', 'HumanEval']:
# #     result = load_jsonl(f'result/{data_set}.jsonl')
# #     if data_set not in idx2id:
# #         idx2id[data_set] = {}
# #     for item in result:
# #         id2idx[item['_id']] = item['index_within_model']
# #         idx2id[data_set][item['index_within_model']] = item['_id']
#
#
# # extra_types = set()
# # for dataset, dataset_labels in label_map.items():
# #     for task_id, task_labels in dataset_labels.items():
# #         for user, user_labels in task_labels.items():
# #             prompt_label = user_labels['prompt']
# #             for inner_index, code_labels in user_labels['code'].items():
# #                 for code_label in code_labels:
# #                     for key in ['hallucination-type-other', 'factors', 'affections']:
# #                         try:
# #                             types = code_label[key]
# #                         except KeyError:
# #                             continue
# #                         types = [types] if isinstance(types, str) else types
# #                         for tp in types:
# #                             if not re.match(r'[A-Z]\d+', tp):
# #                                 extra_types.add((user, key, tp))
# #
# # for extra_type in extra_types:
# #     print(extra_type)
#
#
# user2types = {}
# user_pair = set()
#
# for dataset, dataset_labels in label_map.items():
#     for task_id, task_labels in dataset_labels.items():
#         user_pair.add(tuple(sorted(task_labels.keys())))
#         for user, user_labels in task_labels.items():
#             if user not in user2types:
#                 user2types[user] = []
#             prompt_label = user_labels['prompt']
#             for inner_index, code_labels in sorted(user_labels['code'].items()):
#                 raw_code = data_map[dataset][f'{task_id}-{inner_index}']['raw_generation']
#                 label_list = []
#                 for _ in range(len(raw_code.split('\n'))):
#                     # label_list.append([1] + [0] * len(types))
#                     label_list.append(0)
#                 for code_label in code_labels:
#                     tp = code_label['hallucination-type']
#                     # tps = code_label['affections']
#                     # for tp in tps:
#                     if tp in types:
#                         tp = types.index(tp) + 1
#                         for lineno in range(int(code_label['start-lineno']) - 1, int(code_label['end-lineno'])):
#                             # label_list[lineno][0] = 0
#                             # label_list[lineno][tp] = 1
#                             label_list[lineno] = tp
#                 user2types[user].append(np.array(label_list))
#
#
# for pair in user_pair:
#     print(pair, len(user2types[pair[0]]), len(user2types[pair[1]]))
#     consistency = calculate_consistency(user2types[pair[0]], user2types[pair[1]])
#     print(pair, consistency)
#
#
# factors_cnt = {x: 0 for x in factors}
# types_cnt = {x: [0, 0, 0, 0] for x in types}
# affections_cnt = {x: 0 for x in affections}
# code_cnt = 0
# label_cnt = 0
#
# for dataset, dataset_labels in label_map.items():
#     items = list(dataset_labels.items())
#     random.shuffle(items)
#     for task_id, task_labels in items:
#         for user, user_labels in task_labels.items():
#             if user not in schedule or 'prefer' not in schedule[user]:
#                 continue
#             prompt_label = user_labels['prompt']
#             for inner_index, code_labels in sorted(user_labels['code'].items()):
#                 for code_label in code_labels:
#                     tp = code_label['hallucination-type']
#                     if tp in types_cnt:
#                         types_cnt[tp][int(inner_index)] += 1
#                         # if tp.startswith('R2'):
#                         #     print(tp)
#                         data = data_map[dataset][f'{task_id}-{inner_index}']
#                         if True:  # tp.startswith('K2') and dataset.startswith('H') and user != 'wuhaotian':
#                             # context = source_data_map[dataset][task_id]['']
#                             ref = data['reference_solution']
#                             if 'aeiou' in ref.lower():
#                                 print(data)
#                             # if 'import' in ref:
#                             #     libs = re.findall(r'import (.*?)[\n\s]', ref)
#                             #     for lib in libs:
#                             #         if ref.count(lib) >= 3:
#                             #             print(data)
#                             print(data)
#                     elif tp == 'other' and '无法' in code_label['hallucination-type-other']:
#                         types_cnt['C5'][int(inner_index)] += 1
#                         # data = data_map[dataset][f'{task_id}-{inner_index}']
#                         # print(data)
#                     else:
#                         continue
#                     for factor in code_label['factors']:
#                         if factor in factors_cnt:
#                             factors_cnt[factor] += 1
#                     for affection in code_label['affections']:
#                         if affection in affections_cnt:
#                             affections_cnt[affection] += 1
#                     label_cnt += 1
#                 if len(code_labels) > 0:
#                     code_cnt += 1
#
# types_cnt = {type2desc[k]: v for k, v in types_cnt.items()}
# sums = [sum(v[i] for v in types_cnt.values()) for i in range(4)]
# print(sums)
#
# colors = ['red', 'green', 'blue', 'orange']
# fig, ax = plt.subplots(figsize=(14, 6))
# for desc, value in sorted(types_cnt.items(), key=lambda x: sum(x[1])):
#     left = 0
#     for i, (v, c) in enumerate(zip(value, colors)):
#         v = v / sums[i] * 100
#         ax.barh(desc, v, left=left, color=c, alpha=0.6, edgecolor='grey', zorder=3)
#         left += v
#
# ax.set_xlabel('Frequency (%)', fontdict={'fontsize': 14})
# ax.set_ylabel('Hallucination Categories', fontdict={'fontsize': 14})
# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.xaxis.grid(True, zorder=1)
#
# plt.tight_layout()
# plt.show()
#
#
# # print(label_cnt)
# # label_cnt = sum(map(lambda x: x[1], types_cnt.items()))
# # print(code_cnt)
# # print(label_cnt)
# # print(factors_cnt)
# # factors_cnt = {x[0]: round(x[1] / label_cnt * 100, 4) for x in factors_cnt.items()}
# # for l in range(1, 4):
# #     st = set(map(lambda x: x[:l], factors_cnt.keys()))
# #     mp = {k: 0.0 for k in st}
# #     for k, v in factors_cnt.items():
# #         mp[k[:l]] += v
# #     print(mp)
# # print(types_cnt)
# # types_cnt = {x[0]: round(x[1] / label_cnt * 100, 4) for x in types_cnt.items()}
# #
# # print(affections_cnt)
# # print({x[0]: f'{round(x[1] / label_cnt * 100, 4)}%' for x in affections_cnt.items()})


def mediation_effect_test():
    lengths = {
        1: [],
        2: [],
        3: []
    }
    all_lengths = copy.deepcopy(lengths)

    for item in iter_data():
        all_lengths[item['prompt_label']['logic-complexity']].append(item['prompt_length'])
        if len(item['code_labels']) > 0:
            lengths[item['prompt_label']['logic-complexity']].append(item['prompt_length'])

    bins = 80
    length_range = (min(*all_lengths[1], *all_lengths[2], *all_lengths[3]),
                    max(*all_lengths[1], *all_lengths[2], *all_lengths[3]))
    print(length_range)
    length_data = []
    complexity_data = []
    proportion_data = []

    for complexity, all_length in all_lengths.items():
        length = lengths[complexity]
        count_all, length_edges = np.histogram(all_length, bins=bins, range=length_range)
        count, _ = np.histogram(length, bins=bins, range=length_range)
        length_avg = (length_edges[:-1] + length_edges[1:]) / 2
        proportion = np.divide(count, count_all, out=np.full_like(count, -1, dtype=np.float64), where=count != 0)
        for le, p in zip(length_avg, proportion):
            if p == -1:
                continue
            length_data.append(le)
            complexity_data.append(complexity)
            proportion_data.append(p * 100)

    df = pd.DataFrame({
        'length': length_data,  # 自变量 X
        'complexity': complexity_data,  # 中介变量 M
        'proportion': proportion_data  # 因变量 Y
    })
    # print(df)

    # **第一步：X → Y（总效应 c）**
    model1 = smf.ols('proportion ~ length', data=df).fit()
    c = model1.params['length']
    p1 = model1.pvalues['length']
    print(model1.summary())

    # **第二步：X → M（路径 a）**
    model2 = smf.ols('complexity ~ length', data=df).fit()
    a = model2.params['length']
    p2 = model2.pvalues['length']
    print(model2.summary())

    # **第三步：X, M → Y（路径 b, c'）**
    model3 = smf.ols('proportion ~ length + complexity', data=df).fit()
    b = model3.params['complexity']
    c_prime = model3.params['length']
    p3 = model3.pvalues['length']
    p4 = model3.pvalues['complexity']
    print(model3.summary())

    # **第四步：M → Y **
    model4 = smf.ols('proportion ~ complexity', data=df).fit()
    p5 = model4.pvalues['complexity']
    print(model4.summary())

    # **Bootstrap 计算间接效应 (a * b) 的置信区间**
    def bootstrap_mediation(data, n_bootstrap=5000):
        _indirect_effects = []
        np.random.seed(42)

        for _ in range(n_bootstrap):
            sample_data = data.sample(n=len(data), replace=True)  # 重新抽样
            # 拟合新的回归
            model_M_bs = smf.ols('complexity ~ length', data=sample_data).fit()
            model_Y_bs = smf.ols('proportion ~ length + complexity', data=sample_data).fit()
            # 计算新的 a 和 b
            a_bs = model_M_bs.params['length']
            b_bs = model_Y_bs.params['complexity']
            _indirect_effect = a_bs * b_bs
            _indirect_effects.append(_indirect_effect)

        # 计算 95% 置信区间
        lower_bound = np.percentile(_indirect_effects, 2.5)
        upper_bound = np.percentile(_indirect_effects, 97.5)
        return _indirect_effects, lower_bound, upper_bound

    # 计算 Bootstrap 中介效应
    indirect_effects, lower_ci, upper_ci = bootstrap_mediation(df)

    # **结果展示**
    print(f"\n总效应 (c): {c}")
    print(f"直接效应 (c'): {c_prime}")
    print(f"间接效应 (a * b): {a * b}")
    print(f"p值: X -> Y (X) = {p1}, X -> M (X) = {p2}, X + M -> Y (X) = {p3}, X + M -> Y (M) = {p4}, M -> Y (M) = {p5}")
    print(f"Bootstrap 95% CI for a*b: ({lower_ci}, {upper_ci})")


def calculate_proportion():
    types_cnt = defaultdict(int)
    top_types_cnt = defaultdict(int)
    factor_cnt = defaultdict(int)
    affection_cnt = defaultdict(int)
    dataset_cnt = defaultdict(int)
    model_cnt = defaultdict(int)
    all_hallucination_cnt = 0
    code_cnt = 0
    hallucination_code_cnt = 0
    for item in iter_data():
        code_cnt += 1
        if len(item['code_labels']) > 0:
            hallucination_code_cnt += 1
        for label in item['code_labels']:
            all_hallucination_cnt += 1
            dataset_cnt[item['dataset']] += 1
            model_cnt[item['model']] += 1
            types_cnt[label['hallucination-type']] += 1
            top_types_cnt[desc2top_type[label['hallucination-type']]] += 1
            for factor in label['factors']:
                factor_cnt[factor] += 1
            for affection in label['affections']:
                affection_cnt[affection] += 1
    for cnt_dict in [types_cnt, top_types_cnt, factor_cnt, affection_cnt]:
        for desc, count in cnt_dict.items():
            print(f"{desc}: {count / all_hallucination_cnt * 100:.2f}%")
    for cnt_dict in [dataset_cnt, model_cnt]:
        for desc, count in cnt_dict.items():
            print(f"{desc}: {count}")
    print(f'total {all_hallucination_cnt} hallucinations in {hallucination_code_cnt} codes, all {code_cnt} codes')


def passk_statistics():
    result_dict = defaultdict(list)
    for item in iter_data():
        eval_result = item['evaluation_result']
        if item['dataset'].startswith('CE'):
            passed = eval_result['is_pass']
        else:
            passed = eval_result['status'] == 'pass'
        for label in item['code_labels']:
            result_dict[label['hallucination-type']].append(passed)
        if len(item['code_labels']) == 0:
            result_dict['no hallucinations'].append(passed)
        else:
            result_dict['hallucinations'].append(passed)
    passk = {}
    for desc, result in result_dict.items():
        passk[desc] = sum(result) / len(result) * 100
    for desc, pk in sorted(passk.items(), key=lambda x: x[1]):
        print(f'{desc.split( )[0]} & ', end='')
    print()
    for desc, pk in sorted(passk.items(), key=lambda x: x[1]):
        print(f'{" ".join(desc.split( )[1:])} & ', end='')
    print()
    for desc, pk in sorted(passk.items(), key=lambda x: x[1]):
        print(f'{pk :.2f} & ', end='')


def hallucination_statistics(by='model'):
    def map_desc(_desc):
        if _desc == 'Mathematics & Natural Science':
            return r'Mathematics \&', 'Natural Science'
        if _desc == 'Useless Statements (executed without effect)':
            return 'Useless Statements', '(executed without effect)'
        if _desc == 'Useless Statements (unexecuted)':
            return 'Useless Statements', '(unexecuted)'
        return _desc, ''

    def wrap_multirow(content, rows=2):
        if isinstance(content, float):
            content = rf'{content:.2f}\%'
        if multi_row:
            return rf'\multirow{{{rows}}}{{*}}{{{content}}}'
        return content

    def wrap_bf(content):
        if isinstance(content, float):
            content = rf'{content:.2f}\%'
        return rf'\textbf{{{content}}}'

    if by == 'model':
        cols = models_simple_name
    else:
        cols = sorted(list(datasets2desc.keys()))
    print(cols)

    types_cnt = {v: [0] * len(cols) for v in type2desc.values()}
    by_hall_res = {c: [] for c in cols}

    for item in iter_data():
        by_hall_res[item[by]].append(len(item['code_labels']) == 0)
        for code_label in item['code_labels']:
            desc = code_label['hallucination-type']
            types_cnt[desc][cols.index(item[by])] += 1
    for col, res in by_hall_res.items():
        print(f'{col}: {res.count(True)} non-hallucination codes, {res.count(False)} hallucination codes')

    sums = [sum(cnt[i] for cnt in types_cnt.values()) for i in range(len(cols))]
    total = sum(sums)
    print(sums, total)

    for i0, (desc, value) in enumerate(list(sorted(types_cnt.items(), key=lambda x: -sum(x[1]))) + [('Total', sums)]):
        print(r'    \midrule')
        desc1, desc2 = map_desc(desc)
        multi_row = desc2 != ''
        print(f'    {desc1}', end='')
        for i, v in enumerate(value):
            print(rf' & {wrap_multirow(v)} & {wrap_multirow(v / total * 100)}', end='')
        if by == 'model':
            print(rf' & {wrap_multirow(wrap_bf(sum(value)))} & {wrap_multirow(wrap_bf(sum(value) / total * 100))} \\')
        else:
            print(r' \\')
        if multi_row:
            print(f'    {desc2}', end='')
            for _ in value:
                print(' & &', end='')
            if by == 'model':
                print(r' & & \\')
            else:
                print(r' \\')


def chi2_contingency():
    contingency_table = [[434, 190], [443, 181]]
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"p值: {p_value}")


def mitigate_analysis():
    mit_codes = []
    ref_codes = []
    mit_code_types = []
    ref_code_types = []
    for task_id, task_labels in label_map['CEPython'].items():
        for user, user_labels in task_labels.items():
            if 'prefer' not in schedule[user]:
                continue
            user_labels = user_labels['code']
            if '7' in user_labels:
                for codes, code_types, code_labels in [(ref_codes, ref_code_types, user_labels['0']),
                                                       (mit_codes, mit_code_types, user_labels['7'])]:
                    codes.append(len(code_labels) > 0)
                    for code_label in code_labels:
                        code_types.append(code_label['hallucination-type'])
    print(len(mit_codes))
    print('ref result:', f'& {sum(ref_codes)} & {round(sum(ref_codes) / len(ref_codes) * 100, 2)}\\% ')
    print('mit result:', f'& {sum(mit_codes)} & {round(sum(mit_codes) / len(mit_codes) * 100, 2)}\\% ')
    for desc, types in [('ref', ref_code_types), ('mit', mit_code_types)]:
        print(f'{desc}: ', end='')
        for key in ['R', 'C', 'K']:
            type_cnt = sum(map(lambda x: x.startswith(key), types))
            if key == 'C':
                type_cnt += sum(map(lambda x: x == 'other', types))
            print(f'& {type_cnt} & {round(type_cnt / len(types) * 100, 2)}\\% ', end='')
        print()


def cal_cause_for_each_llm():
    result = defaultdict(list)
    for item in iter_data():
        model = item['model']
        for code_label in item['code_labels']:
            result[model].append('Model-related Causes' in code_label['factors'])
    for model, lst in result.items():
        print(model, round(sum(lst) / len(lst) * 100, 2))


def difference_between_hm_and_ep():
    total_count = defaultdict(int)
    slight_count = defaultdict(int)
    for item in iter_data():
        if item['dataset'] != 'HumanEval':
            continue
        eval_res = item['evaluation_result']
        for code_label in item['code_labels']:
            tp = code_label['hallucination-type']
            if eval_res['status'] == 'fail':
                total_count[tp] += 1
                if eval_res['base_status'] == 'pass':
                    slight_count[tp] += 1
    total = sum(total_count.values())
    slight = sum(slight_count.values())
    for desc, count in total_count.items():
        s_count = slight_count[desc]
        print(f'{desc}: total: {round(count / total * 100, 2)}% ({count})\tslight: {round(s_count / slight * 100, 2)}% ({s_count})')


if __name__ == '__main__':
    # calculate_proportion()
    # passk_statistics()
    # hallucination_statistics('model')
    # chi2_contingency()
    # mitigate_analysis()
    # cal_cause_for_each_llm()
    difference_between_hm_and_ep()
