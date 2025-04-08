import json
import random
import matplotlib.pyplot as plt

from constants import type2desc, factor2desc, factors_color, affection2desc, affections_color
from tools import iter_data


def map_desc(_desc):
    if _desc.count(' ') == 1:
        return _desc.replace(' ', '\n')
    if _desc.count('/') == 1:
        return _desc.replace('/', '/\n')
    if _desc == 'Mathematics & Natural Science':
        return 'Mathematics\n& Natural\nScience'
    if _desc == 'Useless Statements (executed without effect)':
        return 'Useless\nStatements\n(executed\nwithout effect)'
    if _desc == 'Useless Statements (unexecuted)':
        return 'Useless\nStatements\n(unexecuted)'
    return _desc


def main(target='factor'):
    if target == 'factor':
        target2desc = factor2desc
    else:
        target2desc = affection2desc
    targets = sorted(set(target2desc.values()))
    targets_cnt = {v: [0] * len(targets) for v in type2desc.values()}
    types_cnt = {v: 0 for v in type2desc.values()}

    for item in iter_data():
        for code_label in item['code_labels']:
            tp = code_label['hallucination-type']
            types_cnt[tp] += 1
            for t in code_label[f'{target}s']:
                targets_cnt[tp][targets.index(t)] += 1

    targets_cnt = {k: [v[i] / types_cnt[k] * 100 for i in range(len(v))] for k, v in targets_cnt.items()}
    print(targets_cnt)

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(16, 8))

    if target == 'factor':
        markers = {
            'P1': 'x',
            'P3': 's',
            'M2': '^'
        }
        colors = factors_color
    else:
        markers = {
            'A1': 'D',
            'A2': '^',
            'A4': 's',
            'A5': 'o'
        }
        colors = affections_color
    markers = {target2desc[k]: v for k, v in markers.items()}

    desc_list = list(targets_cnt.keys())
    desc_list[2], desc_list[3] = desc_list[3], desc_list[2]
    if target != 'factor':
        desc_list[2], desc_list[7] = desc_list[7], desc_list[2]
        desc_list[3], desc_list[5] = desc_list[5], desc_list[3]
        desc_list[6], desc_list[8] = desc_list[8], desc_list[6]
        desc_list[4], desc_list[6] = desc_list[6], desc_list[4]
    for i in range(len(targets)):
        lst = [targets_cnt[desc][i] for desc in desc_list]
        ax.plot([x for x in range(len(desc_list))], lst, linewidth=4, zorder=3, alpha=1.0, color=colors[targets[i]])
        ax.scatter([x for x in range(len(desc_list))], lst, alpha=1.0, marker=markers[targets[i]],
                   color=colors[targets[i]], s=100, label=targets[i])

    ax.xaxis.grid(True, zorder=1)
    ax.yaxis.grid(True, zorder=1)
    plt.xlabel('Hallucination Categories', fontsize=30, labelpad=20)
    plt.ylabel('Proportion (%)', fontsize=30, labelpad=25)
    ax.tick_params(axis='x', which='major', labelsize=17)
    ax.tick_params(axis='y', which='major', labelsize=25)
    ax.set_xticks([x for x in range(len(desc_list))])
    ax.set_xticklabels([map_desc(desc) for desc in desc_list])
    if target == 'factor':
        ax.legend(loc='lower right', fontsize=22, bbox_to_anchor=(0.45, 0.5))
    else:
        ax.legend(loc='lower right', fontsize=22, bbox_to_anchor=(0.44, 0.5))
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{"cause" if target == "factor" else "impact"}_analyze.pdf', format='pdf')


if __name__ == '__main__':
    main('factor')
    # main('affection')
