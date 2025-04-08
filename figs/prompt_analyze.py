import random
import matplotlib.pyplot as plt
import numpy as np
from tools import iter_data


def main():
    lengths = []
    complexities = []
    all_lengths = []
    all_complexities = []
    random.seed(42)

    for item in iter_data():
        all_lengths.append(item['prompt_length'])
        # all_complexities.append(item['prompt_label']['logic-complexity'] + random.random() - 0.5)
        all_complexities.append(item['prompt_label']['logic-complexity'])
        if len(item['code_labels']) > 0:
            lengths.append(all_lengths[-1])
            complexities.append(all_complexities[-1])

    bins = (16, 3)  # 设定网格划分数量
    heatmap_range = ([min(*all_lengths), max(*all_lengths)], [0.5, 3.5])  # 坐标范围

    # 计算两组散点在网格中的数量
    count1, _, _ = np.histogram2d(lengths, complexities, bins=bins, range=heatmap_range)
    count2, _, _ = np.histogram2d(all_lengths, all_complexities, bins=bins, range=heatmap_range)

    # 计算比值（避免除零）
    ratio = np.divide(count1, count2, out=np.full_like(count1, -0.4), where=(count2 != 0)) * 100

    # 绘制热力图
    plt.rcParams['font.family'] = 'Times New Roman'
    _, ax = plt.subplots(figsize=(16, 12))
    cax = plt.imshow(ratio.T, origin='lower', extent=(*heatmap_range[0], *heatmap_range[1]), cmap='gray_r',
                     aspect='auto')
    cbar = plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label="Hallucinatory Code Proportion (%)", fontsize=40, labelpad=25)
    plt.scatter(all_lengths, all_complexities, c='lightgreen', s=1.5, label="No Hallucination Data Points", alpha=0.8)
    plt.scatter(lengths, complexities, c='red', s=1.5, label="Hallucination Data Points", alpha=0.8)
    lgd = plt.legend(fontsize=25)
    for handle in lgd.legend_handles:
        handle.set_alpha(1.0)
    plt.xlabel('Prompt Tokens (#)', fontsize=40, labelpad=25)
    plt.ylabel('Text Complexity', fontsize=40, labelpad=25)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_yticks([1.0, 2.0, 3.0])
    plt.tight_layout()
    plt.savefig('prompt_analyze_origin.pdf', format='pdf')
    # plt.show()


if __name__ == '__main__':
    main()


"""
Length Analyze
"""
# bins = np.linspace(min(min(length), min(hallu_length)), max(max(length), max(hallu_length)), 21)
# hist_len, _ = np.histogram(length, bins)
# hist_hallu_len, _ = np.histogram(hallu_length, bins)
# print(hist_len)
# print(hist_hallu_len)
# ratios = hist_hallu_len / hist_len * 100
#
# plt.bar(bins[:-1], ratios, width=np.diff(bins), edgecolor='black', alpha=0.5, color='g', zorder=3)
#
#
# plt.grid(axis='y', zorder=1)
# plt.xlabel('Prompt Tokens (#)', fontdict={'fontsize': 14})
# plt.ylabel('Proportion (%)', fontdict={'fontsize': 14})

"""
Complexity Analyze
"""
# bins = np.linspace(1, 6, 6)
# print(bins)
# hist_comp, _ = np.histogram(complexity, bins)
# hist_hallu_comp, _ = np.histogram(hallu_complexity, bins)
# print(hist_comp)
# print(hist_hallu_comp)
# ratios = hist_hallu_comp / hist_comp * 100
# print(ratios)
# plt.bar(bins[:-1], ratios, width=0.3 * np.diff(bins), edgecolor='black', alpha=0.5, color='orange', zorder=3)
# plt.plot([1, 2, 3, 4, 5], ratios, marker='v', markersize=12, zorder=5)
#
# plt.grid(axis='y', zorder=1)
# plt.xlabel('Logic Complexity', fontdict={'fontsize': 14})
# plt.ylabel('Proportion (%)', fontdict={'fontsize': 14})


plt.tight_layout()

# plt.savefig('prompt_complexity.pdf', format='pdf')
# plt.show()
