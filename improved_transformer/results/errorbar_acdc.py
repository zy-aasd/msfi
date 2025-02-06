import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 你提供的表格数据
data = {
    'DSC': [91.73,91.80,91.75,91.95,91.74,92.01,91.79,91.82,91.72,91.89],
    'RV': [90.35,90.26,89.98,90.57,90.12,90.72,90.32,90.42,90.00,90.27],
    'Myo': [89.09,89.27,89.38,89.40,89.29,89.42,89.24,89.21,89.42,89.43],
    'LV': [95.74,95.88,95.89,95.89,95.80,95.87,95.83,95.84,95.74,95.97]
}

# 计算均值和标准差
mean = {key: np.mean(value) for key, value in data.items()}
std = {key: np.std(value) for key, value in data.items()}
n = len(data['DSC'])  # 数据点数量

# 计算标准误差和95%的置信区间
se = {key: std[key] / np.sqrt(n) for key in data}
ci = {key: 1.96 * se[key] for key in data}


# 使用柔和的 Pastel1 调色板
colors = [
    '#6a0dad',  # 紫色
    '#8a2be2',  # 蓝紫色
    '#4682b4',  # 蓝色
    '#5f9ea0',  # 蓝绿色
    '#3cb371',  # 绿色
    '#2e8b57',  # 深绿色
    '#9acd32',  # 黄绿色
    '#ffd700',  # 金色
    '#ff8c00',  # 橙色
    '#ffa07a',  # 淡橙色
]

# 绘制带误差棒的图
labels = list(data.keys())
means = list(mean.values())
errors = list(ci.values())
plt.figure(figsize=(6, 5))
bars = plt.bar(labels, means, yerr=errors, capsize=9, color=colors, edgecolor='black', width=0.4)

# 设置y轴范围为80到100
plt.ylim(70, 100)

# 添加数据标签
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + errors[i] + 1,
             f'{means[i]:.2f} ± {errors[i]:.2f}', ha='center', va='bottom', fontsize=10)
# 在图形左上角添加误差棒代表的意义
plt.axhline(color='black', linestyle='-', label=f'Mean ± 95% CI')
plt.legend(loc='upper left')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('./errorbar_acdc.png')