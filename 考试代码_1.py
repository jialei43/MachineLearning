# 导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体（macOS 兼容版本）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

# ======================== 1. 准备数据 ========================
data = {
    'x': [1, 3, 2, 1, 3],
    'y': [14, 24, 18, 17, 27]
}
df = pd.DataFrame(data)
print("原始数据集：")
print(df)
print("-" * 30)
# ======================== 3. sklearn库训练模型 ========================
X = df['x'].values.reshape(-1, 1)
y = df['y'].values
lr_model = LinearRegression()
lr_model.fit(X, y)
sklearn_slope = lr_model.coef_[0]
sklearn_intercept = lr_model.intercept_

print("【sklearn库计算结果】")
print(f"斜率（k）：{sklearn_slope}")
print(f"截距（b）：{sklearn_intercept}")
print(f"回归方程：y = {sklearn_slope}x + {sklearn_intercept}")
print("-" * 30)

# ======================== 4. 绘制可视化图像 ========================
# 生成拟合直线的x、y值（让直线更平滑，覆盖数据范围）
x_fit = np.linspace(df['x'].min() - 0.5, df['x'].max() + 0.5, 100)
y_fit = sklearn_slope * x_fit + sklearn_intercept

# 创建画布
plt.figure(figsize=(8, 6), dpi=100)
# 绘制原始数据散点
plt.scatter(df['x'], df['y'], color='#e74c3c', s=80, label='原始数据点', zorder=3)
# 绘制线性回归拟合直线
plt.plot(x_fit, y_fit, color='#2980b9', linewidth=2, label='拟合直线')
# 标注回归方程、斜率、截距

# 添加网格、图例
# 调整布局，防止标签被截断
plt.tight_layout()
# 显示图像
plt.show()

# 可选：保存图像到本地（取消注释即可，保存为高清png）
# plt.savefig('线性回归拟合图.png', dpi=300, bbox_inches='tight')