# 导包
from sklearn.preprocessing import StandardScaler

"""
数据预处理:标准化
标准化
    概述:它是特征预处理一种方法 采用  sklean.preprocessing.standardScaler类
    公式:
        x'=(x-该列平均值)/该列标准差
    应用场景:
        比较适合大数据集的应用场景,当数据量比较大的情况,受到最大值和最小值的影响会变得微乎其微.
        在实际开发中一般采用标准化进行处理.
    总结:
        无论是归一化还是标准化,目的都是避免特征因为量纲问题导致权重不同,从而影响最终预测结果.目的是一样.

"""
x = [
    [90, 2,10,40],
    [60, 4,15,45],
    [75, 3,13,46]
    ]
# 创建标准化对象
scaler = StandardScaler()

transform = scaler.fit_transform(x)
print(transform)

print('-' * 34)

# 特征处理平均值
print(f'特征处理平均值:{scaler.mean_}')
print('-' * 34)

# 特征处理标准化
print(f'特征处理标准化: {scaler.scale_}')
print('-' * 34)

# 特征处理方差
print(f'特征处理方差:{scaler.var_}')
print('-' * 34)

