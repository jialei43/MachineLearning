# 导包
from sklearn.preprocessing import MinMaxScaler

"""
    案例: 演示特征预处理 之归一化
    回顾:特征工程目的和步骤
        目的:
            利用专业知识背景  和技巧处理数据 ,用于提升模型性能
        步骤:
            1:特征提取
            2:特征预处理(归一化和标准化)
            3:特征降维
            4:特征选择
            5:特征组合
        特征预处理解释:
            背景:
                在实际开发中,如果多个特征列因为量纲问题,导致数值差距过大.
                则会导致 模型预测值出现偏差
                    例如:
                        身高->单位 米
                        体重->单位 kg(公斤)
                    为了保证每个特征列,对最终预测结果 的权重都是相近的.
                    所以要进行特征预处理操作.
            实现方式:
                归一化
                标准化
        归一化介绍:
            概述: 它是特征预处理一种方案.  采用 sklean.preprocessing.MinMaxScaler类
                对原始的数据做处理,获取1个区间[mi,mx]=>默认 [0,1]
            公式:
                x'=(x-min)/(max-min)
                x''=x'*(mx-mi)+mi
            弊端:
                强依赖于该列特征的最大值 和最小值,如果差值比较大的话,计算效果可能不明显
            适用场景:
                归一化适用于小数据集的特征预处理
"""

# 准备特征数据
x = [
    [90, 2,10,40],
    [60, 4,15,45],
    [75, 3,13,46]
    ]

# 创建归一化对象
minmax_scaler = MinMaxScaler(feature_range=(0, 1))
# 具体的归一化动作
## 扩展 : fit_transform(data) 和 transform(data)
## fit_transform(data) : 训练并归一化 适用于训练集(计算一次,均值,标准差,最大值和最小值即可)
## transform(data) : 只归一化 适用于测试集(已经训练过)
transform = minmax_scaler.fit_transform(x)
print(transform)
