"""
    交叉验证解释:
        概述:
            交叉验证是一种更完善\可信度更高的模型评估方法,其思路是把训练集分成N份
            每次都取1份当做验证集.其他当训练集.完计算模型评估
            思路:
                把训练集分成N份,例如分为4分- >也叫作:4折交叉验证
                第1次: 把第1分数据作为 验证集(测试集),其他叫做训练集,训练模型,模型预测,获取:准确率->准确率1
                第2次: 把第2分数据作为 验证集(测试集),其他叫做训练集,训练模型,模型预测,获取:准确率->准确率2
                第3次: 把第3分数据作为 验证集(测试集),其他叫做训练集,训练模型,模型预测,获取:准确率->准确率3
                第4次: 把第4分数据作为 验证集(测试集),其他叫做训练集,训练模型,模型预测,获取:准确率->准确率4
                然后计算上述4次准确率的平均值 ,作为:模型最终 准确率

            情况1:无独立测试集时:
                交叉验证完成后,若已经确定最优的模型
                用全部原始数据(不划分训练和验证)重新训练模型,此时模型将学习整个数据集模式
            情况2:有独立测试集时:
                交叉验证仅用于训练集内部调优,最终需要未参与交叉验证的独立测试集评估模型泛化能力,
                避免数据窥探(即模型在训练模型中"见过"测试集数据)
        目的:
            为了让模型最终结果更准确,更能代表模型的综合水平
        好处:
            相较于单一切分的数据集,交叉验证结果可信度更高
    网格搜索:
        目的/作用:
            寻找最优超参
        原理:
            接收超参可能出现的值,然后针对于超参数的每一个值进行交叉验证,获取最优的超参组合.
        超参数:
            需要用户手动录入的数据.
    大白话解释:
        网格搜索+交叉验证.本质指的是GridSearchCV这个API .
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # 加载鸢尾花数据集
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV  # 分割训练集和测试集  寻找最优的超参
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.neighbors import KNeighborsClassifier  # KNN算法 分类对象
from sklearn.metrics import accuracy_score  # 模型评估,计算模型预测准确率

iris_data = load_iris()

"""
在机器学习中，train_test_split() 是一个非常常用的 API，
通常用来将数据集划分为训练集（x_train, y_train）和测试集（x_test, y_test）。
它的基本作用是随机划分数据集，并返回四个值：训练集的特征（x_train）、测试集的特征（x_test）、训练集的标签（y_train）和测试集的标签（y_test）。

random_state 超参数
作用: random_state 用于控制数据划分的随机性。它指定了一个固定的种子值，使得每次调用 train_test_split 时，数据划分的结果是可复现的。即使多次运行相同的代码，得到的训练集和测试集划分都会相同。
影响:
随机性：当不指定 random_state 时，每次运行代码时，train_test_split 会随机地选择不同的训练集和测试集，可能导致模型训练和评估结果的不一致。
固定种子：通过指定固定的 random_state 值（例如 random_state=12），你能保证每次划分的数据集是相同的，特别是在进行调试或展示代码时，保持结果的稳定性是非常重要的
"""
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=12)


transfer=StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

model=KNeighborsClassifier()
param_dict={
    'n_neighbors':[i for i in range(2,11)]  #
}

model=GridSearchCV(model,param_dict,cv=4)
model.fit(x_train,y_train)
print(f"最优评分:{model.best_score_}")
print(f"最优超参组合:{model.best_params_}")
print(f"最优估计器对象:{model.best_estimator_}")
# print(f"")

#使用最优的评分
model2=KNeighborsClassifier(n_neighbors=3)
model2.fit(x_train,y_train)
y_pre=model2.predict(x_test)
print(f"准确率:{accuracy_score(y_test,y_pre)}")