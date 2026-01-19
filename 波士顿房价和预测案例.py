"""
    案例:波士顿房价预测,正规方程结果
        回顾:
            正规方程解法和梯度下降的区别:
                相同点:都是用来找损失函数最小值,评估 回归模型
                不同点:
                    正规方程:一次性求解,资源开销比较大,适合小批量干净的数据集,如果矩阵不可逆,也无法计算
                    梯度下降:迭代求解,资源开销相对较小,使用于大批量的数据集,实际开发推介
                        分析:
                            全梯度下降算法:(FGD)每次迭代时,使用全部样本的梯度值
                            随机梯度下降算法:(SGD)每次迭代时,随机选择并使用一个梯度值
                            小批量梯度下降算法:(mini-batch)每次迭代时,随机选择并使用小批量的样本梯度值
                            随机平均梯度下降算法:(SAG) 每次迭代时,随机选择一个样本的梯度值和以往样本的梯度值的平均值
                                将每次的梯度值存储到列表中,当前值和以往的列表值求平均值

            线性模型评估方案:
                方案一:平均绝对误差(MAE): 误差绝对值的和的平均值
                方案二:均方根误差(RMSE):  误差平方和的平均值的平方根
                方案三:均方误差(MSE):     误差平方和的平均值

            线性回归模型:
                y = w0 + w1*x1 + w2*x2 + ... + wn*xn
                w0:截距
                w1-wn:回归系数
                x1-xn:自变量
                y:因变量

                正规方程:
                    w = (X.T*X)^-1 * X.T * y
                    X.T*X:自变量的转置乘自变量
                    X.T*y:自变量的转置乘因变量

            机器学习项目开发流程:
                1:获取数据
                2:数据基本处理(数据集划分)
                3:特征工程
                    特征提取,特征预处理(标准化,归一化)
                4:模型训练
                5:模型评估
                6:模型预测
"""
# 导包
# from sklearn.datasets import load_boston # 加载波士顿房价预测案例数据集
from sklearn.model_selection import train_test_split # 数据集划分
from sklearn.preprocessing import StandardScaler # 特征处理_标准化
from sklearn.linear_model import LinearRegression # 线性回归方程
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
import pandas as pd
import numpy as np

# load_boston = load_boston()
# print(load_boston.DESCR)

# 方式一:通过线上获取数据
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]
# print(data)

# 1. 方式二:通过本地获取数据
local_df_data = pd.read_csv("data/boston.csv")
print(local_df_data.info())

feature_columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
label_column = "MEDV"

data = local_df_data[feature_columns]
target = local_df_data[label_column]

# 2. 数据集划分
x_train,x_test,y_train,y_test = train_test_split(data, target, test_size=0.2, random_state=18)

# 3. 特征工程(标准化和归一化)
stand = StandardScaler()
x_train = stand.fit_transform(x_train)
x_test = stand.transform(x_test)

# 4. 模型训练
# 4.1创建模型对象
model = LinearRegression()

# 4.2模型训练
model.fit(x_train, y_train)

# 5. 模型预测
predict = model.predict(x_test)

print("预测结果:", predict)
print("真实结果:", y_test)
print(f"权重:{model.coef_}")
print(f"截距:{model.intercept_}")

# 模型评估  预测
print(f"(测试集)平均绝对误差为:{mean_absolute_error(y_test, predict)}")
print(f"(测试集)均方误差为:{mean_squared_error(y_test, predict)}")
print(f"(测试集)均方根误差为:{root_mean_squared_error(y_test, predict)}")
print(f"(测试集)均方根误差:{np.sqrt(mean_squared_error(y_test, predict))}")

