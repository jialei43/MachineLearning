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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler # 特征处理_标准化
from sklearn.linear_model import SGDRegressor # 线性回归方程
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
# stand = StandardScaler()
# x_train = stand.fit_transform(x_train)
# x_test = stand.transform(x_test)

# 4. 模型训练
# 4.1创建模型对象
"""
这是 Scikit-Learn 中 SGDRegressor（随机梯度下降回归器）的初始化参数翻译与详细解析。
SGDRegressor 是一个通过随机梯度下降来拟合线性模型的回归方法。
核心超参数 (Core Hyperparameters)
    loss="squared_error" (损失函数):指定要使用的损失函数。
    "squared_error": 普通最小二乘拟合（默认）。
    "huber": 鲁棒回归，减少异常值的影响。
    "epsilon_insensitive": 类似于 SVR（支持向量回归）
     penalty="l2" (惩罚项/正则化):正则化类型。"l2": 
     岭回归 (Ridge)，倾向于让权重均匀变小。"l1": Lasso 回归，倾向于产生稀疏权重（让某些特征权重变为 0）。
     "elasticnet": L1 和 L2 的结合。
     alpha=0.0001:正则化强度。值越大，正则化越强，模型越简单，有助于防止过拟合。
     l1_ratio=0.15:混合比。仅当 penalty="elasticnet" 时有效，代表 L1 正则化所占的比例。
训练与优化设置 (Training & Optimization)
    fit_intercept=True:是否拟合截距（偏置项 $b$）。如果数据已经中心化，可以设为 False。
    max_iter=1000:最大迭代次数（训练轮数/Epochs）。
    tol=1e-3 (容差):停止准则。如果在连续的迭代中损失降低小于这个值，训练将提前停止。
    shuffle=True:每一轮迭代后是否打乱训练数据。建议开启以提高收敛性能。
    random_state=None:随机数种子，用于控制打乱数据和初始化的随机性。
学习率控制 (Learning Rate)
    learning_rate="invscaling" (学习率策略):"constant": 学习率保持为 eta0 不变。
    "optimal": 根据启发式算法自动调整。
    "invscaling": 随着时间推移逐渐减小学习率（默认）。
    "adaptive": 如果损失不下降，则自动减小学习率。
    eta0=0.01:初始学习率。
    power_t=0.25:逆缩放学习率（invscaling）的指数。
提前停止与验证 (Early Stopping)
    early_stopping=False:是否在损失停止改善时自动停止训练。如果设为 True，会分出一部分验证集。
    validation_fraction=0.1:从训练数据中划出 10% 作为验证集，用于执行提前停止。
    n_iter_no_change=5:在停止前，允许多少个轮次（epochs）模型性能没有提升。
其他工具 (Miscellaneous)
    verbose=0:是否输出训练过程中的日志信息。
    warm_start=False:如果设为 True，再次调用 fit 时会保留上次训练的结果继续练，而不是重新初始化。
    average=False:如果设为 True 或整数，则计算并存储所有更新权重的平均值，有时能提升稳定性。
建议：如果你在处理大规模数据集（样本数超过 10 万），SGDRegressor 是非常理想的选择。
"""
# model = SGDRegressor()

# 2. 创建模型并配置参数
# 建议：SGD 对特征缩放非常敏感，所以通常配合 StandardScaler 使用
model = make_pipeline(
    StandardScaler(),
    SGDRegressor(
        loss="squared_error",  # 损失函数：均方误差
        penalty="l2",           # 正则化：L2（防止模型过拟合）
        alpha=0.01,             # 正则化强度
        max_iter=1000,          # 最大迭代次数
        tol=1e-3,               # 停止条件：损失下降小于这个值就停
        learning_rate="adaptive", # 学习率策略：自适应（如果损失不降，学习率就减小）
        eta0=0.01,              # 初始学习率
        random_state=42         # 随机种子：保证结果可复现
    )
)
# 4.2模型训练
model.fit(x_train, y_train)
train_predict = model.predict(x_train)
# print("预测结果:", predict)

# 5. 模型预测
test_predict = model.predict(x_test)

# print("预测结果:", test_predict)
# print("真实结果:", y_test)
print(f"权重:{model.named_steps['sgdregressor'].coef_}")
print(f"截距:{model.named_steps['sgdregressor'].intercept_}")

# 模型评估  预测
print(f"(训练集)平均绝对误差为:{mean_absolute_error(y_train, train_predict)}")
print(f"(训练集)均方误差为:{mean_squared_error(y_train, train_predict)}")
print(f"(训练集)均方根误差为:{root_mean_squared_error(y_train, train_predict)}")
print(f"(训练集)均方根误差:{np.sqrt(mean_squared_error(y_train, train_predict))}")


print(f"(测试集)平均绝对误差为:{mean_absolute_error(y_test, test_predict)}")
print(f"(测试集)均方误差为:{mean_squared_error(y_test, test_predict)}")
print(f"(测试集)均方根误差为:{root_mean_squared_error(y_test, test_predict)}")
print(f"(测试集)均方根误差:{np.sqrt(mean_squared_error(y_test, test_predict))}")

