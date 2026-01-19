"""
    案例:演示正则化解决拟合问题
    回顾:
        拟合: 指的是模型和数据之间的关系,即预测值和真是值之间的关系
        欠拟合: 模型在训练集,测试机表现都很不好
            原因: 模型训练过于简单
        过拟合: 模型在训练集表现很好,测试机表现很差
            原因: 模型过于复杂,数据不纯,数据量少
        正好拟合(泛化): 在训练集和测试即表现都很好
        奥卡姆剃刀: 在误差相同的情况下(泛化程度一样)的情况下,优先去选择简单的模型
        正则化解释:
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
# import matplotlib
# matplotlib.use('TkAgg')  # 强制使用 Tkinter 后端

# 代码实现_欠拟合_绘图
def demo01_欠拟合():
    # 1.生成数据
    # 指定numpy.random的随机种子,确保每次生成的数据相同
    np.random.seed(28)
    # 1.2 生成x轴值
    x = np.random.uniform(-3, 3, 100)
    # 1.3 生成机遇x轴的y轴值
    # 回顾: 一元线性回归公式: y = wx + b
    # normal(0, 1, 100) 含义: 生成100个服从正太分布的随机数,均值为0,标准差为1
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)
    # print(f"x:{x}")
    # print(f"y:{y}")
    # 当前的x,y是一位数组,x需要转换成二维数组
    print(x.shape)
    # print(y.shape)
    # 数据预处理
    x_train = x.reshape(-1, 1) # 转换成二维数组
    print(x_train.shape)
    # 3. 创建模型对象
    model = LinearRegression()
    model.fit(x_train, y)

    # 4. 模型预测
    y_predict = model.predict(x_train)

    # 5. 模型评估
    print(f"MSE:{mean_squared_error(y, y_predict)}")
    print(f"MAE:{mean_absolute_error(y, y_predict)}")
    print(f"RMSE:{root_mean_squared_error(y, y_predict)}")

    # 6. 绘图
    plt.scatter(x, y)
    plt.plot(x, y_predict, color='red')
    plt.show()

#
def demo02_正好拟合():
    # 1.生成数据
    np.random.seed(28)
    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)
    x_train = x.reshape(-1, 1)

    # 过拟合的数据因为只有一列,特征太少所以造成过拟合问题,现在我们添加特征
    # hstack 垂直合并
    x_train = np.hstack((x_train, x_train ** 2))

    model = LinearRegression()
    model.fit(x_train, y)
    y_predict = model.predict(x_train)
    print(f"MSE:{mean_squared_error(y, y_predict)}")
    print(f"MAE:{mean_absolute_error(y, y_predict)}")
    print(f"RMSE:{root_mean_squared_error(y, y_predict)}")
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()

def demo03_过拟合():
    # 1.生成数据
    np.random.seed(28)
    x = np.random.uniform(-3, 3, 1000)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, x.size)
    x_train = x.reshape(-1, 1)

    # 过拟合的数据因为只有一列,特征太少所以造成过拟合问题,现在我们添加特征
    # hstack 垂直合并
    x_train = np.hstack([x_train,x_train**2,x_train**26,x_train**4,x_train**8,x_train**25,
                         x_train**15,x_train**8,x_train**20,x_train**39])

    model = LinearRegression()
    model.fit(x_train, y)
    y_predict = model.predict(x_train)
    print(f"MSE:{mean_squared_error(y, y_predict)}")
    print(f"MAE:{mean_absolute_error(y, y_predict)}")
    print(f"RMSE:{root_mean_squared_error(y, y_predict)}")
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()


if __name__ == '__main__':
    # demo01_欠拟合()
    # demo02_正好拟合()
    demo03_过拟合()
