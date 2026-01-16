"""
回顾:机器学习建模流程
    1:加载数据集
    2:数据的预处理
    3:特征工程
        特征预处理(归一化或者标准化)
    4:模型训练->KNN算法
    5:模型预测
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # 加载鸢尾花数据集
import seaborn as sns
from sklearn.model_selection import train_test_split  # 分割训练集和测试集
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.neighbors import KNeighborsClassifier  # KNN算法 分类对象
from sklearn.metrics import accuracy_score  # 模型评估,计算模型预测准确率


# 1:定义函数 dm01_load_iris() 加载鸢尾花数据集
def dm01_load_iris():
    # 1.加载鸢尾花数据集
    iris_data = load_iris()
    # print(iris_data)
    # 数据集格式介绍:
    # {
    # "data":  特征矩阵
    # "target": 标签数组
    # "frame": None
    # "target_names": 类别标签名称
    # "DESCR":   数据集描述
    # "feature_names" 特征名称
    # "filename":  文件名
    # "data_module": 模型
    # }
    # 分析1:获取未知数据keys
    print(iris_data.keys())

    # 分析2:获取前5行特征矩阵
    print(iris_data.data[:5])

    # 分析3:获取是所有特征字段
    print(iris_data.feature_names)


# 2:定义函数 查看数据散点图 ,对鸢尾花数据集进行可视化
def dm02_show_iris():
    # 1.加载数据
    iris_data = load_iris()
    # 2.将上述的数据封装成df对象,用于:可视化展示
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    # print(iris_df)
    iris_df['label'] = iris_data.target
    # print(iris_df)
    # 做可视化分析
    # 参1:数据集  iris_df
    # x:x轴 花瓣长度  y:y轴:花瓣宽度
    # hue :颜色(根据鸢尾花的标签来分组,不同颜色展示)
    # fit_reg=False 不绘制拟合回归线, True 绘制拟合回归线
    sns.lmplot(data=iris_df, x="petal length (cm)", y="petal width (cm)", hue='label', fit_reg=True)
    plt.title("iris data")
    plt.show()


# 3.定义函数 dm03_train_test_split() 实现:划分 训练集 和测试集
def dm03_train_test_split():
    # 1.加载数据集
    iris_data = load_iris()
    # 2.具体划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                        random_state=13)
    print(f"训练集 x-特征:{x_train}")  # 120
    print(f"测试集 x-特征:{x_test}")  # 30
    print(f"训练集 y-标签:{y_train}")  # 120
    print(f"测试集 y-标签:{y_test}")  # 30


# 4.定义函数 完成模型训练与评估
def dm04_模型训练与评估():
    # 1.加载数据
    iris_data = load_iris()
    # 2.数据预处理
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                        random_state=18)

    # 3特征工程,子公司特征预处理(选做)
    # 3.1创建标准化对象
    transfer = StandardScaler()
    # 3.2对特征列进行标准化
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.模型训练
    # 4.1实例化模型
    model = KNeighborsClassifier(n_neighbors=5)

    # 4.2模型训练
    model.fit(x_train, y_train)  # 训练集特征训练集标签

    # 5.模型预测
    y_predict = model.predict(x_test)
    print(f"(测试集)预测结果为:{y_predict}")
    print(f"(测试集)真实结果为:{y_test}")

    # 6.模型评估
    # 方式一:model.score
    print(f"直接评估:准确率:{model.score(x_test, y_test)}")  # 测试集特征  测试集表标签
    # 方式二:accuracy_score()
    print(f"预测值和真实值,准确率:{accuracy_score(y_test, y_predict)}")

    # 7.模型使用
    x_test_new = [[2.1, 3.5, 5.6, 3.2]]
    y_test_prd = model.predict(x_test_new)
    print(f"(新数据集)预测结果:{y_test_prd}")


if __name__ == '__main__':
    # dm01_load_iris()
    # dm02_show_iris()
    # dm03_train_test_split()
    # for i in range(100):
    dm04_模型训练与评估()
