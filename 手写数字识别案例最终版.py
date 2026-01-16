import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")


# 定义函数 训练模型,并保存训练好的模型
def train_save_model():
    # 1.加载数据集
    df = pd.read_csv("./data/手写数字识别.csv")

    # 2.获取所有的特征和标签数据
    x_train = df.iloc[1:,1:]
    y_train = df.iloc[:,0]

    # 对特征进行预处理(标准化处理)
    stand = StandardScaler()
    x_train = stand.fit_transform(x_train)

    # 创建模型对象
    model = KNeighborsClassifier()

    #创建网格搜索和交叉验证的对象
    gridsearch = GridSearchCV(model, param_grid={"n_neighbors":range(2,245)}, cv=5)

