"""
案例:
    演示KNN算法 识别图片,即:手写数字识别案例
    介绍:
        每张图片都有28*28像素组成,即我们csv文件每一行都有784像素点,表示图片(每个像素点)的颜色
        最终构成图像
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
from collections import Counter

warnings.filterwarnings('ignore',module='sklearn')

def show_digit(idx):
    df = pd.read_csv('data/手写数字识别.csv')
    print(df)
    if idx < 0 | idx > len(df)-1:
        print('索引值错误')
        return
    x = df.iloc[:,1:] # 获取特征矩阵
    y = df.iloc[:,0] # 获取标签向量

    print(f'您传入的索引,对应的数字是:{y[idx]}')

    x=x.iloc[idx].values.reshape(28,28)
    plt.imshow(x,cmap='gray')
    plt.axis('off')
    # plt.show()
    print(f'查看所有标签的分布情况:{Counter(y)}')

# 2.定义函数 训练模型,并保存训练好的模型
def train_model():
    df = pd.read_csv('data/手写数字识别.csv')
    x = df.iloc[:,1:]
    y = df.iloc[:,0]

    # 2.2 特征工程
    x = x/255

    # parmar1:特征列 param2:标签列 param3:测试集比例 param4:随机种子 param5:是否分层
    x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=18,stratify=y)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(f'准确率:{accuracy_score(y_test,y_pred)}')
    joblib.dump(model,"./data/model.pkl")


# 3.定义函数 测试模型
def test_model():
    # 读取图片
    img = plt.imread('data/demo_5.png')
    print(img) # 28*28的二维像素数组
    # 读取模型,加载模型
    model = joblib.load("./data/model.pkl")
    # 模型训练
    # img_predict = model.predict(img.reshape(1,784))
    # img.reshape(1,-1) -1是自动计算
    img_predict = model.predict(img.reshape(1,-1))
    print(f'识别结果:{img_predict}')
if __name__ == '__main__':
    show_digit(9)
    # # train_model()
    # test_model()