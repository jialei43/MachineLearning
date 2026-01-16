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
    x_train = df.iloc[:,1:]
    y_train = df.iloc[:,0]

    # 对特征进行预处理(标准化处理)
    stand = StandardScaler()
    x_train = stand.fit_transform(x_train)

    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=22)

    # 创建模型对象
    model = KNeighborsClassifier()

    #创建网格搜索和交叉验证的对象
    gridsearch = GridSearchCV(model, param_grid={"n_neighbors":range(2,245)}, cv=5)

    # 模型训练
    gridsearch.fit(x_train, y_train)

    print(f"准确率:{gridsearch.score(x_test, y_test)}")
    print(f"参数:{gridsearch.best_params_}")
    joblib.dump(gridsearch.best_estimator_,"./data/model.pkl")

def load_model_predict():
    # 1.加载模型
    model = joblib.load("./data/model.pkl")
    # 2.加载数据
    img = plt.imread("./data/demo_5.png")

    # 将img 的28*28的二维数组,转换成1*784的二维数组
    img_new = img.reshape(1,-1)

    # 对特征进行预处理(标准化处理)
    stand = StandardScaler()
    img_new = stand.fit_transform(img_new)

    # 模型预测
    img_predict = model.predict(img_new)

    print(f"识别结果:{img_predict}")

if __name__ == '__main__':
    train_save_model()
    load_model_predict()



