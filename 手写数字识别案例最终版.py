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



    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=22)

    # 对特征进行预处理(标准化处理)
    stand = StandardScaler()
    x_train = stand.fit_transform(x_train)
    x_test = stand.transform(x_test)

    # 创建模型对象
    model = KNeighborsClassifier()

    #创建网格搜索和交叉验证的对象
    gridsearch = GridSearchCV(model, param_grid={"n_neighbors":range(2,10)}, cv=5)

    # 模型训练
    gridsearch.fit(x_train, y_train)

    print(f"准确率:{gridsearch.score(x_test, y_test)}")
    print(f"参数:{gridsearch.best_params_}")
    # 保存模型和标准化器
    joblib.dump(gridsearch.best_estimator_, "./data/model.pkl")
    joblib.dump(stand, "./data/stand.pkl")

    # 保存训练数据的统计信息
    train_stats = {
        'train_min': x_train.min(),
        'train_max': x_train.max(),
        'train_mean': x_train.mean(),
        'train_dtype': str(x_train.dtype)
    }
    joblib.dump(train_stats, "./data/train_stats.pkl")

    print(f"模型和标准化器保存成功")



def load_model_predict():
    # 1.加载模型、标准化器和训练统计信息
    model = joblib.load("./data/model.pkl")
    stand = joblib.load("./data/stand.pkl")
    train_stats = joblib.load("./data/train_stats.pkl")

    print("=== 训练数据统计 ===")
    print(f"训练数据范围: [{train_stats['train_min']:.2f}, {train_stats['train_max']:.2f}]")
    print(f"训练数据均值: {train_stats['train_mean']:.2f}")

    # 2.加载测试图像
    img = plt.imread("./data/demo_2.png")

    print("=== 测试图像原始统计 ===")
    print(f"测试图像范围: [{img.min():.4f}, {img.max():.4f}]")

    # 3.对齐数值范围
    img_new = img.reshape(1, -1)

    # 如果训练数据是0-255范围，而测试图像是0-1范围
    if train_stats['train_max'] > 1.0 and img.max() <= 1.0:
        print("将测试图像从[0,1]转换到[0,255]")
        img_new = img_new * 255.0
    # 如果训练数据是0-1范围，而测试图像是0-255范围
    elif train_stats['train_max'] <= 1.0 and img.max() > 1.0:
        print("将测试图像从[0,255]转换到[0,1]")
        img_new = img_new / 255.0

    print(f"调整后测试图像范围: [{img_new.min():.2f}, {img_new.max():.2f}]")

    # 4.使用训练时的标准化器进行标准化
    img_scaled = stand.transform(img_new)

    # 5.模型预测
    img_predict = model.predict(img_scaled)

    print(f"识别结果:{img_predict}")

if __name__ == '__main__':
    # train_save_model()
    load_model_predict()



