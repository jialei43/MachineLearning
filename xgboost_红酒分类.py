"""
    案例:
        通过xgboost 极限梯度提升树  完成红酒品质分类
"""
from collections import Counter # 统计数据是否平衡,也就是统计标签列类别是否平衡

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import StratifiedKFold #分层k折交叉验证,类似于网格搜索时的cv
from sklearn.utils import compute_sample_weight  # 正确处理样本权重


def dm01_data_split():
    """
        加载数据
    :return:
    """
    path = "./data/红酒品质分类.csv"
    df = pd.read_csv(path)

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]-3

    # 3.考虑样本均衡
    print(f'查看标签分布是否均衡:{Counter(y)}')
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=28,stratify=y)

    pd.concat([x_train,y_train],axis=1).to_csv("./data/红酒训练集_train.csv",index=False)
    pd.concat([x_train,y_train],axis=1).to_csv("./data/红酒训练集_test.csv",index=False)

def dm02_xgboost_model():
    # 读取训练集
    train_data = pd.read_csv("./data/红酒训练集_train.csv")
    test_data = pd.read_csv("./data/红酒训练集_test.csv")

    x_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1]

    x_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]

    standardScaler = StandardScaler()
    x_train = standardScaler.fit_transform(x_train)
    x_test = standardScaler.transform(x_test)

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=28)

    gd = GridSearchCV(
        xgb.XGBClassifier(
            # random_state=28,
            objective='multi:softmax',  # 多分类问题,使用多分类模型
            num_class=3,
            eval_metric='mlogloss'
        ),
        param_grid={
            "n_estimators": list(range(20, 200, 40)),
            "max_depth": list(range(3, 10, 2)),
            "learning_rate": [0.1, 0.01, 0.05],
            # "min_samples_leaf":list(range(3,10,1)),
            "random_state": list(range(20, 100, 20)),
        },
        cv=skf,
        verbose=1

    )
    gd.fit(x_train,y_train)

    cw = compute_sample_weight('balanced', y_train)

    # model.fit(x_train,y_train,sample_weight=cw)
    gd.fit(x_train,y_train,sample_weight=cw)

    # print(f'准确率:{model.score(x_test,y_test)}')
    print(f'准确率:{gd.score(x_test,y_test)}')
    print(f'model:{gd.best_estimator_}')
    #保存模型
    joblib.dump(gd,"./data/红酒分类模型.pkl")
    joblib.dump(gd,"./data/红酒分类模型.pkl")
    print("保存模型成功")

def dm03_use_model():
    # 读取训练集
    train_data = pd.read_csv("./data/红酒训练集_train.csv")
    test_data = pd.read_csv("./data/红酒训练集_test.csv")

    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    # 加载模型
    model = joblib.load("./data/红酒分类模型.pkl")

    y_pred = model.predict(x_test)
    y_train_pred = model.predict_proba(x_train)
    print(f"参数:{model.best_params_}")
    print(f'测试集准确率:{accuracy_score(y_test,y_pred)}')
    print(f'训练集准确率:{accuracy_score(y_train,y_train_pred)}')

if __name__ == '__main__':
    # dm01_data_split()
    # dm02_xgboost_model()
    dm03_use_model()