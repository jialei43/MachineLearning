"""

"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv("./data/breast-cancer-wisconsin.csv")

# 数据基本处理
data.replace({"?": np.nan}, inplace=True)
data.dropna(axis = 0,inplace=True)
data.info()

# 特征工程: 特征提取 特征予处理 特征降维 特征选取 特征组合
# 获取特征矩阵和标签向量
x = data.iloc[:,1:-1]
y = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=28)

# 数据标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 创建模型对象 逻辑回归模型
model = LogisticRegression()
model.fit(x_train,y_train)

# 模型预测
y_predict = model.predict(x_test)

# 模型评估
print(f"准确率:{accuracy_score(y_test,y_predict)}")


# 模型评估
print(f"MAE:")
