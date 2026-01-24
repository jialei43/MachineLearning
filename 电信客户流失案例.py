import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report
from sklearn.preprocessing import StandardScaler

# 获取数据
df = pd.read_csv("data/churn.csv")

# 数据基本处理

# 替换 No,yes 数据清洗
df.replace( {"No":0,"Yes":1}, inplace=True)
print(df.head)

print('-'*34)
# 特征工程
x = df.iloc[:,2:]
print(x.info())
y = df.iloc[:,0]
print(y.head)
print('-'*34)
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# 标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 创建模型对象
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("预测结果:", y_pred)
print("真实结果:", y_test)
print(f"权重:{model.coef_}")
print(f"截距:{model.intercept_}")
print(f"准确率:{accuracy_score(y_test, y_pred)}")

# 模型评估  预测
# print(f"(测试集)平均绝对误差为:{mean_absolute_error(y_test, predict)}")
# print(f"(测试集)均方误差为:{mean_squared_error(y_test, predict)}")
# print(f"(测试集)均方根误差为:{root_mean_squared_error(y_test, predict)}")
# print(f"(测试集)均方根误差:{np.sqrt(mean_squared_error(y_test, predict))}")

lable = [0,1]
df_lable = ["不流失(正例)", "流失(反例)"]
cm = confusion_matrix(y_test, y_pred, labels=lable)
df_cm = pd.DataFrame(cm, index=df_lable, columns=df_lable)

print(df_cm)

# 打印y_preA 和y_preB的精确率、召回率、F1
print(f"y_preA的准确率:{accuracy_score(y_test, y_pred)}")
print(f"y_preA的精确率:{precision_score(y_test, y_pred, pos_label=0)}")
print(f"y_preA的召回率:{recall_score(y_test, y_pred, pos_label=0)}")
print(f"y_preA的F1:{f1_score(y_test, y_pred, pos_label=0)}")

print(f"分类报告:{classification_report(y_test,y_pred)}")