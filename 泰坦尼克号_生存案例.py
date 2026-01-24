import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 获取数据
data_df = pd.read_csv("data/train.csv")

# 数据基本处理
x = data_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = data_df['Survived']

# 缺失值填充
x['Age'].fillna(x['Age'].mean(), inplace=True)

# print(x.info())

#2.3对字符串进行one-hot 独热编码
x=pd.get_dummies(x)
x.drop(['Sex_female'], axis=1, inplace=True)

# print(x.head())

#2.4切分数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=66666,stratify=y)

#3.特征工程(标准化)
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

#4.模型训练   默认使用基尼值
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

#5.模型预测
y_predict=model.predict(x_test)
print(f"预测结果:{y_predict}")

#6.模型评估
print(f"准确率:{model.score(x_test,y_test)}")
print(f"分类报告:{classification_report(y_test,y_predict)}")

plt.figimage(model.feature_importances_)