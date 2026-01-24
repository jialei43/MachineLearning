import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


df = pd.read_csv('./data/wine0501.csv')
df.info()

print(df['Class label'].unique())

df = df[df['Class label']!=1]

print(df['Class label'].unique())

y = df.iloc[:, 0]
# x = df.iloc[:, 9:]
x = df[['Alcohol','Magnesium','Total phenols','Color intensity','OD280/OD315 of diluted wines']]
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

x_train ,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2, random_state=22)

standardScaler = StandardScaler()
x_train = standardScaler.fit_transform(x_train)
x_test = standardScaler.transform(x_test)

# 场景一:单一决策树 ->充当一个弱分类器
model = DecisionTreeClassifier(max_depth=3)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(f'预测值:{y_pred}')
print(f'准确率:{model.score(x_test,y_test)}')
print("-" * 34)

adaBoostModel = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=70, learning_rate=0.01, algorithm='SAMME')
adaBoostModel.fit(x_train,y_train)
y_pred2 = adaBoostModel.predict(x_test)
print(f'预测值:{y_pred2}')
print(f'准确率:{adaBoostModel.score(x_test,y_test)}')
