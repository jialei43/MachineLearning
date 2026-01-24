import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./data/train.csv')
df.info()

x = df[['Pclass', 'Pclass', 'Sex', 'Age']]
y = df['Survived']

x = pd.get_dummies(x)

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2, random_state=28)

standardScaler = StandardScaler()
x_train = standardScaler.fit_transform(x_train)
x_test = standardScaler.transform(x_test)

# 场景一:单一决策树
model = DecisionTreeClassifier(max_depth=4)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(f'预测值:{y_pred}')
print(f'准确率:{model.score(x_test,y_test)}')
print("-" * 34)

# 场景二:随机森林
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=20,max_depth=4)
model2.fit(x_train,y_train)
y_pred2 = model2.predict(x_test)
print(f'预测值:{y_pred2}')
print(f'准确率:{model2.score(x_test,y_test)}')
print("-" * 34)

# 场景三:采用网格搜索,交叉验证
model3 = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV
param_grid = {
    "n_estimators":list(range(20,100,10)),
    "max_depth":list(range(3,10,1)),
    "min_samples_leaf":list(range(3,10,1))
}
grid_search = GridSearchCV(model3, param_grid=param_grid, cv=5)
grid_search.fit(x_train,y_train)
grid_search.predict(x_test)

print(f'准确率:{grid_search.score(x_test,y_test)}')
print(f'最佳参数:{grid_search.best_params_}')
print(f'最佳结果:{grid_search.best_score_}')
print(f'最佳估计器:{grid_search.best_estimator_}')
