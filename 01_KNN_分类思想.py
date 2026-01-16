from sklearn.neighbors import KNeighborsClassifier

# 创建模型(算法)对象
model = KNeighborsClassifier(n_neighbors=3)

# 准备训练集(x_train,y_train)

x_train = [[1, 3], [5, 15], [3, 8], [7, 4], [5, 5], [6, 6]]
y_train = [0, 1, 0, 1, 0, 1]

x_test = [[2, 2], [4, 7], [6, 10], [8, 5]]
result = model.fit(x_train, y_train)
print(result)

y_test = model.predict(x_test)
print(y_test)