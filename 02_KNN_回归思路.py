from sklearn.neighbors import KNeighborsRegressor

# 创建模型(算法)对象
model = KNeighborsRegressor(n_neighbors=3)

# 准备训练集(x_train,y_train)

x_train = [[1, 3], [5, 15], [3, 8], [7, 4], [5, 5], [6, 6]]
y_train = [0.1, 0.3, 1.2, 1.4, 1.5, 1.6]

x_test = [[3, 2], [8, 7], [6, 10], [5, 5], [4, 4], [2, 2]]
model.fit(x_train, y_train)
print(model)

y_test = model.predict(x_test)
print(y_test)
