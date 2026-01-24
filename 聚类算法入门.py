import matplotlib
matplotlib.use('TkAgg')  # 设置matplotlib后端为TkAgg，确保图形界面正常显示

from matplotlib import pyplot as plt  # 导入matplotlib绘图库
from sklearn.cluster import KMeans   # 导入K均值聚类算法
from sklearn.datasets import make_blobs  # 导入生成聚类样本数据的函数

# 生成聚类样本数据
# n_samples=1000: 生成1000个样本点
# n_features=2: 每个样本有2个特征维度
# centers=[[-1,-1],[0,0],[1,1],[2,2]]: 指定4个聚类中心坐标
# random_state=28: 随机种子，保证结果可复现
x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1],[0,0],[1,1],[2,2]], random_state=28)

# 绘制原始数据散点图
# x[:,0]: 所有样本的第一个特征值作为x轴坐标
# x[:,1]: 所有样本的第二个特征值作为y轴坐标
plt.scatter(x[:,0], x[:,1])
plt.show()  # 显示图像

# 创建KMeans模型实例
# n_clusters=4: 指定要聚成4个簇(cluster)
# random_state=28: 随机种子，保证聚类结果可复现
model = KMeans(n_clusters=4, random_state=28)

# 使用fit_predict方法对数据进行聚类
# 输入数据x，返回每个样本所属的聚类标签
y_pred = model.fit_predict(x)

# 使用聚类结果绘制散点图
# c=y_pred: 根据聚类标签y_pred给不同簇分配不同颜色
plt.scatter(x[:,0], x[:,1], c=y_pred)
plt.show()  # 显示聚类结果图
