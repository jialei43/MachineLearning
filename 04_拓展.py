import matplotlib.pyplot as plt
import numpy as np

# 为什么需要排序?
# 解释: plt.plot()绘制线条时,matplotlib会按照数据点的顺序依次连接它们.
# 如果x数组不是有序的,绘制线条会在这些点之间来回跳跃.形成多条交叉线.
#
# x = [3, 1, 2]
# y = [9, 1, 4]

# 错误一:
# plt.plot(x,y)
# plt.show()

# # 错误2:分析:  是排序了.x轴排序了.但是对应的y轴有问题.
# plt.plot(np.sort(x), y)
# plt.show()

# np.argsort(x)

x=[3,1,4,2,5]
print(np.argsort(x))  # 返回是 [1 2 0 ] ->因为排序后的索引巡视:
