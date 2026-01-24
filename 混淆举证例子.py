"""
    演示:混淆举证
        混淆举证:
            概述:用于展示真实值和预测值之间的 正例,反例的情况

            默认:会用分类少的样本作为正例

"""
import pandas as pd
from  sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

# 定义数据
y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性','良性','良性','良性',]

# 定义标签 正例 反例 默认是默认的分类少样本作为正例
lable = ['恶性', '良性']
df_lable = ['恶性(正例)', '良性(反例)']

# 定义预测结果
# 预测对了3个恶性,4个良性
y_pre_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性','良性','良性','良性']

# 把上述的结果转换为混淆矩阵g
cm_A = confusion_matrix(y_train, y_pre_A, labels=lable)
print(cm_A)

# 将混淆矩阵转换为 dataframe
df_A = pd.DataFrame(cm_A, index=df_lable, columns=df_lable)
print(df_A)

y_pre_b = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性','恶性', '恶性', '恶性']
cm_b = confusion_matrix(y_train, y_pre_b, labels=lable)
df_B = pd.DataFrame(cm_b, index=df_lable, columns=df_lable)
print(df_B)

# 打印y_preA 和y_preB的精确率、召回率、F1
print(f"y_preA的精确率:{precision_score(y_train, y_pre_A, pos_label='恶性')}")
print(f"y_preB的精确率:{precision_score(y_train, y_pre_b, pos_label='恶性')}")

print(f"y_preA的召回率:{recall_score(y_train, y_pre_A, pos_label='恶性')}")
print(f"y_preB的召回率:{recall_score(y_train, y_pre_b, pos_label='恶性')}")

print(f"y_preA的F1:{f1_score(y_train, y_pre_A, pos_label='恶性')}")
print(f"y_preB的F1:{f1_score(y_train, y_pre_b, pos_label='恶性')}")



