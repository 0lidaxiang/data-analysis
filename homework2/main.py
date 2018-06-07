# coding: utf-8

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn  

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

dataset = pd.read_csv('HW2_pokemon.csv')
class_mapping = {label:idx for idx,label in enumerate(set(dataset['Body_Style']))}
dataset['Body_Style'] = dataset['Body_Style'].map(class_mapping)
print(dataset.shape)
dataset_num = dataset[['Total','HP','Attack','Defense','Sp_Atk','Sp_Def','Speed', 'Height_m', 'Weight_kg','Body_Style' ]]

ss = StandardScaler()
dataset_num = ss.fit_transform(dataset_num)
dataset_scaled  = dataset.copy()
dataset_scaled[['Total','HP','Attack','Defense','Sp_Atk','Sp_Def','Speed', 'Height_m', 'Weight_kg' ,'Body_Style']] = dataset_num

# n_clusters 分成多少群
kmeans = KMeans(n_clusters = 280, init = 'k-means++', random_state = 280)
y_kmeans = kmeans.fit_predict(dataset_num)

# 画出 K-means 的分类情况
dataset['y_kmeans'] = y_kmeans
seaborn.violinplot(x='y_kmeans', y='Sp_Atk', data=dataset)
plt.show()

print("y_kmeans : ", type(y_kmeans), y_kmeans.shape)
print(y_kmeans[:20])
test_dataset = pd.read_csv('subject.csv')
print(type(test_dataset), test_dataset.shape)

index = 0
res = []
# MIDDLE = 29
for td1,td2  in zip(test_dataset["0"], test_dataset["1"]):
    td1_i = int(td1[7:])
    td2_i = int(td2[7:])
    if y_kmeans[td1_i] == y_kmeans[td2_i]:
        res.append([index, 1])
    else:
        res.append([index, 0])
    index += 1

print(res[:10])

# 把預測的結果生成 kaggle要求的格式
# pair，值為0~999，第二欄取名：answer
res_csv_file_path = "result.csv"
with open(res_csv_file_path, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(('pair', 'answer'))
    ids = 0
    for val in res:
        writer.writerow((str(ids),val[1]))
        ids += 1
print("--------------- Program executes finished.")

