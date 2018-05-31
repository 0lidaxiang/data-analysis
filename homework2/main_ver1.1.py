
# coding: utf-8

# In[29]:


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn  

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[30]:


dataset = pd.read_csv('HW2_pokemon.csv')
print(dataset.shape)
dataset_num = dataset[['Total','HP','Attack','Defense','Sp_Atk','Sp_Def','Speed']]


# In[31]:


ss = StandardScaler()
dataset_num = ss.fit_transform(dataset_num)
dataset_scaled  = dataset.copy()
dataset_scaled[['Total','HP','Attack','Defense','Sp_Atk','Sp_Def','Speed']] = dataset_num

loss = []
for i in range(1, 41):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset_num)
    loss.append(kmeans.inertia_)
plt.plot(range(1, 41), loss)
# plt.title('The Elbow Method')
# plt.xlabel('clusters数目')
# plt.ylabel('WCSS')
plt.xticks(np.arange(1, 41, 1.0))
plt.grid( axis='x')
plt.show()


# In[32]:


# n_clusters 应该是 5~10之间比较好
kmeans = KMeans(n_clusters = 82, init = 'k-means++', random_state = 84)
y_kmeans = kmeans.fit_predict(dataset_num)

# 画出 Kmeans 的分类情况
dataset['y_kmeans'] = y_kmeans
seaborn.violinplot(x='y_kmeans', y='Total', data=dataset)
plt.show()


# In[33]:


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
# 后来发现这种判断不能用，会大幅降低判断的准确性        
#     if y_kmeans[td1_i] < MIDDLE and  y_kmeans[td2_i] < MIDDLE:
#         print(index, td1_i, td2_i, y_kmeans[td1_i] , y_kmeans[td2_i])
        res.append([index, 1])
#     elif y_kmeans[td1_i] >= MIDDLE and  y_kmeans[td2_i] >= MIDDLE:
#         res.append([index, 1])
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
print("---------------execute finished.")

