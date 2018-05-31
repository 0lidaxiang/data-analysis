
# coding: utf-8

# In[ ]:


# 第一欄取名：pair，值為0~999
# 第二欄取名：answer，值為0 or 1
# 這次Public data 70% Private data 30%
# 作業繳交格式如下圖 : 在subject.csv中判斷兩個pokemon是否為相同屬性，是為1，不是為0


# In[3]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')
sns.set(style='white', font_scale=0.9)
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.color_palette(flatui)

np.set_printoptions(threshold=np.nan)
pd.set_option("display.max_columns",100)


# In[12]:


dataset = pd.read_csv('HW2_pokemon.csv')


# In[44]:


print(dataset.shape)
dataset.head(10)


# In[41]:


# dataset.info()


# In[16]:


dataset_num = dataset[['Total','HP','Attack','Defense','Sp_Atk','Sp_Def','Speed']]


# In[17]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
dataset_num = sc_X.fit_transform(dataset_num)


# In[19]:


dataset_scaled  = dataset.copy()
dataset_scaled[['Total','HP','Attack','Defense','Sp_Atk','Sp_Def','Speed']] = dataset_num


# In[20]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 41):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset_num)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 41), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(np.arange(1, 41, 1.0))
plt.grid(which='major', axis='x')
plt.show()


# In[145]:


# n_clusters 应该是 5~10之间比较好
kmeans = KMeans(n_clusters = 82, init = 'k-means++', random_state = 84)
y_kmeans = kmeans.fit_predict(dataset_num)

print("y_kmeans : ", type(y_kmeans), y_kmeans.shape)
print(y_kmeans[:30])
test_dataset = pd.read_csv('subject.csv')
print(type(test_dataset), test_dataset.shape)

index = 0
res = []
MIDDLE = 29
for td1,td2  in zip(test_dataset["0"], test_dataset["1"]):
    td1_i = int(td1[7:])
    td2_i = int(td2[7:])
    if y_kmeans[td1_i] == y_kmeans[td2_i]:
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
import csv
res_csv_file_path = "result.csv"
with open(res_csv_file_path, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(('pair', 'answer'))
    ids = 0
    for val in res:
        writer.writerow((str(ids),val[1]))
        ids += 1
print("---------------execute finished.")


# In[22]:


dataset['y_kmeans'] = y_kmeans


# In[23]:


dataset.head()


# In[59]:


sns.violinplot(x='y_kmeans', y='Total', data=dataset)
plt.show()


# In[25]:


sns.violinplot(x='y_kmeans', y='Attack', data=dataset)
plt.show()


# In[26]:


sns.violinplot(x='y_kmeans', y='Defense', data=dataset)
plt.show()


# In[27]:


dataset.sort_values('Defense', axis=0, ascending=False).head(10)


# In[28]:


sns.violinplot(x='y_kmeans', y='Speed', data=dataset)
plt.show()


# In[29]:


dataset.sort_values('Speed', axis=0, ascending=False).head(15)


# In[31]:


sns.violinplot(x='y_kmeans', y='Sp_Atk', data=dataset)
plt.show()


# In[32]:


sns.violinplot(x='y_kmeans', y='Sp_Def', data=dataset)
plt.show()


# In[34]:


dataset.sort_values('Sp_Def', axis=0, ascending=False).head(10)


# In[38]:


# Clusters by Height_m，Weight_kg

#Get counts by type and cluster
#We need to merge the two columns Type 1 and Type 2 together the type can appear in either column
data_pct_1 = dataset.groupby(['Height_m', 'y_kmeans'])['Name'].count().to_frame().reset_index()
data_pct_1.columns = ['Type', 'y_kmeans', 'count_1']

data_pct_2 = dataset.groupby(['Weight_kg', 'y_kmeans'])['Name'].count().to_frame().reset_index()
data_pct_2.columns = ['Type', 'y_kmeans', 'count_2']

data_pct = data_pct_1.merge(data_pct_2, how='outer',
                            left_on=['Type', 'y_kmeans'],
                            right_on=['Type', 'y_kmeans'])

data_pct.fillna(0, inplace=True)
data_pct['count'] = data_pct['count_1'] + data_pct['count_2']

#Get counts by type
data_pct_Total = data_pct.groupby(['Type']).sum()['count'].reset_index()
data_pct_Total.columns = ['Type', 'count_total']

#Merge two dataframes and create percentage column
data_pct = data_pct.merge(right=data_pct_Total, 
                                    how='inner',
                                    left_on='Type',
                                    right_on='Type')

data_pct['pct'] = data_pct['count'] / data_pct['count_total']

#Create Graph
sns.barplot(x='Type', y='pct', data=data_pct, estimator=sum, ci=None, color='#34495e', label='4')
sns.barplot(x='Type', y='pct', data=data_pct[data_pct['y_kmeans'] <= 3], 
            estimator=sum, ci=None, color='#e74c3c', label='3') 
sns.barplot(x='Type', y='pct', data=data_pct[data_pct['y_kmeans'] <= 2], 
            estimator=sum, ci=None, color='#95a5a6', label='2') 
sns.barplot(x='Type', y='pct', data=data_pct[data_pct['y_kmeans'] <= 1], 
            estimator=sum, ci=None, color='#3498db', label='1') 
sns.barplot(x='Type', y='pct', data=data_pct[data_pct['y_kmeans'] == 0], 
            estimator=sum, ci=None, color='#9b59b6', label='0') 

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.xticks(rotation=90)
plt.ylabel('Percentage')
plt.tight_layout()
plt.show()


# In[40]:


#Clusters by Body_Style


#Get counts by body_Style and cluster
#We need to merge the two columns Type 1 and Type 2 together the type can appear in either column
data_pct = dataset.groupby(['Body_Style', 'y_kmeans'])['Name'].count().to_frame().reset_index()
data_pct.columns = ['Body_Style', 'y_kmeans', 'count']

#Get counts by type
data_pct_Total = data_pct.groupby(['Body_Style']).sum()['count'].reset_index()
data_pct_Total.columns = ['Body_Style', 'count_total']

#Merge two dataframes and create percentage column
data_pct = data_pct.merge(right=data_pct_Total, 
                                    how='inner',
                                    left_on='Body_Style',
                                    right_on='Body_Style')

data_pct['pct'] = data_pct['count'] / data_pct['count_total']

#Create Graph
sns.barplot(x='Body_Style', y='pct', data=data_pct, estimator=sum, ci=None, color='#34495e', label='4')
sns.barplot(x='Body_Style', y='pct', data=data_pct[data_pct['y_kmeans'] <= 3], 
            estimator=sum, ci=None, color='#e74c3c', label='3') 
sns.barplot(x='Body_Style', y='pct', data=data_pct[data_pct['y_kmeans'] <= 2], 
            estimator=sum, ci=None, color='#95a5a6', label='2') 
sns.barplot(x='Body_Style', y='pct', data=data_pct[data_pct['y_kmeans'] <= 1], 
            estimator=sum, ci=None, color='#3498db', label='1') 
sns.barplot(x='Body_Style', y='pct', data=data_pct[data_pct['y_kmeans'] == 0], 
            estimator=sum, ci=None, color='#9b59b6', label='0') 

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.xticks(rotation=90)
plt.ylabel('Percentage')
plt.tight_layout()
plt.show()

