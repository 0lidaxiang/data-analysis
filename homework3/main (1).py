
# coding: utf-8

# In[95]:


import pandas as pd
import csv
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[ ]:


df = pd.read_excel('./data.xlsx')
df.head()


# In[48]:


df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]


# In[81]:


basket = (df[df['Country'] =="United Kingdom"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


# In[82]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket.drop('POSTAGE', inplace=True, axis=1)


# In[83]:


df['Description'].shape


# In[84]:


# basket[0:1].shape
# print(basket[0:1]["ACRYLIC JEWEL ANGEL,PINK"])
# print()
# for i in basket[0:1][:40]:
#     print(i)


# In[88]:


frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)


# In[100]:


# support  min threshold(0.01)，confidence  min threshold(0.5)
rules1 = association_rules(frequent_itemsets, metric="support", min_threshold= 0.01)
rules1.head()


# In[101]:


rules2 = association_rules(frequent_itemsets, metric="confidence", min_threshold= 0.5)
rules2.head()


# In[105]:


result1 = rules2[ (rules2['support'] >= 0.01) &
       (rules2['confidence'] >= 0.5) ]
print(result1.shape)
result1.head()


# In[177]:


result2 = result1

# test_res_1 = result2[:1]["antecedants"].values[0]
# slipt_str = test_res_1.split("{'")[1].split("'}")[0]

# teststr = "60 CAKE CASES DOLLY GIRL DESIGN"

# print( slipt_str, teststr, "\n",  slipt_str == teststr, "\n")
# print(result1["consequents"][:1])


# In[200]:


# create the result csv
import numpy as np
submit_data = pd.read_csv('submit.csv')
print(submit_data.shape)
submit_data.head()

print(submit_data["Association Rule antecedants"][:3][0])
print(submit_data["Association Rule antecedants"][:3])


index = 1
final_result = np.zeros(3055)
print(type(result2), result2[:3], result2.shape)


for ant_res2,con_res2 in zip(result2["antecedants"], result2["consequents"]):
#     for sub in submit_data:
        print("\n\n res2", type(ant_res2), ant_res2)
        print("\n\n res2", type(con_res2), con_res2 )
#         test_res_1 = res2["antecedants"].values[0]
#         slipt_str1 = test_res_1.split("{'")[1].split("'}")[0]
        
#         test_res_2 = res2["consequents"].values[0]
#         slipt_str2 = test_res_1.split("{'")[1].split("'}")[0]
        
#         for sub1 in sub["Association Rule antecedants"]:
#             if sub1 == slipt_str1:
#                 for sub2 in sub["Association Rule consequents"]:
#                     if sub2 == slipt_str2: 
#                         final_result[index] = 1
#                         print("1")
        index += 1
        


# In[20]:


basket2 = (df[df['Country'] =="Germany"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets2 = basket2.applymap(encode_units)
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

rules2[ (rules2['lift'] >= 4) &
        (rules2['confidence'] >= 0.5)]


# In[23]:




