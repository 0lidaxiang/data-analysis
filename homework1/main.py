#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import csv
import math
import numpy as np
import pandas as pd

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Activation,Dense, Dropout
from datetime import datetime,timedelta


# 定義變量，包括資料夾路徑、參與ANN計算的屬性
TRAIN_FILE_PATH= './data/training_data.csv'
TEST_FILE_PATH= './data/testing_data.csv'

train_cols = ['age', 'job', 'marital','education', 'default', 'balance', 'housing', 'loan'
       , 'contact','day', 'month', 'duration', 'campaign', 'pdays' , 'previous', 'poutcome', "y"]
test_cols = ['age', 'job', 'marital','education', 'default', 'balance', 'housing', 'loan'
       , 'contact','day', 'month', 'duration', 'campaign', 'pdays' , 'previous', 'poutcome']

# 讀取csv中的資料
train_df = pd.read_csv(TRAIN_FILE_PATH)
test_df = pd.read_csv(TEST_FILE_PATH)
train_df = train_df[train_cols]
test_df = test_df[test_cols]


# 預處理資料的 function
# 把原始資料全部轉換成數字格式
def PreprocessData(raw_df , data_type):
    df = raw_df.drop(['default'], axis=1)
#     df = raw_df.drop(['month'], axis=1)
     
#     print(type(df['marital'][0]), df['marital'][0])
#     df['job'] = df['job'].map({"unknown" : 0, 'services': 1, 'admin.': 2, 'technician': 3,'blue-collar': 4, 
#                                'housemaid': 5, 'entrepreneur': 6, 'self-employed': 7, 'retired': 8,
#                                'management': 9, "student" : 10, "unemployed" : 11}).astype(int)
    df['marital'] = df['marital'].map({ 'divorced': 0 , 'single': 1, 'married': 2}).astype(int)
#     df['education'] = df['education'].map({ 'secondary': 0 , 'primary': 1, 'tertiary': 2, "unknown" :3}).astype(int)
#     df['balance'] = df['marital'].map({ 'divorced': 0 , 'single': 1, 'married.': 2}).astype(int)
    df['housing'] = df['housing'].map({ 'no': 0 , 'yes': 1, 'married.': 2}).astype(int)
    df['loan'] = df['loan'].map({ 'no': 0 , 'yes': 1, 'married.': 2}).astype(int)
    df['contact'] = df['contact'].map({ 'unknown': 0 , 'telephone': 1, 'cellular': 2}).astype(int)
#     df['day'] = df['marital'].map({ 'divorced': 0 , 'single': 1, 'married.': 2}).astype(int)
#     df['month'] = df['month'].map({ 'jan': 1, 'feb': 2, 'mar': 3 , 'apr': 4 , 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,'sep': 9, 'oct': 10 , 'nov': 11, 'dec': 12}).astype(int)
#     df['duration'] = df['marital'].map({ 'divorced': 0 , 'single': 1, 'married.': 2}).astype(int)
#     df['campaign'] = df['marital'].map({ 'divorced': 0 , 'single': 1, 'married.': 2}).astype(int)
#     df['pdays'] = df['marital'].map({ 'divorced': 0 , 'single': 1, 'married.': 2}).astype(int)
#     df['previous'] = df['marital'].map({ 'divorced': 0 , 'single': 1, 'married.': 2}).astype(int)
    df['poutcome'] = df['poutcome'].map({ 'failure': 0 , 'success': 1, 'unknown': 2, 'other': 3}).astype(int)
    df['duration'] = df['duration'] /100.0
    df['age'] = df['age'] /100.0
    
    if data_type == "train":
        df['y'] = df['y'].map({ 'no': 0 , 'yes': 1}).astype(int)
        df1 = df[df.y.isin([1])]
        print(len(df), len(df1)," Deposit percentage :" , len(df1)/ len(df))
        
        df0 = df[df.y.isin([0])].sample(frac=0.03) 
        newdf = df1.append(df0).sample(frac=1) 
        print(len(newdf), len(df1)," Deposit percentage :" , len(df1)/ len(newdf))
    else:
        df['y'] = 0

    x_OneHot_df = pd.get_dummies(data=df, columns = ["month", 'job', 'education'])
    ndarray = x_OneHot_df.values
    
    label = ndarray[:,14]  
    Features = ndarray[:,1:] 
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)
    return scaledFeatures, label

# 調用函數來預處理資料，轉換成數字格式 
train_result, train_label = PreprocessData(train_df, "train")
test_feature, test_label = PreprocessData(test_df , "test")


# 構建ANN模型，一層輸入一層輸出
model = Sequential()
model.add(Dense(units=13, input_dim= 40))
model.add(Activation('relu'))

# model.add(Dense(units=10))
# model.add(Activation('relu'))

# model.add(Dense(units=5))
# model.add(Activation('relu'))

model.add(Dense(units=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# 訓練 model
model.fit(x = train_result, y = train_label, epochs = 100, validation_split = 0.1, batch_size = 10, verbose = 1)
scores = model.evaluate(x = train_result, y = train_label, batch_size=5)
# 使用訓練好的 model 去預測測試資料
res = model.predict(test_feature, batch_size=5, verbose=0)
print(scores)


# 把預測的結果生成 kaggle要求的格式
res_csv_file_path = "result.csv"
with open(res_csv_file_path, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(('id', 'ans'))
    ids = 0
    for val in res:
        if val[0] > 0.5:
            writer.writerow((str(ids), str(1)))
        else:
            writer.writerow((str(ids), str(0)))
        ids += 1

# 計算多少人會辦理定存的概率
print("Count how many people will Deposit: ")
with open(res_csv_file_path) as csvfile:
    counts = 0
    train_csv = csv.DictReader(csvfile)
    for row in train_csv:
        if row["ans"]  == "1":
#             print(row["id"])
            counts += 1
print(counts, "test predict result Deposit percentage :" , counts / 4252)


# 計算 model 輸出結果的分布
# 以便下次訓練前來調整 model
counts1 = 0
counts2 = 0
counts3 = 0
counts4 = 0
counts5 = 0
counts6 = 0
for val in res:
    if val[0] > 0.2:
        counts1+=1
    if val[0] > 0.3:
        counts2+=1
    if val[0] > 0.4:
        counts3+=1
    if val[0] > 0.5:
        counts4+=1
    if val[0] > 0.6:
        counts5+=1
    if val[0] > 0.7:
        counts6+=1
print(counts1, "Deposit percentage 0.2 :" , counts1 / 4252)
print(counts2, "Deposit percentage 0.3 :" , counts2 / 4252)
print(counts3, "Deposit percentage 0.4 :" , counts3 / 4252)
print(counts4, "Deposit percentage 0.5 :" , counts4 / 4252)
print(counts5, "Deposit percentage 0.6 :" , counts5 / 4252)
print(counts6, "Deposit percentage 0.7 :" , counts6 / 4252)

