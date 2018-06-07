# Introduction of data analysis course of NTUST 2018

## Introduction

This course content includes that Association Analysis,Classification(Supervised Method:),Clustering(Unsupervised Method).

## Homework1

### Goal
The first operation is based on the basic information of the bank's customers to **predict whether the customer will finally settle the deposit**.

----------
作业一是根据银行客户的基本资料预测客户最后是否会办理定存。

### Introduction
There are a total of 16 attributes relating to clients in the training materials, including age, marital status, education level... (see training_data.csv for details), and 1 output attribute (yes or no to handle the deposit ) ).

The result is a CSV file. (A total of 4523 columns, 2 rows)
The first column is id, indicating the bank customer's number (Note! id starts at 0)
The second column is ans, indicating whether the customer will handle the deposit (if it is yes, please use 1; if it is no, please use 0).

-------------

训练资料里一共有16个有关客户的属性，包括年龄，婚姻状况，教育程度......（详细请见training_data.csv的栏位），以及1个输出属性（是或否会办理定存）。

结果是一个CSV文件。(一共有4523列，2行)
第一栏为id，表示银行客户的编号(注意! id从0开始)
第二栏为ans，表示客户是否会办理定存(如果是yes的话，请用1表示，如果是no的话，请用0表示)


### Attribute Introduction
+ age：年齡
+ job：工作類型
+ marital：婚姻狀況
+ education：教育程度
+ default：是否有未履行債務
+ balance：平均每年餘額
+ housing：是否有住房貸款
+ loan：是否有個人貸款嗎
+ contact：聯繫人通信類型
+ day：每個月的最後一個聯繫日
+ month：每年的最後一個聯繫月份
+ duration：上次聯繫持續時間(e.g.電話通話時間)
+ campaign：在此活動和此客戶中執行的聯繫數量
+ pdays：客戶最近一次與之前活動聯繫後經過的天數（-1表示之前未聯繫過客戶）
+ previous：此活動和此客戶端之前執行的聯繫數量  
+ poutcome：以前的營銷活動的結果（分類：“未知”，“其他”，“失敗”，“成功”）  
+ y： 客戶是否訂購了定期存款


## Homework2

### Goal  


This homework maybe modifyed from [this program](http://minimaxir.com/2016/08/pokemon-3d/). And add some more attributes. Our goal is to use clustering method to analysis which Pokémon is most same.

-------------

这个作业可能从[这个程序](http://minimaxir.com/2016/08/pokemon-3d/)修改而来， 并添加一些更多的属性。 我们的目标是使用聚类分析法 (K-means) 来分析哪个神奇宝贝最相同。

### Introduction
A [highly-upvoted post](https://www.reddit.com/r/dataisbeautiful/comments/4uumqs/gotta_plot_em_all_the_height_versus_weight_of/) on the Reddit subreddit '/r/dataisbeautiful' by  [/u/nvvknvvk](https://www.reddit.com/user/nvvknvvk) charts the Height vs. Weight of the original 151 Pokémon.

Anh Le of Duke University posted a cluster analysis of the original 151 Pokémon using principal component analysis (PCA), by compressing the 6 primary Pokémon stats into 2 dimensions.However, those visualizations think too small, and only on a small subset of Pokémon. This dataset can also be broken into three dimensions to analyze and research.

--------------

一篇 [高点赞数的帖子](https://www.reddit.com/r/dataisbeautiful/comments/4uumqs/gotta_plot_em_all_the_height_versus_weight_of/) 在Reddit subreddit'/ r / dataisbeautiful'中的[/ u / nvvknvvk](https：/ /www.reddit.com/user/nvvknvvk)图表中罗列了原始151神奇宝贝的身高与体重。

杜克大学的Anh Le使用主成分分析（PCA）发布了对原始151神奇宝贝的聚类分析，将6个主要神奇宝贝统计压缩到2个维度。但是，这些可视化想象太小，并且只能在一小部分神奇宝贝上。也可以将这些数据暴力破解成三个维度来分析研究。

## Homework3

### Goal
This is a data set from the retailer's transaction data. We want to analyze the common collocation choices when shopping in countries such as the United Kingdom and France.

---------

這是一個來自零售商的交易數據的資料集。我们要分析英国法国等国家居民在购物时的常见搭配选择。

### Introduction
Maybe i dont have time to do this homework.

### Attribute Introduction
+ InvoiceNo 是發票號碼
+ StockCode是商品代碼
+ Description是商品名稱
+ Quantity是購買數量
+ InvoiceDate是購買日期
+ UnitPrice是商品單價
+ CustomerID是客戶編號
+ Country是客戶所在的國家
+ 信用卡交易 改為"被取消的交易"

## result.csv Introduction
- 第一欄取名：index，值為0~3054
- 第二欄取名：label，值為0 or 1。
