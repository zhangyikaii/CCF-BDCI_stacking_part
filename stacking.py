# -*- coding: utf-8 -*-
import pandas as pd
import os
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

a_file = "./model_0/test.csv"
adf = pd.read_csv(a_file)
adf['label_0'] = 0
adf['label_1'] = 0
adf['label_2'] = 0

ensemble_num = 4
for i in [0, 1, 2, 4]:
    atmp = pd.read_csv('./model_{}/test.csv'.format(i))
    btmp = pd.read_csv('./model_{}/predict.csv'.format(i))
    
    adf['label_0'] += atmp['label_0'] / ensemble_num
    
    if i == 0:
        bdf = btmp
    else:
        bdf = pd.concat([bdf, btmp])
        print(len(bdf))
        
# print(len(adf), len(bdf))

one_hot = pd.get_dummies(bdf['label'])
bdf.drop(['label'], axis=1, inplace=True)
X_train = bdf.drop(['id'], axis=1).values
bdf = pd.concat([bdf,one_hot], axis=1)

# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=11))
Y_train = bdf.drop(['id', 'label_0', 'label_1', 'label_2'], axis=1).values

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, Y_train)

X_test = adf.drop(['id'], axis=1).values
Y_test = rf.predict(X_test)

result = np.argmax(Y_test, axis = 1)

import csv

redf = pd.read_csv('./submit_example.csv')
reId = redf.drop(['label'], axis=1).values
# 写入文件
csvFile = open('./result.csv','w', newline='', encoding='UTF-8') # 设置newline，否则两行之间会空一行
writer = csv.writer(csvFile)

writer.writerow(['id', 'label'])
for i in range(len(result)):
    if i % 50000 == 0:
        print(reId[i])
    writer.writerow([reId[i][0], int(result[i])])
    
csvFile.close()

