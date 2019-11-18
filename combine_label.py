import pandas as pd
import os
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

num = "4"
a_file = "Train_DataSet_Label.csv"
b_file = "./model_" + num + "/predict.csv"
adf = pd.read_csv(a_file)
bdf = pd.read_csv(b_file)

bdf['label'] = 0

for i, val in enumerate(bdf.values):
    bdf.iloc[i, 4] = list(adf[adf['id'] == val[0]]['label'])[0]

bdf.to_csv(b_file,encoding='utf-8',index=False)
   
