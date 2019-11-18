import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


BATCH_SIZE=512
EPOCHS=20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

a_file = "./result/model_0/test.csv"
adf = pd.read_csv(a_file)
adf['label_0'] = 0
adf['label_1'] = 0
adf['label_2'] = 0

ensemble_num = 4
for i in [0, 1, 2, 4]:
    atmp = pd.read_csv('./result/model_{}/test.csv'.format(i))
    btmp = pd.read_csv('./result/model_{}/predict.csv'.format(i))
    
    if i == 0:
        X_train = [[list(i)] for i in btmp.drop(['id', 'label'], axis=1).values]
        X_test = [[list(i)] for i in atmp.drop(['id', 'label'], axis=1).values]
        print(len(X_train))
    else:
        btemp = [list(i) for i in btmp.drop(['id', 'label'], axis=1).values]
        for i, val in enumerate(btemp):
            X_train[i].append(val)
            
        atemp = [list(i) for i in atmp.drop(['id', 'label'], axis=1).values]
        for i, val in enumerate(atemp):
            X_test[i].append(val)

x_train = []
for i, val in enumerate(X_train):
    tmp = []
    for j in val:
        for k in j:
            tmp.append(k)
    x_train.append(tmp)
print(x_train[0])

x_test = []
for i, val in enumerate(X_test):
    tmp = []
    for j in val:
        for k in j:
            tmp.append(k)
    x_test.append(tmp)
print(x_test[0])

X_train = np.array(x_train)
X_test = np.array(x_test)
print(X_train[0])
print(X_test[0])
# print(len(adf), len(bdf))

bdf = pd.read_csv('./result/model_0/predict.csv')
Y_train = pd.get_dummies(bdf['label']).values

print(Y_train)
print("=" * 80)
print(len(X_train))
print(len(Y_train))

# rf.fit(X_train, Y_train)
# Y_test = rf.predict(X_test)
# result = np.argmax(Y_test, axis = 1)

# import tensorflow as tf
import torch
import torch.utils.data as Data

torch_dataset = Data.TensorDataset(torch.from_numpy(X_train).double(), 
                                   torch.from_numpy(Y_train).double())
train_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


import torch.nn as nn
from math import sqrt
import torch

class NNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(1, 12))
        nn.init.uniform_(self.W, 0, 1)
        
    def forward(self,x):

        tmp = self.W.expand(x.size(0),x.size(1)).mul(x)
        # print('tmp', tmp, '\n', tmp.size(), '\nw', self.W, '\nx', x, '\n', x.size())
        label_0 = tmp[:, 0]+tmp[:, 3]+tmp[:, 6]
        label_1 = tmp[:, 1]+tmp[:, 4]+tmp[:, 7]
        label_2 = tmp[:, 2]+tmp[:, 5]+tmp[:, 8]
        # print(label_0.size(), label_1.size(),label_2.size())
        array = torch.cat([label_0.expand(1, label_0.size(0)), label_1.expand(1, label_1.size(0)), label_2.expand(1, label_2.size(0))], dim=0)
        array = torch.t(array)
        # print('array',array.size())
        
        # sumLabel = np.exp(label_0) + np.exp(label_1) + np.exp(label_2);
        out = F.log_softmax(array)
        print("=" * 80)
        # print('out', out.size())
        return out

model = NNmodel().to(DEVICE)
print(model.parameters())
optimizer = optim.Adam(model.parameters())


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())
        print(output.size())
        print(target.size())
        print(target.long())
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
print(DEVICE)
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    # test(model, DEVICE, test_loader)
