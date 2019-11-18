import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prefix1", default=None, type=str, required=True)
parser.add_argument("--prefix2", default=None, type=str, required=True)
parser.add_argument("--prefix3", default=None, type=str, required=True)
parser.add_argument("--out_path", default=None, type=str, required=True)
args = parser.parse_args()

aWei = 1.1
bWei = 0.9
cWei = 1.1

k=30
df=pd.read_csv('data/submit_example.csv')
df['0']=0
df['1']=0
df['2']=0
for i in range(10):
    temp=pd.read_csv('{}{}/test.csv'.format(args.prefix1,i))
    df['0']+=temp['label_0']/k * aWei
    df['1']+=temp['label_1']/k * aWei
    df['2']+=temp['label_2']/k * aWei

for i in range(10):
    temp=pd.read_csv('{}{}/test.csv'.format(args.prefix2,i))
    df['0']+=temp['label_0']/k * bWei
    df['1']+=temp['label_1']/k * bWei
    df['2']+=temp['label_2']/k * bWei

for i in range(10):
    temp=pd.read_csv('{}{}/test.csv'.format(args.prefix3,i))
    df['0']+=temp['label_0']/k * cWei
    df['1']+=temp['label_1']/k * cWei
    df['2']+=temp['label_2']/k * cWei

df['label']=np.argmax(df[['0','1','2']].values,-1)
df[['id','label']].to_csv(args.out_path,index=False)
