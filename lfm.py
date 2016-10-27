import os
import numpy as np
import pandas as pd
import math  
from numba import jit
from datetime import datetime 
import mpmath as mp
import sys
import codecs
  
train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test.csv')
u_max = train['uid'].max()+1
i_max = train['iid'].max()+1
u_ms = train.groupby('uid')['score'].mean()
i_ms = train.groupby('iid')['score'].mean()
train_len = len(train)

def learning_lfm(n,alpha,lambd):
    if os.path.exists('./p-pian.npy'):
        sys.stdout.write('loading... \n')
        p = np.load('./p-pian.npy')
        q = np.load('./q-pian.npy')
    else:
        p = np.random.rand(u_max,5)*3
        q = np.random.rand(i_max,5)
    list_result(predict(p ,q), 'lfm-pian.csv')
    for step in range(n):
        error = 0.0
        for i in range(train_len):
            x = train.values[i]
            uid = x[0]
            iid = x[1]
            rui = x[2]
            pui = (u_ms[uid]+i_ms[iid])+sum(p[uid]*q[iid])
            eui = rui-pui
            error += eui
            for k in range(5):
                p[uid][k] += alpha*(q[iid][k]*eui-lambd*p[uid][k])
                q[iid][k] += alpha*(p[uid][k]*eui-lambd*q[iid][k])
                if p[uid][k]> 5:
                    p[uid][k] = 5
                if p[uid][k]<-5:
                    p[uid][k] = -5
                if q[iid][k]< 0:
                    q[iid][k]=0
                if q[iid][k]>1:
                    q[iid][k]=1
        np.save('p-pian.npy', p)
        np.save('q-pian.npy', q)
        sys.stdout.write("Train Setp: %d/%d, LR: %f, TE: %f \n" % (step, n , alpha, error))
        list_result(predict(p ,q), 'lfm-pian.csv')
        alpha = alpha * 0.9
    return p,q

def predict(p, q):
    n = len(test)
    s = []
    users = test['uid']
    items = test['iid']

    count = 0
    for i in range(n):
        x = test.values[i]
        uid = x[0]
        iid = x[1]
        tmp = (u_ms[uid]+i_ms[iid])+sum(p[uid]*q[iid])
        if tmp >= 5.0:
            count += 1 
            s.append(i_ms[test['iid'][i]])
        else:
            s.append(tmp)
    sys.stdout.write("Test Step Error : %d / %d \n\n" % (count, n))
    return s
    
def list_result(data, filename):
    result_file = codecs.open('../result/' + filename, 'w', 'utf-8')
    result_file.write('score\n')
    for socre in data:
        result_file.write(str(socre) + '\n')
    result_file.close()
       
start = datetime.now()
p,q = learning_lfm(50,0.09,0.001)
stop = datetime.now()
print(stop-start)
