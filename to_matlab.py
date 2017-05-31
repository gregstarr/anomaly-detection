#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:45:51 2017

@author: greg
"""

from gregAD import make_dset, make_kmat, knn_score
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import savemat


l = 200

    
train_data,train_y = make_dset(l,ap=.02)
train_k = make_kmat(train_data)
train_G = knn_score(train_k) #simmilarity metric
R = np.empty_like(train_y) #ranking metric [0,1)
for i in range(l):
    R[i] = np.sum(train_G[i]>train_G)/l
         
row = []
col = []
d = []
pairs = []
for i in range(l):
    for j in range(l):
        if R[i] > R[j]:
            row.append(len(col)//2)
            row.append(len(col)//2)
            col.append(i)
            col.append(j)
            d.append(1)
            d.append(-1)
            pairs.append([i,j])
            
row = np.array(row,dtype=float)
col = np.array(col,dtype=float)
d = np.array(d,dtype=float)

        
A = csr_matrix((d,(row,col)))
C = np.ones(A.shape[0])

savemat('/home/greg/Documents/MATLAB/PrimalRankSVM/arrays2.mat',
        {'data'    :   train_data,
         'y'       :   train_y,
         'K'       :   train_k,
         'row'     :   row,
         'col'     :   col,
         'd'       :   d,
         'C'       :   C})
    
np.savez('arrays',data=train_data,y=train_y,K=train_k,row=row,col=col,d=d,C=C)