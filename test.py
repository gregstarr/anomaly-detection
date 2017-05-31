#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:42:25 2017

@author: greg
"""

import numpy as np
from ranksvm_k import ranksvm_k
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from gregAD import knn_score
from scipy.io import loadmat

dic = np.load('arrays.npz')
l = len(dic['y'])
th = 1.0e-6
A = csr_matrix((dic['d'],(dic['row'],dic['col'])))
beta,asv = ranksvm_k(dic['K'],A,dic['C'],prec=1.0e-4)
yy = np.dot(dic['K'][np.abs(beta[:,0])>th,:].T,beta[np.abs(beta)>th,None])
R_rank = np.empty_like(dic['y'])
for i in range(len(dic['y'])):
    R_rank[i] = np.sum(yy[i]>yy)/l
    
train_G = knn_score(dic['K']) #simmilarity metric
R_knn = np.empty_like(dic['y']) #ranking metric [0,1)
for i in range(l):
    R_knn[i] = np.sum(train_G[i]>train_G)/l
        

alphas = np.arange(0,1,.01)
fa1 = np.empty(100)
tp1 = np.empty(100)
fa2 = np.empty(100)
tp2 = np.empty(100)

for i,a in enumerate(alphas):
    rank_class = np.sign(R_rank-a)
    fa1[i] = np.sum(rank_class[dic['y']==1] == -1)/np.sum(dic['y']==1)
    tp1[i] = np.sum(rank_class[dic['y']==-1] == -1)/np.sum(dic['y']==-1)
    knn_class = np.sign(R_knn-a)
    fa2[i] = np.sum(knn_class[dic['y']==1] == -1)/np.sum(dic['y']==1)
    tp2[i] = np.sum(knn_class[dic['y']==-1] == -1)/np.sum(dic['y']==-1)
    

mat = loadmat('/home/greg/Documents/MATLAB/PrimalRankSVM/fatp.mat')

plt.figure()
plt.plot(fa1,tp1,label='RankSVM')
plt.plot(fa2,tp2,label='KNN')
plt.plot(mat['PRank_FA'].T,mat['PRank_DET'].T,label='RankSVM - Matlab')
plt.plot(mat['knn_FA'].T,mat['knn_DET'].T,label='KNN - Matlab')
plt.legend()