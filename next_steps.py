#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:45:51 2017

@author: greg
"""

from gregAD import make_dset, make_kmat, knn_score
from ranksvm_k import ranksvm_k
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import itertools

TRIALS = 2

l = 200
fa1 = np.empty((100,TRIALS))
tp1 = np.empty((100,TRIALS))
fa2 = np.empty((100,TRIALS))
tp2 = np.empty((100,TRIALS))
alphas = np.arange(0,1,.01)

for trial in range(TRIALS):
    
    #Create training data
    train_data,train_y = make_dset(l,ap=.01)
    #assemble kernel matrix
    print('assembling training kernel matrix...')
    train_k = make_kmat(train_data)
    #determine knn scores for training data, predict training anomalies
    train_G = knn_score(train_k) #simmilarity metric
    R = np.empty_like(train_y) #ranking metric [0,1)
    for i in range(l):
        R[i] = np.sum(train_G[i]>train_G)/l
    
        
    #test various C values to find best ranker    
    idx = np.arange(l)
    np.random.shuffle(idx)
    KFOLD = 4
    th = 1.0e-6
    cs = np.logspace(-3,3,20)
    L = np.zeros_like(cs)
    
    for fold in range(KFOLD):
        idx_tr = idx[~np.logical_and(idx>=fold*l/KFOLD,idx<(fold+1)*l/KFOLD)]
        idx_te = idx[np.logical_and(idx>=fold*l/KFOLD,idx<(fold+1)*l/KFOLD)]
        k_tr = train_k[idx_tr][:,idx_tr]
        k_te = train_k[idx_tr][:,idx_te]
            
        #create preference matrix A
        row = []
        col = []
        d = []
        pairs = []
        for i in range(len(idx_tr)):
            for j in range(len(idx_tr)):
                if R[idx_tr[i]] > R[idx_tr[j]]:
                    row.append(len(col)//2)
                    row.append(len(col)//2)
                    col.append(i)
                    col.append(j)
                    d.append(1)
                    d.append(-1)
                    pairs.append([i,j])
                
        A = csr_matrix((np.array(d,dtype=np.int8), (np.array(row),np.array(col))),dtype=np.int8)
    
        for i,c in enumerate(cs):
            print('C: {}'.format(c))
            C = np.ones(A.shape[0])*c
            beta,asv = ranksvm_k(k_tr,A,C)
            print()
            yy = np.dot(k_tr[np.abs(beta[:,0])>th,:].T,beta[np.abs(beta)>th,None])
            yyt = np.dot(k_te[np.abs(beta[:,0])>th,:].T,beta[np.abs(beta)>th,None])
            for a,b in itertools.product(range(len(idx_te)),repeat=2):
                if yyt[a] < yyt[b] and R[idx_te[a]] > R[idx_te[b]]:
                    L[i] += 1
                    
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
            
    A = csr_matrix((np.array(d), (np.array(row),np.array(col))))
    C = np.ones(A.shape[0])*cs[np.argmin(L)]
    beta,asv = ranksvm_k(train_k,A,C,prec=1.0e-5)
    yy = np.dot(train_k[np.abs(beta[:,0])>th,:].T,beta[np.abs(beta)>th,None])
    
    test_data,test_y = make_dset(l,ap=.2)
    test_k = make_kmat(train_data,test_data)
    
    yyt = np.dot(test_k[np.abs(beta[:,0])>th,:].T,beta[np.abs(beta)>th,None])
    
    R_rank = np.empty_like(R)
    for i in range(l):
        R_rank[i] = np.sum(yyt[i]>yy)/l
    test_G = knn_score(test_k)
    R_knn = np.array([np.sum(q>train_G)/l for q in test_G])
        
    for i,a in enumerate(alphas):
        knn_class = np.sign(R_knn-a)
        fa1[i,trial] = np.sum(np.logical_and(knn_class==-1,test_y==1))/np.sum(test_y==1)
        tp1[i,trial] = np.sum(np.logical_and(knn_class==-1,test_y==-1))/np.sum(test_y==-1)    
    for i,a in enumerate(alphas):
        rank_class = np.sign(R_rank-a)
        fa2[i,trial] = np.sum(np.logical_and(rank_class==-1,test_y==1))/np.sum(test_y==1)
        tp2[i,trial] = np.sum(np.logical_and(rank_class==-1,test_y==-1))/np.sum(test_y==-1)
        
fa1 = np.mean(fa1,1)
tp1 = np.mean(tp1,1)
fa2 = np.mean(fa2,1)
tp2 = np.mean(tp2,1)

plt.figure()
plt.plot(fa1,tp1)
plt.plot(fa2,tp2)
print('knn training AUC = {:.3f}'.format(np.trapz(tp1,fa1)))
print('RankSVM training AUC = {:.3f}'.format(np.trapz(tp2,fa2)))