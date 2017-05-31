#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:45:51 2017

This script is the first run at using the ranking SVM to speed up testing
stage, right now it runs 4 fold cross validation
"""
import os
if 'PYTHONSTARTUP' in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')
from gregAD import make_dset, make_kmat, knn_score
from ranksvm_k import ranksvm_k
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import itertools
from time import time

def create_prefmat(R,idx=None,dtype=np.int8):
    if idx is None:
        idx = np.arange(len(R))
    row = []
    col = []
    d = []
    weights = []
    for i in range(len(idx)):
        for j in range(len(idx)):
            if R[idx[i]] > R[idx[j]]:
                row.append(len(col)//2)
                row.append(len(col)//2)
                col.append(i)
                col.append(j)
                d.append(1)
                d.append(-1)
                weights.append(R[idx[i]] - R[idx[j]])
            
    return csr_matrix((np.array(d,dtype=dtype),(np.array(row),np.array(col))),
                      dtype=dtype), np.array(weights)

    

if __name__ == "__main__":
    
    from argparse import ArgumentParser
    descr = "Anomaly Detection"

    p = ArgumentParser(description=descr)
    p.add_argument('-l','--n_groups',type=int,help='number of groups',
                   default=1000)
    p.add_argument('-p','--plots',action='store_true',help='plots',
                   default=False)
    p.add_argument('-t','--trials',type=int,help='Overall number of trials '
                   'for everythin',default=1)
    p.add_argument('-k','--folds',type=int,help='K-fold cross validation',
                   default=4)
    p.add_argument('-u','--train_anom',type=float,help='training anomalies '
                   'percentage',default=.02)
    p.add_argument('-v','--test_anom',type=float,help='testing anomalies '
                   'percentage',default=.3)
    p = p.parse_args()
    
    TRIALS = p.trials
    
    l = p.n_groups
    
    fa1 = np.empty((100,TRIALS))
    tp1 = np.empty((100,TRIALS))
    fa2 = np.empty((100,TRIALS))
    tp2 = np.empty((100,TRIALS))
    alphas = np.arange(0,1,.01)
    train_ap = p.train_anom
    test_ap = p.test_anom
    
    tic = time()
    for trial in range(TRIALS):
        print('\n--------------TRIAL {}---------------'.format(trial+1))
        
        #Create training data
        train_data,train_y = make_dset(l,ap=train_ap)
        #assemble kernel matrix
        print('assembling training kernel matrix...')
        train_k = make_kmat(train_data)
        #determine knn scores for training data, ranking score
        train_G = knn_score(train_k) #simmilarity metric
        R = np.empty_like(train_y) #ranking score [0,1)
        for i in range(l):
            R[i] = np.sum(train_G[i]>train_G)/l
        
        #test various C values to find best ranker based on 
        #pairwise disagreement loss
        print('\nK FOLD CROSS VALIDATION')
        tic = time()
        idx = np.arange(l)
        np.random.shuffle(idx)
        KFOLD = p.folds
        th = 1.0e-6
        cs = np.logspace(-3,3,20)   #C's to test
        L = np.zeros_like(cs)   #pairwise disagreement loss
        for fold in range(KFOLD):
            print('\nFOLD: {}'.format(fold+1))
            #training and testing portions
            idx_tr = idx[~np.logical_and(idx>=fold*l/KFOLD,idx<(fold+1)*l/KFOLD)]
            idx_te = idx[np.logical_and(idx>=fold*l/KFOLD,idx<(fold+1)*l/KFOLD)]
            k_tr = train_k[idx_tr][:,idx_tr]
            k_te = train_k[idx_tr][:,idx_te]
            #create preference matrix A
            A,weights = create_prefmat(train_G,idx=idx_tr)
            #try the different C's
            #optimal C has lowest L
            for i,c in enumerate(cs):
                print('C: {}'.format(c))
                C = weights*c
                beta,asv = ranksvm_k(k_tr,A,C)
                yy = np.dot(k_tr[np.abs(beta[:,0])>th,:].T,beta[np.abs(beta)>th,None])
                yyt = np.dot(k_te[np.abs(beta[:,0])>th,:].T,beta[np.abs(beta)>th,None])
                for a,b in itertools.product(range(len(idx_te)),repeat=2):
                    if yyt[a] < yyt[b] and R[idx_te[a]] > R[idx_te[b]]:
                        L[i] += 1
                      
        #Set up testing stage
        print('\nTESTING')
        A,weights = create_prefmat(R)
        C = weights*cs[np.argmin(L)]
        print('OPTIMAL C: {}'.format(cs[np.argmin(L)]))
        beta,asv = ranksvm_k(train_k,A,C)
        #training data rankSVM score
        yy = np.dot(train_k[np.abs(beta[:,0])>th,:].T,beta[np.abs(beta)>th,None])
        
        #create test data
        print('assembling test kernel matrix...')
        test_data,test_y = make_dset(l,ap=test_ap)
        test_k = make_kmat(test_data,train_data)
        
        #test data rankSVM score
        yyt = np.dot(test_k[:,np.abs(beta[:,0])>th],beta[np.abs(beta)>th,None])
        R_rank = np.empty_like(R)
        for i in range(l):
            R_rank[i] = np.sum(yyt[i]>yy)/l
            
        #test data KNN score
        test_G = knn_score(test_k)
        R_knn = np.array([np.sum(q>train_G)/l for q in test_G])
            
        #determine ROC curves and AUC
        for i,a in enumerate(alphas):
            knn_class = np.sign(R_knn-a)
            fa1[i,trial] = np.sum(np.logical_and(knn_class==-1,test_y==1))/np.sum(test_y==1)
            tp1[i,trial] = np.sum(np.logical_and(knn_class==-1,test_y==-1))/np.sum(test_y==-1)    
        for i,a in enumerate(alphas):
            rank_class = np.sign(R_rank-a)
            fa2[i,trial] = np.sum(np.logical_and(rank_class==-1,test_y==1))/np.sum(test_y==1)
            tp2[i,trial] = np.sum(np.logical_and(rank_class==-1,test_y==-1))/np.sum(test_y==-1)
        print('knn AUC = {:.3f}'.format(np.trapz(tp1[:,trial],fa1[:,trial])))
        print('RankSVM AUC = {:.3f}'.format(np.trapz(tp2[:,trial],fa2[:,trial])))
            
    fa1 = np.mean(fa1,1)
    tp1 = np.mean(tp1,1)
    fa2 = np.mean(fa2,1)
    tp2 = np.mean(tp2,1)
    
    print('\nOverall:')
    print('knn AUC = {:.3f}'.format(np.trapz(tp1,fa1)))
    print('RankSVM AUC = {:.3f}'.format(np.trapz(tp2,fa2)))

    print('the whole ordeal took {} seconds'.format(time()-tic))
    #Plotting
    if p.plots:
            
        plt.figure()
        plt.plot(fa1,tp1)
        plt.plot(fa2,tp2)
        plt.show()