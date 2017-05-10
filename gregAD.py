import numpy as np
from scipy.optimize import linprog
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import itertools
from time import time

def make_dset(l=1000,   #total groups
              d=2,      #dimension of data  
              n=100,    #number of points per group
              ap=.01,   #percentage of groups that are anomalous
              sigma = np.array([[.01, .008],[.008, .01]]) #covariance matrix
              ):
    
    data = np.empty((l,n,d))
    theta = np.radians(-60)
    rot = np.array([[np.cos(theta), -np.sin(theta)], 
                    [np.sin(theta),  np.cos(theta)]])
    y = np.ones(l)
    
    for i in range(l):
        if np.random.rand()<ap:
            y[i] = -1
            anom = np.random.rand()
            if anom > .75:
                    data[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                                np.dot(sigma,rot),
                                                                (n,))
            elif anom > .5:
                data[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,)+1,
                                                            sigma,
                                                            (n,))
            elif anom > .25:
                data[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),[[.01,0],[0,.01]],(n,))
            else:
                data[i,:,:] = .32*(np.random.rand(n,2)-.5)+.3*np.random.randn(2,)
        else:
            data[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                        sigma,
                                                        (n,))
            
    return data,y

def make_kmat(X,Y=None,bw=.005):
    if Y is None:
        Y = X
            
    K = np.empty((X.shape[0],Y.shape[0]))
    for i,j in itertools.product(range(X.shape[0]),range(Y.shape[0])):
        if not i%(l//100):
            print('{:0>2d}% done'.format(int(100*i/l)),end='\r')
        K[i,j] = np.sum(rbf_kernel(X[i,:,:],
                                   Y[j,:,:],
                                   gamma = .5/bw))/(X.shape[1]*Y.shape[1])
    
    return K

def gen_levels(scores,m):
    levels = scores.copy()
    step = (np.max(levels)-np.min(levels))/m
    for i in range(m):
        levels[np.logical_and(levels>i*step,levels<(i+1)*step)] = i + 1
        
    return levels

def gen_pairs(levels):
    pairs = []
    for i in range(len(levels)):
        idx = np.where(levels[i]>levels)[0]
        for j in range(len(idx)):
            pairs.append([i,idx[j]])
            
    return np.array(pairs)
        

def knn_score(K,knn=3):
    
    score = np.empty((K.shape[0],))
    for i in range(K.shape[0]):
        score[i] = np.sum(sorted(K[i,:])[-knn-1:-1])/knn
        
    return score

def gen_A(K,pairs):
    A = np.zeros((pairs.shape[0],pairs.shape[0]+K.shape[0]))
    for i in range(pairs.shape[0]):
        for j in range(K.shape[0]):
            A[i,j] = K[j,pairs[i,0]] - K[j,pairs[i,1]]
            A[i,K.shape[0]+i] = 1
    return A

if __name__ == "__main__":
    
    from argparse import ArgumentParser
    descr = "Anomaly Detection"

    p = ArgumentParser(description=descr)
    p.add_argument('-l','--n_groups',type=int,help='number of groups',default=1000)
    p = p.parse_args()
    
    l = p.n_groups
    
    tic = time()
    
    train_data,train_y = make_dset(l=l)
        
    print('assembling training kernel matrix...')
    
    train_k = make_kmat(train_data)
    
    train_knn = knn_score(train_k)
    
    train_pred = np.ones_like(train_y)
    for i in range(l):
        if np.sum(train_knn[i]>train_knn)/l < .1:
            train_pred[i] = -1
                
                
    test_data,test_y = make_dset(l=l)
    print('assembling test kernel matrix...')
    test_k = make_kmat(test_data,train_data)
              
    test_knn = knn_score(test_k)
              
    test_pred = np.ones_like(test_y)
    for i in range(l):
        if np.sum(test_knn[i]>test_knn)/l < .1:
            test_pred[i] = -1
        
    print('time: {:.2f} seconds'.format(time()-tic))
    
    print('TRAINING DATA ==> precision: {:4.3f}, fallout: {:4.3f}'.format(
            np.sum(np.logical_and(train_pred==-1,train_y==-1))/np.sum(train_y==-1),
            np.sum(np.logical_and(train_pred==-1,train_y==1))/np.sum(train_y==1)))
    
    print('TEST DATA ==> precision: {:4.3f}, fallout: {:4.3f}'.format(
            np.sum(np.logical_and(test_pred==-1,test_y==-1))/np.sum(test_y==-1),
            np.sum(np.logical_and(test_pred==-1,test_y==1))/np.sum(test_y==1)))
        
    
    levels = gen_levels(train_knn,3)
    pairs = gen_pairs(levels)
    A = gen_A(train_k,pairs)
    
    C = 1
    c = np.ones((levels.shape[0]+pairs.shape[0],))
    c[levels.shape[0]:] = C
    LP = linprog(c, A_ub = -1*A, b_ub = -1*np.ones(pairs.shape[0]), options={"disp": True})
    print(LP)
    #%% PLOTTING
        
    plt.figure()
    plt.subplot(121)
    plt.title('classified as anomalous')
    for i in range(l):
        if train_pred[i] == -1:
            if train_y[i] == -1:
                plt.plot(train_data[i,:,0],train_data[i,:,1],'b.')
            else:
                plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
    plt.subplot(122)
    plt.title('classified as nominal')
    for i in range(l):
        if train_pred[i] == 1:
            if train_y[i] == -1:
                plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
                
    plt.figure()
    plt.title('anomalous points')
    for i in range(l):
        if train_y[i] == -1:
            if train_pred[i] == -1:
                plt.plot(train_data[i,:,0],train_data[i,:,1],'b.')
            else:
                plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
        
    
    plt.figure()
    plt.subplot(121)
    plt.title('classified as anomalous')
    for i in range(l):
        if test_pred[i] == -1:
            if test_y[i] == -1:
                plt.plot(test_data[i,:,0],test_data[i,:,1],'b.')
            else:
                plt.plot(test_data[i,:,0],test_data[i,:,1],'r.')
    plt.subplot(122)
    plt.title('classified as nominal')
    for i in range(l):
        if test_pred[i] == 1:
            if test_y[i] == -1:
                plt.plot(test_data[i,:,0],test_data[i,:,1],'r.')
                
    plt.figure()
    plt.title('anomalous points')
    for i in range(l):
        if test_y[i] == -1:
            if test_pred[i] == -1:
                plt.plot(test_data[i,:,0],test_data[i,:,1],'b.')
            else:
                plt.plot(test_data[i,:,0],test_data[i,:,1],'r.')
                
    plt.show()
                