import os
if 'PYTHONSTARTUP' in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import itertools
from time import time
from cvxopt import spmatrix, solvers, matrix

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
        try:
            if not i%(X.shape[0]//100):
                print('{:0>2d}% done'.format(int(100*i/X.shape[0])),end='\r')
        except:
            pass
        K[i,j] = np.sum(rbf_kernel(X[i,:,:],
                                   Y[j,:,:],
                                   gamma = .5/bw))/(X.shape[1]*Y.shape[1])
    
    return K

def gen_pairs(scores,m):
    pairs = []
    
    levels = scores.copy()
    bot = np.min(levels)
    top = np.max(levels)
    step = (top-bot)/m
    for i in range(m):
        levels[np.logical_and(levels>=i*step+bot,
                              levels<=(i+1)*step+bot)] = i + 1
        
    for i in range(len(levels)):
        idx = np.where(levels[i]>levels)[0]
        for j in range(len(idx)):
            pairs.append([i,idx[j]])

    return np.array(pairs)

def gen_ranking_matrix(K,scores,pairs=None):
    if pairs is None:
        pairs = np.array([[a,b] for [a,b] in itertools.combinations(range(len(scores)),2)])
    mat = np.empty((pairs.shape[0],pairs.shape[0]))
    y = np.ones(pairs.shape[0])
    for i,pair in enumerate(pairs):
        if scores[pair[0]] < scores[pair[1]]:
            y[i] = -1
        mat[i,:] = (K[pair[0],pairs[:,0]] - 
                    K[pair[0],pairs[:,1]] - 
                    K[pair[1],pairs[:,0]] + 
                    K[pair[1],pairs[:,1]])
        
    return mat, y

def knn_score(K,knn=3):
    
    score = np.empty((K.shape[0],))
    for i in range(K.shape[0]):
        score[i] = np.sum(sorted(K[i,:])[-knn-1:-1])/knn
        
    return score

def gen_A(K,pairs):
    I = []
    J = []
    x = []
    for i in range(pairs.shape[0]):
        for j in range(K.shape[0]):
            I.append(i)
            J.append(j)
            x.append(K[j,pairs[i,0]] - K[j,pairs[i,1]])
        I.append(i)
        J.append(K.shape[0]+i)
        x.append(1)
    for i in range(pairs.shape[0] + K.shape[0]):
        I.append(pairs.shape[0]+i)
        J.append(i)
        x.append(-1)
    return spmatrix(x,I,J)

if __name__ == "__main__":
    
    from argparse import ArgumentParser
    descr = "Anomaly Detection"

    p = ArgumentParser(description=descr)
    p.add_argument('-l','--n_groups',type=int,help='number of groups',default=1000)
    p.add_argument('-p','--plots',action='store_true',help='plots',default=False)
    p.add_argument('-m','--n_levels',type=int,help='number levels to quantize rank',default=None)
    p.add_argument('-t','--test',action='store_true',help='classify test data',default=False)
    p.add_argument('-a','--alpha',type=float,help='confidence level',default=.1)
    p = p.parse_args()
    
    tic = time() # Time the training
    #Create training data
    l = p.n_groups
    train_data,train_y = make_dset(l=l)
    #assemble kernel matrix
    print('assembling training kernel matrix...')
    train_k = make_kmat(train_data)
    #determine knn scores for training data, predict training anomalies
    alpha = p.alpha #confidence level
    train_G = knn_score(train_k) #simmilarity metric
    train_R = np.empty_like(train_y) #ranking metric [0,1)
    for i in range(l):
        train_R[i] = np.sum(train_G[i]>train_G)/l
    train_pred = 2*(train_R > alpha) - 1
    print('\ntraining took {:.2f} seconds'.format(time()-tic))
    
    print('TRAINING DATA: knn ==> precision: {:4.3f}, fallout: {:4.3f}'.format(
            np.sum(np.logical_and(train_pred==-1,train_y==-1))/np.sum(train_y==-1),
            np.sum(np.logical_and(train_pred==-1,train_y==1))/np.sum(train_y==1)))
    
    """
    #IDEA 1
    #Train SVM outlier/inlier
    C = .11
    svm = SVC(C=C, kernel = 'precomputed',verbose=True,class_weight='balanced')
    svm.fit(train_k,train_y)
    print('\ntraining took {:.2f} seconds'.format(time()-tic))
    #score training data using SVM
    train_g = np.sum(svm.dual_coef_*train_k[:,svm.support_],1)
    
    #standard svm classification
    train_class = np.sign(train_g + svm.intercept_)
    
    #compare g score with rest of data
    train_class = np.ones_like(train_g)
    for i in range(l):
        if np.sum(train_g[i]>train_g)/l < .1:
            train_class[i] = -1
    """
    #IDEA 2
    #full ranking svm
    down_idx = np.sort(np.argsort(train_R)[np.arange(0,l,l//150)]) #downsampling
    pairs = np.array([[a,b] for [a,b] in itertools.combinations(down_idx,2)])
    matr,rv = gen_ranking_matrix(train_k[down_idx,:][:,down_idx],train_G[down_idx])
    C = .01
    print()
    svm = SVC(C=C, kernel = 'precomputed',verbose=True)
    svm.fit(matr,rv)
    print()
    train_g = np.empty(l)
    train_class = np.ones_like(train_g)
    for i in range(l):
        train_g[i] = np.sum(svm.dual_coef_*(train_k[pairs[svm.support_,0],i] -
                                            train_k[pairs[svm.support_,1],i])) + svm.intercept_
    for i in range(l):
        if np.sum(train_g[i] > train_g)/l < alpha:
            train_class[i] = -1


    print('TRAINING DATA: svm ==> precision: {:4.3f}, fallout: {:4.3f}'.format(
            np.sum(np.logical_and(train_class==-1,train_y==-1))/np.sum(train_y==-1),
            np.sum(np.logical_and(train_class==-1,train_y==1))/np.sum(train_y==1)))

    if p.test:
        #assemble test data
        test_data,test_y = make_dset(l=l)
        #assemble kernel matrix
        tic = time()
        print('assembling test kernel matrix...')
        """
        IDEA 1
        test_k = make_kmat(test_data,train_data[svm.support_,:])
        #test on SVM
        test_g = np.sum(svm.dual_coef_*test_k,1)
        test_class = np.sign(test_g + svm.intercept_)
        test_class = np.ones_like(test_g)
        for i in range(test_g.shape[0]):
            if np.sum(test_g[i]>train_g)/l < .1:
                test_class[i] = -1
        """
        """
        #IDEA 0
        #test using KNN
        test_knn = knn_score(test_k)
        test_pred = np.ones_like(test_y)
        for i in range(l):
            if np.sum(test_knn[i]>train_knn)/l < alpha:
                test_pred[i] = -1
        """
        #IDEA 2
        svs = np.unique(pairs[svm.support_])
        test_k = make_kmat(test_data,train_data[svs,:,:])
        test_g = np.empty(l)
        test_R = np.empty(l)
        test_class = np.ones(l)
        new_pairs = np.empty_like(pairs[svm.support_])
        for i,pair in enumerate(pairs[svm.support_]):
            a, = np.where(svs==pair[0])
            b, = np.where(svs==pair[1])
            new_pairs[i,:] = [a,b]
            
        for i in range(l):
            test_g[i] = np.sum(svm.dual_coef_*(test_k[i,new_pairs[:,0]] -
                                               test_k[i,new_pairs[:,1]])) + svm.intercept_
            test_R[i] = np.sum(test_g[i] > train_g)/l
            
        test_class = 2*(test_R > alpha) - 1
        
        
        print('\ntesting took {:.2f} seconds'.format(time()-tic))
#        print('TEST DATA: knn ==> precision: {:4.3f}, fallout: {:4.3f}'.format(
#                np.sum(np.logical_and(test_pred==-1,test_y==-1))/np.sum(test_y==-1),
#                np.sum(np.logical_and(test_pred==-1,test_y==1))/np.sum(test_y==1)))
        print('TEST DATA: svm ==> precision: {:4.3f}, fallout: {:4.3f}'.format(
                np.sum(np.logical_and(test_class==-1,test_y==-1))/np.sum(test_y==-1),
                np.sum(np.logical_and(test_class==-1,test_y==1))/np.sum(test_y==1)))        
    """
    #IDEA 4: CVXOPT LP solving

    print('setting up LP...')
    if p.n_levels is not None:
        levels = gen_levels(train_knn,p.n_levels)
        pairs = gen_pairs(levels)
    else:
        pairs = gen_pairs(scores=train_knn)    
        
    G = gen_A(train_k,pairs)
    
    C = .1
    c = matrix(np.ones((test_knn.shape[0]+pairs.shape[0],)))
    c[test_knn.shape[0]:] = C
    h = matrix(-1*np.ones(2*pairs.shape[0]+train_k.shape[0]))
    h[pairs.shape[0]:] = 0
    
    print('solving LP...')
    sol = solvers.lp(c,G,h,solver='glpk')
    print(sol['x'])
    """
    
    
    #%% PLOTTING
    if p.plots:
            
        plt.figure()
        plt.suptitle('Training Data Anomaly Detection Performance')
        plt.subplot(121)
        plt.title('classified as anomalous')
        for i in range(l):
            if train_class[i] == -1:
                if train_y[i] == -1:
                    plt.plot(train_data[i,:,0],train_data[i,:,1],'b.')
                else:
                    plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
        plt.subplot(122)
        plt.title('classified as nominal')
        for i in range(l):
            if train_class[i] == 1:
                if train_y[i] == -1:
                    plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
       
        plt.figure()
        plt.suptitle('Training data')
        for i in range(l):
            if train_class[i] == -1:
                plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
            else:
                plt.plot(train_data[i,:,0],train_data[i,:,1],'b.')
        
        plt.figure()
        plt.subplot(121)
        plt.title('g score')
        for i in range(l):
            plt.plot(train_data[i,:,0],train_data[i,:,1],'.',c=plt.cm.Blues(train_g[i]))
        plt.subplot(122)
        plt.title('R score')
        for i in range(l):
            plt.plot(train_data[i,:,0],train_data[i,:,1],'.',c=plt.cm.Blues(train_R[i]))
                    
        plt.figure()
        plt.title('Training anomalous points')
        for i in range(l):
            if train_y[i] == -1:
                if train_class[i] == -1:
                    plt.plot(train_data[i,:,0],train_data[i,:,1],'b.')
                else:
                    plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
            
        if p.test:
                
            plt.figure()
            plt.suptitle('Test Data Anomaly Detection Performance')
            plt.subplot(121)
            plt.title('classified as anomalous')
            for i in range(l):
                if test_class[i] == -1:
                    if test_y[i] == -1:
                        plt.plot(test_data[i,:,0],test_data[i,:,1],'b.')
                    else:
                        plt.plot(test_data[i,:,0],test_data[i,:,1],'r.')
            plt.subplot(122)
            plt.title('classified as nominal')
            for i in range(l):
                if test_class[i] == 1:
                    if test_y[i] == -1:
                        plt.plot(test_data[i,:,0],test_data[i,:,1],'r.')
                        
            plt.figure()
            plt.title('Test anomalous points')
            for i in range(l):
                if test_y[i] == -1:
                    if test_class[i] == -1:
                        plt.plot(test_data[i,:,0],test_data[i,:,1],'b.')
                    else:
                        plt.plot(test_data[i,:,0],test_data[i,:,1],'r.')
                    
        plt.show()
                    