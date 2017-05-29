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

def make_dset(l=1000,   #total groups
              d=2,      #dimension of data  
              n=100,    #number of points per group
              ap=.02,   #percentage of groups that are anomalous
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


def knn_score(K,knn=3):
    
    score = np.empty((K.shape[0],))
    for i in range(K.shape[0]):
        score[i] = np.sum(sorted(K[i,:])[-knn-1:-1])/knn
        
    return score



if __name__ == "__main__":
    
    from argparse import ArgumentParser
    descr = "Anomaly Detection"

    p = ArgumentParser(description=descr)
    p.add_argument('-l','--n_groups',type=int,help='number of groups',default=1000)
    p.add_argument('-p','--plots',action='store_true',help='plots',default=False)
    p.add_argument('-t','--trials',type=int,help='trials',default=3)
    p.add_argument('-a','--alpha',type=float,help='confidence level',default=.1)
    p = p.parse_args()
    

    l = p.n_groups
    alpha = p.alpha #confidence level
    
    #assemble test data
    TRIALS = p.trials
    
    fa = np.empty((100,TRIALS))
    tp = np.empty((100,TRIALS))
    alphas = np.arange(0,1,.01)
    
    for trial in range(TRIALS):
            
        tic = time() # Time the training
        #Create training data
        
        train_data,train_y = make_dset(l=l)
        #assemble kernel matrix
        print('assembling training kernel matrix...')
        train_k = make_kmat(train_data)
        #determine knn scores for training data, predict training anomalies
        train_g = knn_score(train_k) #simmilarity metric
        R = np.empty_like(train_y) #ranking metric [0,1)
        for i in range(l):
            R[i] = np.sum(train_g[i]>train_g)/l
        train_class = np.sign(R-alpha)
        
        print('TRAINING DATA: knn ==> precision: {:4.3f}, fallout: {:4.3f}'.format(
                np.sum(np.logical_and(train_class==-1,train_y==-1))/np.sum(train_y==-1),
                np.sum(np.logical_and(train_class==-1,train_y==1))/np.sum(train_y==1)))
        
        print('\ntraining took {:.2f} seconds'.format(time()-tic))
            
        test_data,test_y = make_dset(l=l)
        #assemble kernel matrix
        tic = time()
        print('assembling test kernel matrix...')
        test_k = make_kmat(test_data,train_data)
        test_g = knn_score(test_k)
        test_R = np.array([np.sum(q>train_g)/l for q in test_g])
    
        for i,a in enumerate(alphas):
            test_class = np.sign(test_R-a)
            fa[i,trial] = np.sum(np.logical_and(test_class==-1,test_y==1))/np.sum(test_y==1)
            tp[i,trial] = np.sum(np.logical_and(test_class==-1,test_y==-1))/np.sum(test_y==-1)
        
        
        print('\ntesting took {:.2f} seconds'.format(time()-tic))
        test_class = np.sign(test_R-alpha)
        print('TEST DATA: knn ==> precision: {:4.3f}, fallout: {:4.3f}'.format(
                np.sum(np.logical_and(test_class==-1,test_y==-1))/np.sum(test_y==-1),
                np.sum(np.logical_and(test_class==-1,test_y==1))/np.sum(test_y==1)))
        
    fa = np.mean(fa,1)
    tp = np.mean(tp,1)

    
    #%% PLOTTING
    if p.plots:
        
        plt.figure()
        plt.title('ROC')
        plt.plot(fa,tp)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
            
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
        plt.title('Training anomalous points')
        for i in range(l):
            if train_y[i] == -1:
                if train_class[i] == -1:
                    plt.plot(train_data[i,:,0],train_data[i,:,1],'b.')
                else:
                    plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
            
                
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
                    