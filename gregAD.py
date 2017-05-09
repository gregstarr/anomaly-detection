import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
#from IPython import get_ipython
#ipy = get_ipython()
#ipy.magic("matplotlib qt")
plt.style.use('ggplot')
import itertools
from time import time

tic = time()


#KNN
l = 4000    #total groups
d = 2       #dimension of data
n = 100     #number of points per group
ap = .01    #percentage of groups that are anomalous

opt_bw = .005

data = np.empty((l,n,d))
sigma = np.array([[.01, .008],
                  [.008, .01]])
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

K = np.empty((l,l))
for i,j in itertools.product(range(l),repeat=2):
    K[i,j] = np.sum(rbf_kernel(data[i,:,:],
                               data[j,:,:],
                               gamma = .5/opt_bw))/n**2
     

knn = 3
knn_score = np.empty((l,))
for i in range(l):
    knn_score[i] = np.sum(sorted(K[i,:])[-knn-1:-1])/knn

pred = np.empty_like(y)
for i in range(l):
    pred[i] = np.sum(knn_score[i]>knn_score)/l > .1
    
print('TRAINING DATA ==> precision: {:4.3f}, fallout: {:4.3f}'.format(np.sum(np.logical_and(pred==0,y==-1))/np.sum(y==-1),
      np.sum(np.logical_and(pred==0,y==1))/np.sum(y==1)))
    
plt.figure()
plt.subplot(121)
plt.title('classified as anomalous')
for i in range(l):
    if pred[i] == 0:
        if y[i] == -1:
            plt.plot(data[i,:,0],data[i,:,1],'b.')
        else:
            plt.plot(data[i,:,0],data[i,:,1],'r.')
plt.subplot(122)
plt.title('classified as nominal')
for i in range(l):
    if pred[i]:
        if y[i] == -1:
            plt.plot(data[i,:,0],data[i,:,1],'r.')
            
plt.figure()
plt.title('anomalous points')
for i in range(l):
    if y[i] == -1:
        if pred[i] == 0:
            plt.plot(data[i,:,0],data[i,:,1],'b.')
        else:
            plt.plot(data[i,:,0],data[i,:,1],'r.')
            
            
test = np.empty_like(data)
y = np.ones(l)
for i in range(l):
    if np.random.rand()<ap:
        y[i] = -1
        anom = np.random.rand()
        if anom > .75:
                test[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                            np.dot(sigma,rot),
                                                            (n,))
        elif anom > .5:
            test[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,)+1,
                                                        sigma,
                                                        (n,))
        elif anom > .25:
            test[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),[[.01,0],[0,.01]],(n,))
        else:
            test[i,:,:] = .32*(np.random.rand(n,2)-.5)+.3*np.random.randn(2,)
    else:
        test[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                    sigma,
                                                    (n,))
k_test = np.empty((l,l))
for i,j in itertools.product(range(l),repeat=2):
    k_test[i,j] = np.sum(rbf_kernel(test[i,:,:],
                                    data[j,:,:],
                                    gamma = .5/opt_bw))/n**2
          
knn_test = np.empty((l,))
for i in range(l):
    knn_test[i] = np.sum(sorted(k_test[i,:])[-knn-1:-1])/knn
          
pred = np.empty_like(y)
for i in range(l):
    pred[i] = np.sum(knn_test[i]>knn_score)/l > .1
    
plt.figure()
plt.subplot(121)
plt.title('classified as anomalous')
for i in range(l):
    if pred[i] == 0:
        if y[i] == -1:
            plt.plot(data[i,:,0],data[i,:,1],'b.')
        else:
            plt.plot(data[i,:,0],data[i,:,1],'r.')
plt.subplot(122)
plt.title('classified as nominal')
for i in range(l):
    if pred[i]:
        if y[i] == -1:
            plt.plot(data[i,:,0],data[i,:,1],'r.')
            
plt.figure()
plt.title('anomalous points')
for i in range(l):
    if y[i] == -1:
        if pred[i] == 0:
            plt.plot(data[i,:,0],data[i,:,1],'b.')
        else:
            plt.plot(data[i,:,0],data[i,:,1],'r.')
            
print('TEST DATA ==> precision: {:4.3f}, fallout: {:4.3f}'.format(np.sum(np.logical_and(pred==0,y==-1))/np.sum(y==-1),np.sum(np.logical_and(pred==0,y==1))/np.sum(y==1)))
            
print('time: '+str(time()-tic))