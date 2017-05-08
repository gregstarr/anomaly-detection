import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import NuSVC, OneClassSVM
import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
ipy.magic("matplotlib qt")
plt.style.use('ggplot')
import itertools

"""
Binary SVM Anomaly classifier - cheat to find hyperparameters
"""

#%% Set Up Data
l = 200      #total training groups, 1/2 normal, 1/4 translated, 1/4 rotated
d = 2       #dimension of data
n = 100     #number of points per group

data = np.empty((l,n,d))
y = np.ones(l)
y[l//2:] *= -1
sigma = np.array([[.01, .008],
                  [.008, .01]])
theta = np.radians(-60)
rot = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta),  np.cos(theta)]])

for i in range(l//2):
    data[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                sigma,
                                                (n,))
for i in range(l//4):
    data[i+l//2,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                     np.dot(sigma,rot),
                                                     (n,))
    data[i+l//2+l//4,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,)+2,
                                                          sigma,
                                                          (n,))

#Cross validate
indices = np.arange(l)
bandwidths = np.arange(.01,1,.05)
nus = np.arange(.01,.5,.025)
accuracy = np.empty((bandwidths.shape[0],nus.shape[0]))
for a,bw in enumerate(bandwidths):
    print('bandwidth: {0:3.2f}'.format(bw))
    K = np.empty((l,l))
    for i,j in itertools.product(range(l),repeat=2):
        K[i,j] = np.sum(rbf_kernel(data[i,:,:],
                                   data[j,:,:],
                                   gamma = .5/bw))/n**2
    for b,nu in enumerate(nus):
        train_id = np.random.choice(indices,size=int(.7*l),replace=False)
        test_id = indices[~np.in1d(indices,train_id)]
        k_train = K[:,train_id][train_id]
        k_test = K[:,train_id][test_id]
        svm = NuSVC(nu=nu,kernel='precomputed')
        svm.fit(k_train,y[train_id])
        error = np.sum(svm.predict(k_test)==y[test_id])/len(test_id)
        accuracy[a,b] = 1-error
        
plt.pcolormesh(accuracy)
opt_nu = nus[np.argmax(accuracy)%bandwidths.shape[0]]
opt_bw = bandwidths[np.argmax(accuracy)//bandwidths.shape[0]]
print('best accuracy: {0:4.3f}'.format(accuracy.max()))
print('optimal nu: {0:4.3f}'.format(opt_nu))
print('optimal bandwidth: {0:4.3f}'.format(opt_bw))

#TEST
for i in range(l//2):
    data[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                sigma,
                                                (n,))
for i in range(l//4):
    data[i+l//2,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                     np.dot(sigma,rot),
                                                     (n,))
    data[i+l//2+l//4,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,)+2,
                                                          sigma,
                                                          (n,))
    
K = np.empty((l,l))
for i,j in itertools.product(range(l),repeat=2):
    K[i,j] = np.sum(rbf_kernel(data[i,:,:],
                               data[j,:,:],
                               gamma = .5/opt_bw))/n**2
     
svm = NuSVC(nu=opt_nu,kernel='precomputed')
svm.fit(K,y)

test = np.empty((l,n,d))
for i in range(l//2):
    test[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                sigma,
                                                (n,))
for i in range(l//4):
    test[i+l//2,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                     np.dot(sigma,rot),
                                                     (n,))
    test[i+l//2+l//4,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,)+2,
                                                          sigma,
                                                          (n,))
    
K_test = np.empty((l,l))
for i,j in itertools.product(range(l),repeat=2):
    K_test[i,j] = np.sum(rbf_kernel(data[i,:,:],
                                    test[j,:,:],
                                    gamma = .5/opt_bw))/n**2
     
prediction = svm.predict(K_test)
plt.figure()
plt.subplot(121)
plt.plot(test[prediction>0,:,0],test[prediction>0,:,1],'b.')
plt.plot(test[prediction<0,:,0],test[prediction<0,:,1],'r.')
print('test accuracy: {0:4.3f}'.format(np.sum(prediction==y)/l))
print(np.sum(prediction[l//2:l//2+l//4]==-1)/(l/4))


#OCSMM
ocsmm = OneClassSVM(kernel='precomputed',nu=opt_nu)
ocsmm.fit(K)
prediction = ocsmm.predict(K_test)
plt.subplot(122)
plt.plot(test[prediction>0,:,0],test[prediction>0,:,1],'b.')
plt.plot(test[prediction<0,:,0],test[prediction<0,:,1],'r.')
print('test accuracy: {0:4.3f}'.format(np.sum(prediction==y)/l))
print(np.sum(prediction[l//2:l//2+l//4]==-1)/(l/4))

#KNN
knn = 3
knn_score = np.empty((l,))
for i in range(l):
    knn_score[i] = np.sum(sorted(K_test[:l//2,i])[-knn-1:-1])/knn
    print(sorted(K_test[:l//2,i])[-knn-1:-1],end=', ')
    print(y[i])
    
plt.figure()
plt.plot(knn_score)