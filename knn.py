import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
ipy.magic("matplotlib qt")
plt.style.use('ggplot')
import itertools

#KNN
l = 500     #total training groups
d = 2       #dimension of data
n = 100     #number of points per group

opt_bw = .005

data = np.empty((l,n,d))
sigma = np.array([[.01, .008],
                  [.008, .01]])
for i in range(l):
    data[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                sigma,
                                                (n,))

#K = np.empty((l,l))
#for i,j in itertools.product(range(l),repeat=2):
#    K[i,j] = np.sum(rbf_kernel(data[i,:,:],
#                               data[j,:,:],
#                               gamma = .5/opt_bw))/n**2
#     
#knn = 10
#knn_score = np.empty((l,))
#for i in range(l):
#    knn_score[i] = np.sum(sorted(K[i,:])[-knn-1:-1])/knn
    
theta = np.radians(-60)
rot = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta),  np.cos(theta)]])
test_data = np.empty((3*l,n,d))
for i in range(l):
    test_data[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                     sigma,
                                                     (n,))
    test_data[i+l,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                       np.dot(sigma,rot),
                                                       (n,))
    test_data[i+2*l,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,)+1,
                                                         sigma,
                                                         (n,))

test_K = np.empty((3*l,l))
for i,j in itertools.product(range(3*l),range(l)):
    test_K[i,j] = np.sum(rbf_kernel(test_data[i,:,:],
                                    data[j,:,:],
                                    gamma = .5/opt_bw))/n**2
          
plt.figure()
test_score = np.max(test_K,1)

th = np.mean([np.mean(test_score[:l]),np.mean(test_score[l:2*l])])
for i in range(3*l):
    if(test_score[i]>th):
        plt.plot(test_data[i,:,0],test_data[i,:,1],'b.')
    else:
        plt.plot(test_data[i,:,0],test_data[i,:,1],'r.')

print('threshold: '+str(th))
print('error: '+str((np.sum(test_score[l:]>th)+np.sum(test_score[:l]<th))/(3*l)))

#for nn in range(1,6):
#    test_score = np.zeros(3*l)
#    for t in range(3*l):
#        test_score[t] += np.sum(sorted(test_K[t,:])[-nn-1:-1])/nn
#    plt.plot(test_score,label=str(nn))
#    print('nn {}, normal mean {}, anomaly mean {}'.format(nn,np.mean(test_score[:l]),np.mean(test_score[l:2*l])))
#    print(np.mean(test_score[:l])-np.mean(test_score[l:2*l]))
#plt.legend()