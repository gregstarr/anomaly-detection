import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC, OneClassSVM
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
ipy.magic("matplotlib qt")
plt.style.use('ggplot')
import itertools

#%% Set Up Data
l = 40
d = 2
n = 100

data = np.empty((l,n,d))
sigma = np.array([[.01, .008],
                  [.008, .01]])
for i in range(l):
    data[i,:,:] = np.random.multivariate_normal(.3*np.random.randn(2,),
                                                sigma,
                                                (n,))
    
#%% Parameters

# KNN parameters
k = 3
m = 3

# rbf parameters
bw = np.median(pdist(data.reshape(l*n,d)))

"""
RankAD Algorithm
"""

#%% Training Stage

#kernel matrix (a)
K = np.empty((l,l))
for i,j in itertools.product(range(l),repeat=2):
    K[i,j] = np.sum(rbf_kernel(data[i,:,:],
                               data[j,:,:],
                               gamma = .5/bw))/n**2

Gx = np.empty((l,))
Rn = np.empty((l,))
for i in range(l):
    Gx[i] = np.sum(sorted(K[i,:])[-k-1:-1])/k
for i in range(l):
    Rn[i] = np.sum(Gx[i]>Gx)/l # larger kernel value indicates simmilarity
    
# quantize labels (b)
rq = np.empty((l,),dtype=int)
split = (np.max(Rn) - np.min(Rn))/m
for i in range(m):
    rq[np.logical_and(Rn>=i*split,Rn<=(i+1)*split)] = i+1
    
# create set P, solve for minimizer g (c)
P = np.array([[i,j] for i,j in itertools.product(range(l),repeat=2) if rq[i]>rq[j]])

gram = np.empty((len(P),len(P)))
for a,b in itertools.product(range(len(P)),repeat=2):
    gram[a,b] = (K[P[a,0],P[b,0]] - 
                 K[P[a,0],P[b,1]] - 
                 K[P[a,1],P[b,0]] + 
                 K[P[a,1],P[b,1]])

ghat = OneClassSVM(kernel='precomputed',nu=.1)
ghat.fit(gram)

# create g (d)
g = np.zeros((n,d))
for i,a in enumerate(ghat.support_):
    g += ghat.dual_coef_[0,i]*.5*(data[P[a,0],:,:]+data[P[a,1],:,:])/np.sum(ghat.dual_coef_)


# kernel of g and data
rank = np.empty(l)
for i in range(l):
    rank[i] = np.sum(rbf_kernel(g,data[i,:,:],gamma = .5/bw))/n**2 - ghat.intercept_
        
# how well does R_hat estimate Rn
R_hat = np.empty(l)
for i in range(l):
    R_hat[i] = np.sum(rank[i]>rank)/l

    
#%% Test data
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

g_t = [np.sum(rbf_kernel(g,test_data[i,:,:],gamma = .5/bw))/n**2-ghat.intercept_ for i in range(len(test_data))]
R_t = [np.sum(g_t[i]>rank)/l for i in range(len(test_data))]

#%%Plotting

plt.figure()
plt.subplot(121)
for i in range(l):
    plt.scatter(data[i,:,0],data[i,:,1],c = plt.cm.Blues(Rn[i]))
plt.plot(g[:,0],g[:,1],'r.')
plt.subplot(122)
for i in range(3*l):
    plt.scatter(test_data[i,:,0],test_data[i,:,1],c = plt.cm.Blues(R_t[i]))
plt.plot(g[:,0],g[:,1],'r.')

plt.figure()
plt.plot(R_hat, label='$\hat{R}$')
plt.plot(Rn,label='$R_n$')
plt.legend()

print('One Class Ranking SVM accuracy {0:4.3f}'.format(np.sum(ghat.predict(gram)>0)/len(P)))
print('Variance of (Rn-R_hat): {0:4.3f}'.format(np.var(Rn-R_hat)))
print('average score for nominal data: {0:4.3f}'.format(np.sum(R_t[:l])/l))
print('average score for rotated data: {0:4.3f}'.format(np.sum(R_t[l:2*l])/l))
print('average score for translated data: {0:4.3f}'.format(np.sum(R_t[2*l:])/l))
