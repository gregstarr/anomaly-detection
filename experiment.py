import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

n = 1000

sigma = np.array([[.01, .008],
                  [.008, .01]])
theta = np.radians(-60)
rot = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta),  np.cos(theta)]])

bws = np.arange(.001,.02,.0005)
diff1 = np.zeros(len(bws))
diff2 = np.zeros(len(bws))
diff3 = np.zeros(len(bws))
for trial in range(10):
    print('trial: {}'.format(trial))
        
    normal1 = np.random.multivariate_normal([0,0],sigma,(n,))
    normal2 = np.random.multivariate_normal([0,0],sigma,(n,))
    anomaly1 = np.random.multivariate_normal([0,0],np.dot(sigma,rot),(n,))
    anomaly2 = .32*(np.random.rand(n,2)-.5)
    anomaly3 = np.random.multivariate_normal([0,0],[[.01,0],[0,.01]],(n,))
    
    #normal data
    k = np.empty_like(bws)
    for i,bw in enumerate(bws):
        k[i] = np.sum(rbf_kernel(normal1,normal2,gamma=.5/bw))/n**2
    
    #anomaly 1
    ka1 = np.empty_like(bws)
    for i,bw in enumerate(bws):
        ka1[i] = np.sum(rbf_kernel(normal1,anomaly1,gamma=.5/bw))/n**2
    diff1 = trial*(diff1 + k-ka1) / (trial+1)
    
    #anomaly 2
    ka2 = np.empty_like(bws)
    for i,bw in enumerate(bws):
        ka2[i] = np.sum(rbf_kernel(normal1,anomaly2,gamma=.5/bw))/n**2
    diff2 = trial*(diff2 + k-ka2) / (trial+1)
    
    #anomaly 3
    ka3 = np.empty_like(bws)
    for i,bw in enumerate(bws):
        ka3[i] = np.sum(rbf_kernel(normal1,anomaly3,gamma=.5/bw))/n**2
    diff3 = trial*(diff3 + k-ka3) / (trial+1)
    
plt.figure()
plt.subplot(221)
plt.plot(normal1[:,0],normal1[:,1],'b.')
plt.plot(normal2[:,0],normal2[:,1],'r.')
plt.subplot(222)
plt.plot(normal1[:,0],normal1[:,1],'b.')
plt.plot(anomaly1[:,0],anomaly1[:,1],'r.')
plt.subplot(223)
plt.plot(normal1[:,0],normal1[:,1],'b.')
plt.plot(anomaly2[:,0],anomaly2[:,1],'r.')
plt.subplot(224)
plt.plot(normal1[:,0],normal1[:,1],'b.')
plt.plot(anomaly3[:,0],anomaly3[:,1],'r.')

plt.figure()
plt.plot(bws,diff1,label='anomaly 1')
plt.plot(bws,diff2,label='anomaly 2')
plt.plot(bws,diff3,label='anomaly 3')
plt.legend()

print('maximum distance:\n anomaly 1: {0:4.3f}, anomaly 2: {1:4.3f}, anomaly 3: {2:4.3f}'.format(diff1.max(),diff2.max(),diff3.max()))
print('optimal bandwidth:\n anomaly 1: {0:5.4f}, anomaly 2: {1:5.4f}, anomaly 3: {2:5.4f}'.format(bws[np.argmax(diff1)],bws[np.argmax(diff2)],bws[np.argmax(diff3)]))
