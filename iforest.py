import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
ipy.magic("matplotlib qt")
plt.style.use('ggplot')
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

#KNN
l = 5000    #total groups
d = 2       #dimension of data
n = 100     #number of points per group
ap = .02    #percentage of groups that are anomalous

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

     
IF = IsolationForest()
IF.fit(data.reshape((l,n*d)))
pred = IF.predict(data.reshape((l,d*n)))

plt.figure()
for i in range(l):
    if y[i]==-1:
        if pred[i] == -1:
            plt.plot(data[i,:,0],data[i,:,1],'b.')
        else:
            plt.plot(data[i,:,0],data[i,:,1],'r.')
print('precision: {}, fallout: {}'.format(np.sum(np.logical_and(pred==-1,y==-1))/np.sum(y==-1),
      np.sum(np.logical_and(pred==-1,y==1))/np.sum(y==1)))
            
ocsvm = OneClassSVM(gamma=1,nu=.05)
ocsvm.fit(np.mean(data,1))
pred = ocsvm.predict(np.mean(data,1))

plt.figure()
for i in range(l):
    if y[i]==-1:
        if pred[i] == -1:
            plt.plot(data[i,:,0],data[i,:,1],'b.')
        else:
            plt.plot(data[i,:,0],data[i,:,1],'r.')
print('precision: {}, fallout: {}'.format(np.sum(np.logical_and(pred==-1,y==-1))/np.sum(y==-1),
      np.sum(np.logical_and(pred==-1,y==1))/np.sum(y==1)))


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
        
pred = IF.predict(data.reshape((l,d*n)))
plt.figure()
for i in range(l):
    if y[i]==-1:
        if pred[i] == -1:
            plt.plot(data[i,:,0],data[i,:,1],'b.')
        else:
            plt.plot(data[i,:,0],data[i,:,1],'r.')
print('precision: {}, fallout: {}'.format(np.sum(np.logical_and(pred==-1,y==-1))/np.sum(y==-1),
      np.sum(np.logical_and(pred==-1,y==1))/np.sum(y==1)))

pred = ocsvm.predict(np.mean(data,1))
plt.figure()
for i in range(l):
    if y[i]==-1:
        if pred[i] == -1:
            plt.plot(data[i,:,0],data[i,:,1],'b.')
        else:
            plt.plot(data[i,:,0],data[i,:,1],'r.')
print('precision: {}, fallout: {}'.format(np.sum(np.logical_and(pred==-1,y==-1))/np.sum(y==-1),
      np.sum(np.logical_and(pred==-1,y==1))/np.sum(y==1)))