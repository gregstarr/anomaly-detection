from gregAD import make_dset, make_kmat
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.svm import OneClassSVM


fa1 = np.zeros((30,5))
tp1 = np.zeros((30,5))
fa2 = np.zeros((30,5))
tp2 = np.zeros((30,5))
for trial in range(1):
        
    train_data, train_y = make_dset(l=500, ap=.06)
    K = make_kmat(train_data,bw=.05)
    for i,nu in enumerate(np.logspace(-5, 0, 30)):
        ocsmm = OneClassSVM(kernel='precomputed', nu=nu)
        ocsmm.fit(K)
        
        pred = ocsmm.predict(K)
        
        print('nu: {:.2e}, FA: {:.2f}, TP: {:.2f}'.format(
                nu,
                np.sum(np.logical_and(pred==-1,train_y==-1))/np.sum(train_y==-1),
                np.sum(np.logical_and(pred==-1,train_y==1))/np.sum(train_y==1)))
        fa1[i,trial] = np.sum(np.logical_and(pred==-1,train_y==-1))/np.sum(train_y==-1)
        tp1[i,trial] = np.sum(np.logical_and(pred==-1,train_y==1))/np.sum(train_y==1)
        
        ocsvm = OneClassSVM(nu=nu)
        ocsvm.fit(np.mean(train_data,1))
        pred = ocsvm.predict(np.mean(train_data,1))
        fa2[i,trial] = np.sum(np.logical_and(pred==-1,train_y==-1))/np.sum(train_y==-1)
        tp2[i,trial] = np.sum(np.logical_and(pred==-1,train_y==1))/np.sum(train_y==1)
        
fa1 = np.mean(fa1,1)
tp1 = np.mean(tp1,1)
fa2 = np.mean(fa2,1)
tp2 = np.mean(tp2,1)
plt.figure()
plt.plot(sorted(fa1),sorted(tp1),label='OCSMM')
plt.plot(sorted(fa2),sorted(tp2),label='OCSVM')
plt.xlabel('FA')
plt.ylabel('TP')
plt.legend()


nu = .1
ocsmm = OneClassSVM(kernel='precomputed', nu=nu)
ocsmm.fit(K)
pred1 = ocsmm.predict(K)
ocsvm = OneClassSVM(nu=nu)
ocsvm.fit(np.mean(train_data,1))
pred2 = ocsvm.predict(np.mean(train_data,1))

plt.figure()
plt.title('anomalous points')
plt.subplot(121)
for i in range(500):
    if train_y[i] == -1:
        if pred1[i] == -1:
            plt.plot(train_data[i,:,0],train_data[i,:,1],'b.')
        else:
            plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
plt.subplot(122)
for i in range(500):
    if train_y[i] == -1:
        if pred2[i] == -1:
            plt.plot(train_data[i,:,0],train_data[i,:,1],'b.')
        else:
            plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
            
plt.figure()
plt.suptitle('Training Data Anomaly Detection Performance')
plt.subplot(121)
plt.title('classified as anomalous')
for i in range(500):
    if pred1[i] == -1:
        if train_y[i] == -1:
            plt.plot(train_data[i,:,0],train_data[i,:,1],'b.')
        else:
            plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
plt.subplot(122)
plt.title('classified as nominal')
for i in range(500):
    if pred1[i] == 1:
        if train_y[i] == -1:
            plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
            
plt.figure()
plt.suptitle('Training Data Anomaly Detection Performance')
plt.subplot(121)
plt.title('classified as anomalous')
for i in range(500):
    if pred2[i] == -1:
        if train_y[i] == -1:
            plt.plot(train_data[i,:,0],train_data[i,:,1],'b.')
        else:
            plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
plt.subplot(122)
plt.title('classified as nominal')
for i in range(500):
    if pred2[i] == 1:
        if train_y[i] == -1:
            plt.plot(train_data[i,:,0],train_data[i,:,1],'r.')
       