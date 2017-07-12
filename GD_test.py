import matplotlib
matplotlib.use('TkAgg')
from gregAD import make_dset, make_kmat, knn_score
import numpy as np
import matplotlib.pyplot as plt
from GDidea import GD_method
plt.style.use('ggplot')

#Setup Parameters
l = 500
train_ap = .05
knn = 3
C = 0

#Create training data
train_data,train_y = make_dset(l,ap=train_ap)

#assemble kernel matrix
print('assembling training kernel matrix...')
train_k = make_kmat(train_data)

train_G = knn_score(train_k)
R_knn = np.array([np.sum(q>train_G)/l for q in train_G])

alphas = GD_method(train_k,C,train_data,RATE=.1)
score = np.array([np.sum(sorted(alphas*train_k[i,:])[-knn:])/3 for i in range(l)])
R = np.array([np.sum(score[i]>score)/l for i in range(l)])

fak = np.empty(100)
tpk = np.empty(100)
fag = np.empty(100)
tpg = np.empty(100)
for i,a in enumerate(np.arange(0,1,.01)):
    gd_class = np.sign(R-a)
    fag[i] = np.sum(np.logical_and(gd_class==-1,train_y==1))/np.sum(train_y==1)
    tpg[i] = np.sum(np.logical_and(gd_class==-1,train_y==-1))/np.sum(train_y==-1)
    knn_class = np.sign(R_knn-a)
    fak[i] = np.sum(np.logical_and(knn_class==-1,train_y==1))/np.sum(train_y==1)
    tpk[i] = np.sum(np.logical_and(knn_class==-1,train_y==-1))/np.sum(train_y==-1)
print('AUC gd = {:.3f}'.format(np.trapz(tpg,fag)))
print('AUC knn = {:.3f}'.format(np.trapz(tpk,fak)))

print(np.sum(alphas<.001))
plt.figure()
plt.plot(fag,tpg)
plt.plot(fak,tpk)

plt.show()