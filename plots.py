import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from gregAD import make_dset

data,y = make_dset(20,n=200,ap=.1)
for i in range(20):
    if y[i] == -1:
        plt.plot(data[i,:,0],data[i,:,1],'r.',zorder=100)
    else:
        plt.plot(data[i,:,0],data[i,:,1],'b.')
