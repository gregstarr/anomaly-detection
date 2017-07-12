"""
new idea, new objective function no chance it will work

min sum over i: G_x(x_i) - C alpha^T alpha
didn't really work, doesn't perform as well as original, does reduce support
maximizing knn score of everything doesn't help

new idea, minimize difference in knn scores between resulting and original 
while penalizing L2 norm of alpha
didn't work that well

new idea, only correct the ones that are wrong according to the knn classification,
use hinge loss?
"""



from gregAD import make_dset, make_kmat, knn_score
import numpy as np
import matplotlib.pyplot as plt  

def GD_method(K,C,train_data,knn=3,RATE=1):
    l = K.shape[0]
    #initialize alphas, obj
    alphas = np.ones(l)/l
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(121)
    ax1.set_ylim([0,20])
    ax2 = fig.add_subplot(122)
    
    dphandles = []
    for i in range(l):
        dphandles.append(ax2.scatter(train_data[i,:,0],train_data[i,:,1],c=plt.cm.Blues(alphas[i]),alpha=.2))
    line, = ax1.plot(alphas)
    plt.ion()
    plt.show()
    plt.pause(.0001)
    
    G_0 = knn_score(K)
    G_alpha = np.array([np.sum(sorted((alphas*K[i,:])[np.arange(l)!=i])[-knn:])/3 for i in range(l)])
    
    obj_0 = 0
    obj = (np.sum((G_alpha-G_0)**2) + np.sum(C*alphas**2))/2
    t = 0
    while abs(obj-obj_0) > .001:
        t += 1
        print('It = {:3d}, Ob = {:12.4f}, N sv = {:4d}, diff = {:12.4f}, '
              'mag: {:12.4f}'.format(t,obj,np.sum(alphas>.01),np.sum((G_alpha-G_0)**2),np.sum(C*alphas**2),end='\r'))
        obj_0 = obj
        first_term = np.zeros(l)
        for i in range(l):
            for j in np.argsort((alphas*K[i,:])[np.arange(l)!=i])[-knn:]:
                first_term[j] += (G_alpha[i]-G_0[i])*K[i,j]/3
        
        alphas -= RATE*(first_term + C*alphas)
        
        G_alpha = np.array([np.sum(sorted((alphas*K[i,:])[np.arange(l)!=i])[-knn:])/3 for i in range(l)])
        obj = (np.sum((G_alpha-G_0)**2) + np.sum(C*alphas**2))/2
    
        if t % 20 == 0:
                
            line.set_ydata(alphas)
            for i in range(l):
                dphandles[i].set_color(plt.cm.Blues(alphas[i]))
            fig.canvas.draw()
            
    print('trials '+str(t))
    plt.ioff()
    plt.show()
            
    return alphas

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
        
    
    #Setup Parameters
    l = 100
    train_ap = .05
    knn = 3
    KFOLD = 4
    
    #Create training data
    train_data,train_y = make_dset(l,ap=train_ap)
    
    #assemble kernel matrix
    print('assembling training kernel matrix...')
    train_k = make_kmat(train_data)
    
    Cs = np.logspace(-3,2,num=20)
    prec = np.empty_like(Cs)
    idx = np.arange(l)
    
    for fold in range(KFOLD):
        print('\nFOLD: {}'.format(fold+1))
        #training and testing portions
        idx_tr = idx[~np.logical_and(idx>=fold*l/KFOLD,idx<(fold+1)*l/KFOLD)]
        idx_te = idx[np.logical_and(idx>=fold*l/KFOLD,idx<(fold+1)*l/KFOLD)]
        k_tr = train_k[idx_tr][:,idx_tr]
        k_te = train_k[idx_tr][:,idx_te]
    
        for t,C in enumerate(Cs):
            print('------------------- C = {} ----------------------'.format(C))
                
            alphas = GD_method(k_tr,C)
                
            score = np.array([np.sum(sorted(alphas*k_te[:,i])[-knn:])/3 for i in range(len(idx_te))])
            
            R = np.array([np.sum(score[i]>score)/len(idx_te) for i in range(len(idx_te))])
            
            cl = np.sign(R-.05)
            prec[t] = np.sum(train_y[idx_te][cl==-1]==-1)/np.sum(train_y[idx_te]==-1)
    
    C = Cs[np.argmax(prec)]
    print('C = '+str(C))
    
    
    #Create test data
    test_data,test_y = make_dset(l,ap=train_ap)
    #assemble kernel matrix
    print('assembling test kernel matrix...')
    test_k = make_kmat(test_data,train_data)
    
    alphas = GD_method(train_k,C)  
    score = np.array([np.sum(sorted(alphas*test_k[i,:])[-knn:])/3 for i in range(l)])
    R = np.array([np.sum(score[i]>score)/l for i in range(l)])
    
    fa = np.empty(100)
    tp = np.empty(100)
    for i,a in enumerate(np.arange(0,1,.01)):
        rank_class = np.sign(R-a)
        fa[i] = np.sum(np.logical_and(rank_class==-1,train_y==1))/np.sum(train_y==1)
        tp[i] = np.sum(np.logical_and(rank_class==-1,train_y==-1))/np.sum(train_y==-1)
    print('AUC = {:.3f}'.format(np.trapz(tp,fa)))
    print(np.sum(alphas<.001))
    plt.plot(fa,tp)
    plt.show()