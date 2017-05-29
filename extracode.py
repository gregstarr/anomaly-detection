    """
    #IDEA 1
    #Train SVM outlier/inlier
    C = .11
    svm = SVC(C=C, kernel = 'precomputed',verbose=True,class_weight='balanced')
    svm.fit(train_k,train_y)
    print('\ntraining took {:.2f} seconds'.format(time()-tic))
    #score training data using SVM
    train_g = np.sum(svm.dual_coef_*train_k[:,svm.support_],1)
    
    #standard svm classification
    train_class = np.sign(train_g + svm.intercept_)
    
    #compare g score with rest of data
    train_class = np.ones_like(train_g)
    for i in range(l):
        if np.sum(train_g[i]>train_g)/l < .1:
            train_class[i] = -1
    """
    
def gen_A(K,pairs):
    I = []
    J = []
    x = []
    for i in range(pairs.shape[0]):
        for j in range(K.shape[0]):
            I.append(i)
            J.append(j)
            x.append(K[j,pairs[i,0]] - K[j,pairs[i,1]])
        I.append(i)
        J.append(K.shape[0]+i)
        x.append(1)
    for i in range(pairs.shape[0] + K.shape[0]):
        I.append(pairs.shape[0]+i)
        J.append(i)
        x.append(-1)
    return spmatrix(x,I,J)

        """
        IDEA 1
        test_k = make_kmat(test_data,train_data[svm.support_,:])
        #test on SVM
        test_g = np.sum(svm.dual_coef_*test_k,1)
        test_class = np.sign(test_g + svm.intercept_)
        test_class = np.ones_like(test_g)
        for i in range(test_g.shape[0]):
            if np.sum(test_g[i]>train_g)/l < .1:
                test_class[i] = -1
        """
        """
        #IDEA 0
        #test using KNN
        test_knn = knn_score(test_k)
        test_pred = np.ones_like(test_y)
        for i in range(l):
            if np.sum(test_knn[i]>train_knn)/l < alpha:
                test_pred[i] = -1
        """
        
            """
    #IDEA 4: CVXOPT LP solving

    print('setting up LP...')
    if p.n_levels is not None:
        levels = gen_levels(train_knn,p.n_levels)
        pairs = gen_pairs(levels)
    else:
        pairs = gen_pairs(scores=train_knn)    
        
    G = gen_A(train_k,pairs)
    
    C = .1
    c = matrix(np.ones((test_knn.shape[0]+pairs.shape[0],)))
    c[test_knn.shape[0]:] = C
    h = matrix(-1*np.ones(2*pairs.shape[0]+train_k.shape[0]))
    h[pairs.shape[0]:] = 0
    
    print('solving LP...')
    sol = solvers.lp(c,G,h,solver='glpk')
    print(sol['x'])
    """
    
def gen_ranking_matrix(K,scores,m):

    pairs = []
    y=[]
    levels = scores.copy()
    bot = np.min(levels)
    top = np.max(levels)
    step = (top-bot)/m
    for i in range(m):
        levels[np.logical_and(levels>=i*step+bot,
                              levels<=(i+1)*step+bot)] = i + 1
    
    for (i,j) in itertools.combinations(range(scores.shape[0]),r=2):
        if levels[i]>levels[j]:
            y.append(1)
            pairs.append([i,j])
        elif levels[i]<levels[j]:
            y.append(-1)
            pairs.append([i,j])
            
    pairs = np.array(pairs)
    y = np.array(y)
        
    mat = np.empty((pairs.shape[0],pairs.shape[0]))
    for i,pair in enumerate(pairs):
        mat[i,:] = (K[pair[0],pairs[:,0]] - 
                    K[pair[0],pairs[:,1]] - 
                    K[pair[1],pairs[:,0]] + 
                    K[pair[1],pairs[:,1]])

    return mat, y, pairs

def gen_pairs(scores,m):
    pairs = []
    
    levels = scores.copy()
    bot = np.min(levels)
    top = np.max(levels)
    step = (top-bot)/m
    for i in range(m):
        levels[np.logical_and(levels>=i*step+bot,
                              levels<=(i+1)*step+bot)] = i + 1
        
    for i in range(len(levels)):
        idx = np.where(levels[i]>levels)[0]
        for j in idx:
            pairs.append([i,j])

    return np.array(pairs)

    #IDEA 2
    #full ranking svm
    #down_idx = np.sort(np.argsort(train_R)[np.arange(0,l,l//100)]) #downsampling
    #down_idx = np.random.choice(np.arange(l),100)
    #pairs = np.array([[a,b] for [a,b] in itertools.combinations(down_idx,2)])
    #matr,rv = gen_ranking_matrix(train_k[down_idx,:][:,down_idx],train_G[down_idx])
    m = p.n_levels
    matr,rv, pairs = gen_ranking_matrix(train_k,train_R,m)
    
    for i in range(l):
        train_g[i] = np.sum(svm.dual_coef_*(train_k[pairs[svm.support_,0],i] -
                                            train_k[pairs[svm.support_,1],i]))
    for i in range(l):
        if np.sum(train_g[i] > train_g)/l < alpha:
            train_class[i] = -1
            
        #IDEA 2
        svs = np.unique(pairs[svm.support_])
        test_k = make_kmat(test_data,train_data[svs,:,:])
        test_g = np.empty(l)
        test_R = np.empty(l)
        test_class = np.ones(l)
        new_pairs = np.empty_like(pairs[svm.support_])
        for i,pair in enumerate(pairs[svm.support_]):
            a, = np.where(svs==pair[0])
            b, = np.where(svs==pair[1])
            new_pairs[i,:] = [a,b]
            
        for i in range(l):
            test_g[i] = np.sum(svm.dual_coef_*(test_k[i,new_pairs[:,0]] -
                                               test_k[i,new_pairs[:,1]]))
            test_R[i] = np.sum(test_g[i] > train_g)/l
            
        test_class = 2*(test_R > alpha) - 1
        
        plt.figure()
        plt.suptitle('g score comparison')
        plt.subplot(131)
        plt.title('training g score')
        for i in range(l):
            plt.plot(train_data[i,:,0],train_data[i,:,1],'.',c=plt.cm.Blues(train_g[i]))
        plt.subplot(132)
        plt.title('training R score')
        for i in range(l):
            plt.plot(train_data[i,:,0],train_data[i,:,1],'.',c=plt.cm.Blues(train_R[i]))
        plt.subplot(133)
        plt.title('testing g score')
        for i in range(l):
            plt.plot(test_data[i,:,0],test_data[i,:,1],'.',c=plt.cm.Blues(test_g[i]))
            
            
def gen_ranking_matrix(K,scores,m):

    pairs = []
    y=[]
    levels = scores.copy()
    bot = np.min(levels)
    top = np.max(levels)
    step = (top-bot)/m
    for i in range(m):
        levels[np.logical_and(levels>=i*step+bot,
                              levels<=(i+1)*step+bot)] = i + 1
    
    for (i,j) in itertools.combinations(range(scores.shape[0]),r=2):
        if levels[i]>levels[j]:
            y.append(1)
            pairs.append([i,j])
        elif levels[i]<levels[j]:
            y.append(-1)
            pairs.append([i,j])
            
    pairs = np.array(pairs)
    y = np.array(y)
    mat = np.empty((pairs.shape[0],pairs.shape[0]))
    for i,pair in enumerate(pairs):
        mat[i,:] = (K[pair[0],pairs[:,0]] - 
                    K[pair[0],pairs[:,1]] - 
                    K[pair[1],pairs[:,0]] + 
                    K[pair[1],pairs[:,1]])

    return mat, y, pairs




    C = 1
    m = p.n_levels
    matr,rv, pairs = gen_ranking_matrix(train_k,R,m)
    print()
    svm = SVC(C=C, kernel = 'precomputed',verbose=True)
    svm.fit(matr,rv)
    print()
    train_g = np.empty(l)
    for i in range(l):
        train_g[i] = np.sum(svm.dual_coef_*(train_k[pairs[svm.support_,0],i] -
                                            train_k[pairs[svm.support_,1],i]))
    R_hat = np.empty(l)
    for i in range(l):
        R_hat[i] = np.sum(train_g[i] > train_g)/l
    train_class = np.sign(R_hat - alpha)