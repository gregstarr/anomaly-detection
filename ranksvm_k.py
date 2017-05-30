#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:37:23 2017

rankSVM implementation
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, minres

def ranksvm_k(K,A,C,beta=None,
              iter_max_newton=100,
              prec=1.0e-4,
              cg_prec=1.0e-3,
              cg_it=50):
    
    def obj_fun_linear(w,C,out):
        out = np.fmax(0,out)
        obj = np.dot(C[:,None].T,out**2)/2 + np.dot(w.T,K.dot(w))/2
        grad = K.dot(w) - np.dot(np.multiply(C[:,None],out).T*A,K).T
        sv = out>0
        return obj,grad,sv
    
    def line_search_linear(w,d,out,C):
        t = 0
        Kd = A*(K.dot(d[:,None]))
        wd = K.dot(w).T.dot(d[:,None])
        dd = K.dot(d[:,None]).T.dot(K.dot(d[:,None]))
        while True:
            out2 = out - t*Kd
            sv,foo = np.where(out2>0)
            g = wd + t*dd - np.dot(np.multiply(C[sv,None],out2[sv]).T,Kd[sv])
            h = dd + np.dot(Kd[sv].T,np.multiply(Kd[sv],C[sv,None]))
            t = t - g/h
            if g**2/h < 1.0e-10: 
                break
        out = out2.copy()
        return t,out
    
    
    n = K.shape[0]
    
    if beta is None:
        beta = np.zeros(n)[:,None]
        
    it = 0
    out = 1 - A*K.dot(beta)
    
    while True:
        
        it += 1
        if it > iter_max_newton:
            print('max number of newton steps')
            break
        
        obj,grad,sv = obj_fun_linear(beta,C,out)
        
        def hess_vect_mult(w):
            y = K.dot(w[:,None])
            z = np.multiply(np.multiply(C[:,None],sv),A.dot(y))
            y = y + np.dot(z.T*A,K).T
            return y
        
        lo = LinearOperator(K.shape, matvec=hess_vect_mult)
        
        step,info = minres(lo,-grad)
        
        t,out = line_search_linear(beta,step,out,C)
        
        actual_sv = np.sum(A[np.where(sv)[0],:] != 0,0)>0
        
        beta = beta + t*step[:,None]
                
        print('It = {:2d}, Ob = {:10.4f}, N sv = {:4d}, Nsv2 = {:6d}, Decr: {:10.5f}, Lsearch: {:5.3f}'.format(it,obj[0,0],np.sum(actual_sv),np.sum(out<0),(-step.T.dot(grad)/2)[0],t[0,0]))
                
        if -step.T.dot(grad) < prec * obj:
            break
        
    return beta,actual_sv