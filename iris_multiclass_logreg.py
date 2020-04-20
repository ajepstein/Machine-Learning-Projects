#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:05:38 2020

@author: adamepstein
"""


import pandas as pd
import sklearn as sk
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt 
import sklearn.model_selection 


# load and read in iris dataset
iris = pd.read_csv('/Users/adamepstein/Documents/Iris.csv')

#%%
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = iris[features].values
Y = iris['Species'].values

#%%
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, train_size = 0.6, test_size = 0.4)

numTrain = X_train.shape[0]
numTest = X_test.shape[0]

Y_oneHot = pd.get_dummies(Y_train).values
K = Y_oneHot.shape[1]

N, d = X_train.shape
allOnes = np.ones((N, 1))
X_train = np.hstack((allOnes, X_train))

#%%
def softmax(u):
    expu = np.exp(u)
    return expu/np.sum(expu)

def h(u, yi):
    exp = np.exp(u)
    return -yi * u + np.log(1 + exp)

def L(beta, X, Y):
    N = X.shape[0] 
    mySumHi = 0
   
    for i in range(N):
        xihat = X[i] 
        yi = Y[i] 
        xihat = np.insert(xihat, 0, 1, axis = 0)
        dotProduct = beta @ xihat
        mySumHi += h(dotProduct, yi)
    return mySumHi
    
def LogRegGD(X, Y, alpha):
    maxIterations = 500
    # alpha = 0.001 
    beta = np.random.randn(K, d + 1)
    gradNorms = []
    N = X.shape[0]
    # allOnes = np.ones((N, 1))
    # X = np.hstack((allOnes, X))
        
    for idx in range(maxIterations):
        gradient = np.zeros((K, d + 1))
            
        for i in range(N):
                
            XiHat = X[i, :] 
            Yi = Y[i, :]
            qi = softmax(beta @ XiHat)
                
            gradient_i = np.zeros((K, d + 1))
                
            for k in range(K):
                gradient_i[k, :] = (qi[k] - Yi[k]) * XiHat 
            gradient += gradient_i
            
        beta = beta - alpha * gradient
            
        norm_gradient = np.linalg.norm(gradient)
        gradNorms.append(norm_gradient)
    return beta, gradNorms
    

#%% 
plt.figure()   
alpha = [0.00001,0.0001, 0.0005]
#0.0005 is the best alpha level for 500 iterations
for lrate in alpha:
    beta, gradNorms = LogRegGD(X_train, Y_oneHot, lrate)
    plt.semilogy(gradNorms)
    plt.xlabel("Number of Iterations")
    plt.ylabel("The Norm Of The Gradient")

   
N_test = X_test.shape[0]
allOnes = np.ones((N_test, 1))
X_test = np.hstack((allOnes, X_test))

numSuccess = 0
for i in range(N_test):
        
    XiHat = X_test[i, :]
        
    Yi = Y_test[i]
    
    qi = softmax(beta @ XiHat)
        
    k = np.argmax(qi)
            
    if k == 0:
        pred = "Iris-setosa"
    if k == 1:
        pred = "Iris-versicolor"
    if k == 2:
        pred = "Iris-virginica"
            
    if pred == Yi:
        numSuccess += 1    
        
print("The accuracy after multiclass logistic regression is: " + str(numSuccess/N_test))

#High accuracy




