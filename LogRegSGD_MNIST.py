#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:29:28 2020

@author: adamepstein
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%

def softmax(u):
    expu = np.exp(u)
    return expu/np.sum(expu)

def crossEntropy(p,q):
    return -np.vdot(p,np.log(q))

def eval_L(X,Y,beta):
    
    N = X.shape[0]
    L = 0.0
    
    for i in range(N):
        XiHat = X[i]
        Yi = Y[i]
        qi = softmax(beta @ XiHat)
        
        L += crossEntropy(Yi, qi)
    return L        

def LogRegGD(X, Y, alpha):
    maxIterations = 20
    # alpha = 0.001 
    beta = np.random.randn(K, d + 1)
    gradNorms = []
    N = X.shape[0]
    allOnes = np.ones((N, 1))
    X = np.hstack((allOnes, X))
        
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

def logRegSGD(X,Y,alpha):
    numEpochs = 20
    N, d = X.shape
   
    X = np.insert(X,0,1,axis = 1)
    K = Y.shape[1]
    
    beta = np.zeros((K,d+1))
    Lvals = []
    
    for ep in range(numEpochs):
        
        
        L = eval_L(X, Y, beta)
        Lvals.append(L)
        
        print("Epoch is: " + str(ep) + " Cost is: " + str(L))
        
        
        
        prm = np.random.permutation(N)
        
        for i in prm:
            XiHat = X[i]
            Yi = Y[i]
            
            
            qi = softmax(beta @ XiHat) #the @ symbol is dot product--all k dot products at once
            
            
            grad_Li = np.outer(qi - Yi,XiHat)
            
            beta = beta - alpha * grad_Li
    
         
    
    return beta, Lvals

def predictLabels(X,beta):
    
    X = np.insert(X,0,1,axis = 1)
    N = X.shape[0]
    
    predictions= []
    probabilities = []
    
    for i in range(N):
        XiHat = X[i]
        
        qi = softmax(beta @ XiHat)
        
        k = np.argmax(qi)
        predictions.append(k)
        probabilities.append(np.max(qi))
        
    return predictions, probabilities 
    
            
 
#%%
np.random.seed(7)

(X_train,Y_train), (X_test,Y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train/255.0
X_test = X_test/255.0

#reshape the data so each image is a vector w/ 28 entries
N_train, numRows, numCols = X_train.shape
X_train = np.reshape(X_train, (N_train,numRows*numCols))

Y_train = pd.get_dummies(Y_train).values
K = Y_train.shape[1]

N, d = X_train.shape
#%%  
plt.figure()   
alpha = [0.00001,0.0001, 0.0005]
#0.0005 is the best alpha level for 20 epochs
for lrate in alpha:
    print("SGD")
    print("At alpha level: " + str(lrate))
    beta, lVals = logRegSGD(X_train,Y_train,lrate)
    plt.semilogy(lVals)
    plt.xlabel('Epoch')
    plt.ylabel('Cost Function Value')
    plt.title("Cost Function Value over Each Epoch")
#%%
plt.figure()   
alpha = [0.00001,0.0001, 0.0005]
#seems like 0.00001 is the best alpha level for 20 iterations
for lrate in alpha:
    print("Gradient Descent")
    print("At alpha level: " + str(lrate))
    beta, gradNorms = LogRegGD(X_train, Y_train, lrate)
    plt.semilogy(gradNorms)
    plt.xlabel("Number of Iterations")
    plt.ylabel("The Norm Of The Gradient")
    plt.title("Norm of Gradient over Each Iteration")
#%%
plt.figure()
alpha = [0.00001,0.0001, 0.0005]
for lrate in alpha:
    print("SGD VS Gradient Descent")
    print("At alpha level: " + str(lrate))
    #Stochastic Gradient Descent seems faster and more efficient
    beta, gradNorms = LogRegGD(X_train, Y_train, lrate)
    beta, lVals = logRegSGD(X_train,Y_train,lrate)
    plt.semilogy(gradNorms)
    plt.semilogy(lVals)
    plt.title("SGD vs Gradient Descent")
#Orange, red, and brown are for SGD
#blue, green, and purple are for gradient descent    
#%%    
N_test = X_test.shape[0]
X_test = np.reshape(X_test, (N_test, numRows*numCols))
preds, probs = predictLabels(X_test,beta) 
allOnes = np.ones((N_test, 1))
X_test = np.hstack((allOnes, X_test))
#%%
numSuccess = 0
for i in range(N_test):
        
    XiHat = X_test[i, :]
        
    Yi = Y_test[i]
    
    qi = softmax(beta @ XiHat)
        
    k = np.argmax(qi)
               
    if k == Yi:
        numSuccess += 1    
        
print("The accuracy after multiclass logistic regression is: " + str(numSuccess/N_test))
          
    
            
    
