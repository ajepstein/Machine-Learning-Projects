#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:19:11 2020

@author: adamepstein
"""

import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.linear_model
import sklearn.neighbors
import matplotlib.pyplot as plt 

fraud_data = pd.read_csv('/Users/adamepstein/Downloads/ieee-fraud-detection/train_transaction.csv', delimiter  = ',')

Y_train = fraud_data['isFraud']

def prepareFraudData(df):
    cols = ['TransactionAmt', 'card4', 'card6']
    X = df[cols]
    
    oneHot = pd.get_dummies(X['card4'])
    X = X.drop(['card4'], axis = 1)
    X = X.join(oneHot)
    
    
    oneHot2 = pd.get_dummies(X['card6'])
    X = X.drop(['card6'], axis = 1)
    #X['credit'] = oneHot2['credit']
    #X['debit'] = oneHot2['debit']
    X = X.join(oneHot2)

    X['TransactionAmt'] = X['TransactionAmt'].fillna(X['TransactionAmt'].mean())
    
    X = (X - X.mean())/X.std()    
    return X
    
X_train = prepareFraudData(fraud_data)  

df_test = pd.read_csv('/Users/adamepstein/Downloads/ieee-fraud-detection/test_transaction.csv', delimiter = ',')
X_test = prepareFraudData(df_test)# Create and train a classifier   
X_test['debit or credit'] = 0
#remember to give X_test a column named 'debit or credit'
#column of all 0's
#%%
df_fraud = fraud_data[fraud_data['isFraud'] == 1]
Y_train_fraud = Y_train[fraud_data['isFraud'] == 1]
df_notFraud = fraud_data[fraud_data['isFraud'] == 0]

df_notFraud = df_notFraud[0:20000]
Y_train_notFraud = Y_train[fraud_data['isFraud'] == 0]
Y_train_notFraud = Y_train_notFraud[0:20000]
df_reduced = pd.concat([df_fraud,df_notFraud],ignore_index=True)
Y_train_reduced = pd.concat([Y_train_fraud,Y_train_notFraud],ignore_index=True)

X_train_reduced = prepareFraudData(df_reduced)
X_train_reduced['charge card'] = 0
#%%  
def sigmoid(u):
    return np.exp(u)/(1 + np.exp(u))

def h(u, yi):
    exp = np.exp(u)
    return -yi * u + np.log(1+exp)

def L(beta, X, Y):
    N = X.shape[0]
    #col = X.shape[1]
    mySumHi = 0
    for i in range(N):
        xihat = X[i]
        yi = Y[i]
        dotProduct = np.vdot(xihat, beta)
        mySumHi += h(dotProduct, yi) 
    return mySumHi        

def LogReg(X, Y):
    N, d = X.shape
    allOnes = np.ones((N, 1))
    X = np.hstack((allOnes, X)) 
    
    alpha = 0.00001
    beta = np.random.randn(d+1)
    listNormGrad = []
    listLBetas = []
    
    maxiterations = 100
    for idx in range(maxiterations):
        gradient = np.zeros(d+1)
            
        for i in range(N):
            Xi = X[i, :]
            Yi = Y[i]
            qi = sigmoid(np.vdot(Xi, beta))

            
            gradient += (qi - Yi) * Xi
            
            
        norm_gradient = np.linalg.norm(gradient)
        listNormGrad.append(norm_gradient)
        beta = beta - alpha * gradient
        LBeta = L(beta, X, Y)
        listLBetas.append(LBeta)
        print(idx, LBeta)    
    
    
    return beta, listNormGrad, listLBetas     
            


beta, myListNormGrad, listLBetas = LogReg(X_train_reduced,Y_train_reduced)
plt.figure()
plt.semilogy(myListNormGrad)
plt.semilogy(listLBetas)
#%%  

def predict(X, beta):
    N = X.shape[0]
    allOnes = np.ones((N, 1))
    X = np.hstack((allOnes, X))
    predictions = []
    for i in range (N):
        Xi = X[i, :]
        qi = sigmoid(np.vdot(Xi, beta))
        if(qi > 0.5):
            predictions.append(1)
        else:
            predictions.append(0)
            
    return predictions

predictions = predict(X_test, beta)            
        
#%% 
kaggleSubmission = df_test[['TransactionID']]
kaggleSubmission['isFraud'] = predictions
kaggleSubmission.to_csv('/Users/adamepstein/Downloads/ieee-fraud-detection/FraudSubmission.csv',index = False)



   