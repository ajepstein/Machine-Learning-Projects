#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:12:34 2020

@author: adamepstein
"""

#I used the columns TransactionAmt, card4, and card6. 
#I also used all fraudulent rows and 20000 non-fraudulent rows
#I standardized the data on line 50
#I got a score of 0.365567 

import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.linear_model
import sklearn.neighbors



#df1 = pd.read_csv('/Users/adamepstein/Documents/Math 373/ieee-fraud-detection/train_identity.csv', delimiter=',')

#%%
df2 = pd.read_csv('/Users/adamepstein/Downloads/ieee-fraud-detection/train_transaction.csv', delimiter  = ',')
#%%
#df2 = pd.read_csv('/Users/adamepstein/Documents/Math 373/ieee-fraud-detection/train_transaction.csv', delimiter = ',')
#%%
#transactionamt

#x = df2['TransactionAmt']

Y_train = df2['isFraud']

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
    
X_train = prepareFraudData(df2)     
#%%    
df_test = pd.read_csv('/Users/adamepstein/Downloads/ieee-fraud-detection/test_transaction.csv', delimiter = ',')
X_test = prepareFraudData(df_test)# Create and train a classifier   
X_test['debit or credit'] = 0
#remember to give X_test a column named 'debit or credit'
#column of all 0's
#%%  
df2_fraud = df2[df2['isFraud'] == 1]
Y_train_fraud = Y_train[df2['isFraud'] == 1]
df2_notFraud = df2[df2['isFraud'] == 0]

df2_notFraud = df2_notFraud[0:20000]
Y_train_notFraud = Y_train[df2['isFraud'] == 0]
Y_train_notFraud = Y_train_notFraud[0:20000]
df2_reduced = pd.concat([df2_fraud,df2_notFraud],ignore_index=True)
Y_train_reduced = pd.concat([Y_train_fraud,Y_train_notFraud],ignore_index=True)

X_train_reduced = prepareFraudData(df2_reduced)
X_train_reduced['charge card'] = 0
#%%
model = sk.linear_model.LogisticRegression(solver = 'lbfgs', penalty = 'none') # 'none' means we're not using regularization

model.fit(X_train_reduced,Y_train_reduced)
#%%
predictions = model.predict(X_test)
#%%

# Create a kaggle submission.
kaggleSubmission = df_test[['TransactionID']]
kaggleSubmission['isFraud'] = predictions
kaggleSubmission.to_csv('/Users/adamepstein/Downloads/ieee-fraud-detection/kaggleFraudSubmission.csv',index = False)
