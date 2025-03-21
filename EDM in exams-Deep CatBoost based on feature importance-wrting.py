#!/usr/bin/env python
# coding: utf-8

# https://archive.ics.uci.edu/ml/datasets/student+performance

# In[1]:


import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.utils import shuffle
from scipy.stats import ranksums
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
import warnings
from math import log2
from time import time

def KFoldCV(model, data, n_fold=10):
#     num = len(data)
    diff = int(len(data)/n_fold)
    results = np.zeros((n_fold, 4))
    for i in range(n_fold):
        begin = diff*i
        end = diff*(i+1)
#         if i == n_fold-1:
#             end = -1
        test = data[begin:end]
        train = deepcopy(data)
        train = np.delete(train, range(begin, end),axis=0)
        X_train, y_train = train[:,:-1], train[:,-1]
        X_test, y = test[:,:-1], test[:,-1]
        predictY = model.fit(X_train, y_train).predict(X_test)
        mae = np.mean(abs((y-predictY)))
        stdErr = np.std(((y-predictY)))
        error=sum((y-predictY)**2)
        RMSE=np.sqrt(error/len(y))
        MAC = np.dot(y,predictY)**2/(np.dot(y, y)*np.dot(predictY, predictY))
#         print(mae, stdErr, RMSE, MAC)
        results[i,:] = [mae, stdErr, RMSE, MAC]
    return results

warnings.filterwarnings('ignore')

data=pd.read_csv('./exams.csv')
data['writing_score']=data['writing score']
data.drop(columns=['math score'],inplace=True)
data.drop(columns=['reading score'],inplace=True)
data.drop(columns=['writing score'],inplace=True)

cats = ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
       'test preparation course']
oe = OrdinalEncoder()
for c in cats:
    oe.fit(data[c].values.reshape(-1,1))
    data[c] = np.asarray(oe.transform(data[c].values.reshape(-1,1)),dtype=int)
#     np.asarray(data['gender'],dtype=int)
data = data.values
data = shuffle(data)



from sklearn.model_selection import GridSearchCV


def deepCatBoostForward(train, test,importances,ind,n_estimators, max_depth):
#     ind = ind[::-1]
    s = 0
    inds = []
    inds2 = []
    inds3 = []
    for i in ind:
        s += importances[i]
        if s <= 0.05:
            inds.append(i)
        elif s <= 0.1:
            inds2.append(i)
        else:
            inds3.append(i)

    # First layer
    if len(inds) > 0:
        model = gridSearch4CatBoost(train[:, inds], train[:, -1], n_estimators, max_depth)
        model.fit(train[:, inds], train[:, -1])
        predY = model.predict(train[:, inds])
        predY2 = model.predict(test[:, inds])

    # Second layer
    if len(inds2) > 0:
        if len(inds) == 0 and len(inds2) == 1:
            tempTrain = train[:, inds2].reshape(-1, 1)
            tempTest = test[:, inds2].reshape(-1, 1)
        else:
            tempTrain = np.hstack((train[:, inds2], predY.reshape(-1, 1)))
            tempTest = np.hstack((test[:, inds2], predY2.reshape(-1, 1)))

        model = gridSearch4CatBoost(tempTrain, train[:, -1], n_estimators, max_depth)
        model.fit(tempTrain, train[:, -1])
        predYY = model.predict(tempTrain)
        predYY2 = model.predict(tempTest)

    # Third layer
    if len(inds3) > 0:
        if len(inds2) <= 1 and len(inds3) == 0:
            tempTrain = np.hstack((train[:, inds3], predY.reshape(-1, 1), predYY.reshape(-1, 1)))
            tempTest = np.hstack((test[:, inds3], predY2.reshape(-1, 1), predYY2.reshape(-1, 1)))
        else:
            tempTrain = train[:, inds3]
            tempTest = test[:, inds3]

        model = gridSearch4CatBoost(tempTrain, train[:, -1], n_estimators, max_depth)
        model.fit(tempTrain, train[:, -1])
        predY2 = model.predict(tempTest)
    return predY2

def gridSearch4CatBoost(X,y,n_estimators, max_depth):
#     cbc = CatBoostRegressor(verbose=False,task_type="GPU",devices='0:1')
# #     cbc = RandomForestRegressor()
#     grid = {'max_depth': [3,4,5],'n_estimators':[100,300,500]}
#     gscv = GridSearchCV (estimator = cbc, param_grid = grid, cv = 10)
#     gscv.fit(X,y)
#     return gscv.best_estimator_
    cbc = CatBoostRegressor(n_estimators=n_estimators, max_depth=max_depth,verbose=False,task_type="GPU",devices='0:1').fit(X,y)
    return cbc
#     return model.fit(X,y).predict(testX)

def KFoldForward(data, n_estimators, max_depth, n_fold=10):
#     num = len(data)
    diff = int(len(data)/n_fold)
    results = np.zeros((n_fold, 4))
    for i in range(n_fold):
        begin = diff*i
        end = diff*(i+1)
#         if i == n_fold-1:
#             end = -1
        test = data[begin:end]
        train = deepcopy(data)
        train = np.delete(train, range(begin, end),axis=0)
        X_train, y_train = train[:,:-1], train[:,-1]
        
        model = RandomForestRegressor().fit(X_train, y_train)
        importances = model.feature_importances_
        ind = np.argsort(importances)
        
        X_test, y = test[:,:-1], test[:,-1]
#         predictY = model.fit(X_train, y_train).predict(X_test)
        predictY = deepCatBoostForward(train, test,importances,ind,n_estimators, max_depth)
        mae = np.mean(abs((y-predictY)))
        stdErr = np.std(((y-predictY)))
        error=sum((y-predictY)**2)
        RMSE=np.sqrt(error/len(y))
        MAC = np.dot(y,predictY)**2/(np.dot(y, y)*np.dot(predictY, predictY))
#         print(mae, stdErr, RMSE, MAC)
        results[i,:] = [mae, stdErr, RMSE, MAC]
    return results


t1 = time()
n_estimators= [100,300,500]
max_depth = [3,4, 5]
epochs=1
cats = np.zeros((epochs,4))
for i in n_estimators:
    for j in max_depth:
        for epoch in range(epochs):
            data = shuffle(data)
            result = KFoldForward(data,n_estimators=i, max_depth=j)
            cats[epoch,:] = np.mean(result, axis=0)
        print(np.mean(cats, axis=0))
print(time()-t1)


n_estimators  = [500]
max_depth = [4]
epochs=20
catsf = np.zeros((epochs,4))
for i in n_estimators:
    for j in max_depth:
        for epoch in range(epochs):
            data = shuffle(data)
            model = RandomForestRegressor().fit(data[:,:-1], data[:,-1])
            importances = model.feature_importances_
            ind = np.argsort(importances)
            result = KFoldForward(data, n_estimators=i, max_depth=j)
            catsf[epoch,:] = np.mean(result, axis=0)
        print(np.mean(catsf, axis=0))


print(mlps.std(axis=0))
print(svms.std(axis=0))
print(rfs.std(axis=0))
print(xgbs.std(axis=0))
print(cats.std(axis=0))
print(catsf.std(axis=0))


def getTestResult(f1,f2,i):
    print(ranksums(np.asarray(f1[:,i]), np.asarray(f2[:,i])).pvalue)

i = 0
print(getTestResult(mlps, catsf,i))
print(getTestResult(svms, catsf,i))
print(getTestResult(rfs, catsf,i))
print(getTestResult(xgbs, catsf,i))
print(getTestResult(cats, catsf,i))

i = 2
print(getTestResult(mlps, catsf,i))
print(getTestResult(svms, catsf,i))
print(getTestResult(rfs, catsf,i))
print(getTestResult(xgbs, catsf,i))
print(getTestResult(cats, catsf,i))


# In[ ]:




