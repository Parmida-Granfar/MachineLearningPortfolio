#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:13:36 2020
This is a program based on elitedatascience.com project
@author: parmida
"""
#first we input all the libraries and modules
import sklearn 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
#this is for cross validation 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
#this one is for evaluating our model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
print (data.head())
print (data.shape)
print (data.describe())
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)

#this part ifs for data preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
'''print (X_train_scaled.mean(axis=0))
print( X_train_scaled.std(axis=0))
X_test_scaled = scaler.transform(X_test)
print( X_test_scaled.mean(axis=0))
print (X_test_scaled.std(axis=0))'''
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))
print (pipeline.get_params())
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)
print (clf.best_params_)
print (clf.refit)
pred =( clf.predict(X_test))
print (r2_score(y_test, pred))
print (mean_squared_error(y_test, pred))
 
# 10. Save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')
