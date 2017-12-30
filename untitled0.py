#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 01:29:02 2017

@author: vaibhav
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier as RF
from sklearn.model_selection import train_test_split


def ObjectVariableRectification(data1, data2):
    obj_cols = [x for x in data1.columns if data1[x].dtype == 'object']
    for x in obj_cols:
        data1[x] = pd.get_dummies(data1[x])
        data2[x] = pd.get_dummies(data2[x])
    return data1, data2

    

def main():
    train = pd.read_csv('criminal_train.csv')
    test = pd.read_csv('criminal_test.csv')    
    train, test = ObjectVariableRectification(train, test)
    y = np.array(train['Criminal'], dtype = float)
    X = np.array(train.drop(['Criminal', 'PERID'], axis = 1), dtype = float)
    assert(X.shape[0] == y.shape[0])
    print('-----------------Training------------------\n')
    clf = RF(n_estimators = 12, max_depth = 5)
    clf.fit(X, y)
    print(clf.score(X,y))
    print('\n')
    X_train = np.array(test.drop(['PERID'], axis = 1), dtype = float)
    assert[X.shape[1] == X_train.shape[1]]
    print('----------------Predicting-----------------\n')
    predictions = np.array(clf.predict(X_train), dtype = int)
    print('---------------WRITING THE FILE------------\n')
    filePtr = open('MySubmissions.csv', 'a+')
    filePtr.write('PERID,Criminal\n')
    for i in range(X_train.shape[0]):
        filePtr.write(str(test['PERID'][i]))
        filePtr.write(',')
        filePtr.write(str(predictions[i]))
        filePtr.write('\n')
    print('----------FILE SUCCESSFULY WRITTEN---------\n')
    

if __name__ == '__main__':
    main()