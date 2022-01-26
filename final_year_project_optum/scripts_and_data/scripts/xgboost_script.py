# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:50:20 2022

@author: micha
"""

import xgboost 
#import pandas as pd
#import numpy as np 
import sys
from sklearn import metrics

class Model():
    """
    A class for creating xgboost models. Create instance of this class by passing in targetVarColName and a 
    Train, xTest, yTest, xValid and yValid datasets
    i.e. creat an instance of this class in the master script with the following code
    xgb = Model(targetVar, train, xTest, yTest, xValid, yValid)
    """
    def __init__(self, *args):
        if len(args) == 6:
            self.targetVar = args[0]
            self.train = args[1]
            self.xTest = args[2]
            self.yTest = args[3]
            self.xValid = args[4]
            self.yValid = args[5]
        else:
            print("You must pass in the target variable columns name, and a training, XTest, yTest, XValid and yValid pandas dataframes" + 
                  " in the master script. E.g:  xgb = Model(targetVar, train, xTest, yTest, xValid, yValid) ")
            print("\nProgram Execution aborted. Goodbye.")
            sys.exit() ## This will terminate the script execution
    
    
    def createModel(self):
        # src: https://xgboost.readthedocs.io/en/latest/python/examples/sklearn_evals_result.html#sphx-glr-python-examples-sklearn-evals-result-py
        params = {
            'objective': "binary:logistic"
            }
        xgbModel = xgboost.XGBClassifier(**params)
        # About the parameters above:
            # the default value booster = "gbtree", sets the type of model to run at each iteration as a tree-based model
            # note xgboost has a default seed of 0
        self.xgb = xgbModel.fit(self.train.iloc[:,:-1], self.train.iloc[:,-1], 
                                eval_set = [(self.train.iloc[:,:-1], self.train.iloc[:,-1]), (self.xTest, self.yTest)],
                                eval_metric = 'error', verbose = 0)
            # the eval_metric error is used for binary classification
            # verbose = 0 stops each iterations accuracy being outputted to the console
        #evals_result = xgbModel.evals_result()
        #print(evals_result)
        
    def modelAccuracy(self):
        """ Returns decimal accuracy of xgboost model applied to the Test set. E.g. 0.9210..."""
        ## Make predictions using the model
        yTestPred = self.xgb.predict(self.xTest) 
        ## Return the accuracy of the model when applied to the test set 
        return (metrics.accuracy_score(self.yTest, yTestPred)) 
    
    def validAccuracy(self):
       """ Return decimal accuracy of xgboost model when applied to the Validation set """ 
       ## Make predictions using the model 
       yValidPred = self.xgb.predict(self.xValid)
       ## Return the accuracy of the model when applied ot the validation set 
       return (metrics.accuracy_score(self.yValid, yValidPred))
        
    
    
    
    
    
    
    
    
    
    
    
    
    