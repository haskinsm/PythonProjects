# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 21:17:37 2022

@author: micha
"""

from sklearn import svm
from sklearn import metrics
import sys

class Model():
    """
    A class for creating support vector machine models. Create instance of this class by passing in targetVarColName and a 
    Train, xTest, yTest, xValid and yValid datasets
    i.e. 
    svm = Model(targetVar, train, xTest, yTest, xValid, yValid)
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
                   " in the master script. E.g:  svm = Model(targetVar, train, xTest, yTest, xValid, yValid) ")
             print("\nProgram Execution aborted. Goodbye.")
             sys.exit() ## This will terminate the script execution
            
            
    def createModel(self, nTrees = 500):
        """ 
        Function for creating support vector machine models
        """
        
        svmModel = svm.SVC()
        
        ## Train the model using the training sets
        self.svm = svmModel.fit(self.train.iloc[:,:-1], self.train.iloc[:,-1]) # x var and y target var 
        
     
        
    def modelAccuracy(self):
        """ Returns decimal accuracy of support vector machine model applied to the Test set. E.g. 0.9210..."""
        ## Make predictions using the model
        yTestPred = self.svm.predict(self.xTest) 
        ## Return the accuracy of the model when applied to the test set 
        return (metrics.accuracy_score(self.yTest, yTestPred)) 
    
    
    def validAccuracy(self):
        """ Return decimal accuracy of support vector machine model when applied to the Validation set """ 
        ## Make predictions using the model 
        yValidPred = self.svm.predict(self.xValid)
        ## Return the accuracy of the model when applied ot the validation set 
        return (metrics.accuracy_score(self.yValid, yValidPred))