# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:34:53 2022

@author: micha
"""

import sys
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

class Model():
    """
    A class for creating dcesion tree models. Create instance of this class by passing in targetVarColName and a 
    Train, xTest, yTest, xValid and yValid datasets
    i.e. creat an instance of this class in the master script with the following code
    dt = Model(targetVar, train, xTest, yTest, xValid, yValid)
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
                  " in the master script. E.g:  dt = Model(targetVar, train, xTest, yTest, xValid, yValid) ")
            print("\nProgram Execution aborted. Goodbye.")
            sys.exit() ## This will terminate the script execution
    
    
    def createModel(self):
        dtModel = DecisionTreeClassifier()
        self.dt = dtModel.fit(self.train.iloc[:,:-1], self.train.iloc[:,-1])
        
    def modelAccuracy(self):
        """ Returns decimal accuracy of decsion tree model applied to the Test set. E.g. 0.9210..."""
        ## Make predictions using the model
        yTestPred = self.dt.predict(self.xTest) 
        ## Return the accuracy of the model when applied to the test set 
        return (metrics.accuracy_score(self.yTest, yTestPred)) 
    
    def validAccuracy(self):
       """ Return decimal accuracy of decsion tree model when applied to the Validation set """ 
       ## Make predictions using the model 
       yValidPred = self.dt.predict(self.xValid)
       ## Return the accuracy of the model when applied ot the validation set 
       return (metrics.accuracy_score(self.yValid, yValidPred))
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    