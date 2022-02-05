# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:34:53 2022

@author: micha

Script containing the class Model which is used to create decision tree models, and calculate the models 
accuracy when applied to the training and validation set.

Classes:
    Model  
"""

import sys
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

class Model():
    """
    A class for creating decision tree models. Example of creating instance of this class:
    svm = Model(targetVar, train, xTest, yTest, xValid, yValid)
    
    Attributes
    ----------
    targetVarColName : string 
        Name of target variable column in pandas dataframe
    train: pandas dataframe
        Pandas dataframe of training set 
    xTest: pandas dataframe
        Pandas dataframe of x (Explanatory) variables of test set 
    yTest: pandas dataframe
        Pandas dataframe of y (Target) variable of test set 
    xValid: pandas dataframe
        Pandas dataframe of x (Explanatory) variables of validation set
    yValid: pandas dataframe
        Pandas dataframe of y (Target) variable of validation set 
       
    Methods
    -------
    createModel():
        Creates a decision tree model (fitted to the training set)
    modelAccuracy():
        Gets decimal accuracy of model applied to the test set
    validAccuracy():
        Gets decimal accuracy of model applied to the valid set
    
     
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
    
    
    def createModel(self, nTrees):
        """
        Function for creating decision tree models. The created model will be saved to this instance of this object and 
        nothing will be returned. 

        Parameters
        ----------
        nTrees: int
          This is not used in this class, but it is ncessary to have it as the createModel in the random forest class requires
          nTrees to be provided

        Returns
        -------
        None.

        """
        dtModel = DecisionTreeClassifier()
        self.dt = dtModel.fit(self.train.iloc[:,:-1], self.train.iloc[:,-1])
        
    def modelAccuracy(self):
        """
        Returns decimal accuracy of decision tree model applied to the Test set. E.g. 0.9210...

        Returns
        -------
        float
            Decimal accuracy of the model applied to the test set.  E.g. 0.9210...

        """
        ## Make predictions using the model
        yTestPred = self.dt.predict(self.xTest) 
        ## Return the accuracy of the model when applied to the test set 
        return (metrics.accuracy_score(self.yTest, yTestPred)) 
    
    def validAccuracy(self):
        """
        Return decimal accuracy of decsion tree model when applied to the Validation set

        Returns
        -------
        float
            decimal accuracy of support vector machine model applied to validation set

        """
        ## Make predictions using the model 
        yValidPred = self.dt.predict(self.xValid)
        ## Return the accuracy of the model when applied ot the validation set 
        return (metrics.accuracy_score(self.yValid, yValidPred))
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    