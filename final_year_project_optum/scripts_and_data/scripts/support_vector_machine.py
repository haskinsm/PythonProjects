# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 21:17:37 2022

@author: micha

Script containing the class Model which is used to create support vector machine models, and calculate the models 
accuracy when applied to the training and validation set.

Classes:
    Model  
"""

from sklearn import svm
from sklearn import metrics
import sys

class Model():
    """
    A class for creating support vector machine models. Example of creating instance of this class:
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
        Creates a support vector machine model (fitted to the training set)
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
                   " in the master script. E.g:  svm = Model(targetVar, train, xTest, yTest, xValid, yValid) ")
             print("\nProgram Execution aborted. Goodbye.")
             sys.exit() ## This will terminate the script execution
            
            
    def createModel(self, nTrees):
        """
        Function for creating support vector machine models. The created model will be saved to this instance of this object and 
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
        svmModel = svm.LinearSVC() # Use Linear SVC kernel to avoid convergence issues (Also much faster) 
        
        ## Train the model using the training sets
        self.svm = svmModel.fit(self.train.iloc[:,:-1], self.train.iloc[:,-1]) # x var and y target var 
        
        ## Make predictions using the model now to prevent it being made several times  
        # ... prevents double predictions being calculated when F1 and Accuracy functions are both called. 
        self.yTestPred = self.svm.predict(self.xTest)
        self.yValidPred = self.svm.predict(self.xValid)
        
        
        
    def modelAccuracy(self):
        """
        Returns decimal accuracy of support vector machine model applied to the Test set. E.g. 0.9210...

        Returns
        -------
        float
            Decimal accuracy of the model applied to the test set.  E.g. 0.9210...

        """
        ## Return the accuracy of the model when applied to the test set 
        return (metrics.accuracy_score(self.yTest, self.yTestPred)) 
    
    
    def validAccuracy(self):
        """
        Return decimal accuracy of support vector machine model when applied to the Validation set

        Returns
        -------
        float
            decimal accuracy of support vector machine model applied to validation set

        """
        ## Return the accuracy of the model when applied ot the validation set 
        return (metrics.accuracy_score(self.yValid, self.yValidPred))


    def validF1Score(self):
        """
        Returns F1 score of the model when applied to the validation set

        Returns
        -------
        None.

        """
        ## Return the accuracy of the model when applied ot the validation set 
        return (metrics.f1_score(self.yValid, self.yValidPred))
    
    
    def validAUC(self):
        """
        Returns AUC of the model as a decimal when applied to the validation set 

        Returns
        -------
        float
            decimal AUC of the ROC curve for the model applied to the validation set

        """
        ## Return the AUC of the model when applied to the validation set 
        return (metrics.roc_auc_score(self.yValid, self.yValidPred))
    
    
    
    
    