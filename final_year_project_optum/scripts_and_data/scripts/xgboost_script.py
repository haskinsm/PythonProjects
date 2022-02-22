# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:50:20 2022

@author: micha

Script containing the class Model which is used to create XGBoost models, and calculate the models 
accuracy when applied to the training and validation set.

Classes:
    Model 
"""

import xgboost 
import sys
from sklearn import metrics


class Model():
    """
    A class for creating XGBoost models. Example of creating instance of this class:
    xgb = Model(targetVar, train, xTest, yTest, xValid, yValid)
    
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
        Creates a Xgboost model (fitted to the training set)
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
                  " in the master script. E.g:  xgb = Model(targetVar, train, xTest, yTest, xValid, yValid) ")
            print("\nProgram Execution aborted. Goodbye.")
            sys.exit() ## This will terminate the script execution
    
    
    def createModel(self, nTrees = 100):
        """
        Function for creating Xgboost models. The created model will be saved to this instance of the object and 
        nothing will be returned. 

        Parameters
        ----------
        nTrees : int, optional
            DESCRIPTION. Optionally specify the number of estimators to use. The default is 100.

        Returns
        -------
        None.

        """
        # src: https://xgboost.readthedocs.io/en/latest/python/examples/sklearn_evals_result.html#sphx-glr-python-examples-sklearn-evals-result-py
        params = {
            'n_jobs': -1,  # use all processing cores
            'objective': "binary:logistic",# indicate the target variable is binary 
            'n_estimators': nTrees,
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
        
        ## Make predictions using the model now to prevent it being made several times  
        # ... prevents double predictions being calculated when F1 and Accuracy functions are both called. 
        self.yTestPred = self.xgb.predict(self.xTest)
        self.yValidPred = self.xgb.predict(self.xValid)
        
        
    def modelAccuracy(self):
        """
        Returns decimal accuracy of Xgboost model applied to the Test set. E.g. 0.9210...

        Returns
        -------
        float
            Decimal accuracy of the model applied to the test set.  E.g. 0.9210...

        """
        ## Return the accuracy of the model when applied to the test set 
        return (metrics.accuracy_score(self.yTest, self.yTestPred)) 
    
    def validAccuracy(self):
        """
        Return decimal accuracy of Xgboost model when applied to the Validation set

        Returns
        -------
        float
            decimal accuracy of support vector machine model applied to validation set

        """
        ## Return the accuracy of the model when applied ot the validation set 
        return (metrics.accuracy_score(self.yValid, self.yValidPred))
    
    def validF1score(self):
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
        
    
    
    
    
    
    
    
    
    
    
    
    
    