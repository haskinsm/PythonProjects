# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:56:10 2021

@author: micha

sources:
    - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python 
"""
import pandas as pd
import numpy as np
import re
import sklearn
# import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
#'exec(%matplotlib inline)'

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics

# Files I've written
import sklearn_helper as sklHelper #### Make sure your in the general directory for this to work
# i.e. C:\Users\micha\Documents\3rd year\Software Applications\PythonSpyder(Anaconda V)\final_year_project_optum *************


class RandomForest():
    """
    A class for creating random forest models. Create instance of this class by passing in either full dataset or by 
    passing in a training and test dataset
    i.e. 
    rf = RandomForest(<fullDataset>)
    or
    rf = RandomForest(<trainingDataset>, <testingDataset>)
    """
    def __init__(self, *args):
        """ 
        I anticipate two cases: 
            - First being when just given a target var and whole dataset not broken up into training and test datasets
            - Second being when given target var, and a training and test dataset
        """
        if len(args) == 3:
            self.targetVar = args[0]
            self.data = args[1]
            self.yTest = args[2]
        elif len(args) == 4:
            self.targetVar = args[0]
            self.train = args[1]
            self.test = args[2]
            self.yTest = args[3]
            
    def createModel(self):
        """ Function for creating random forest models"""
    
        """
        Parameters that will be used:
        - n_jobs : Number of cores used for the training process. If set to -1, all cores are used.
        - n_estimators : Number of classification trees in your learning model ( set to 10 per default)
        - max_depth : Maximum depth of tree, or how much a node should be expanded. Beware if set to too
         high a number would run the risk of overfitting as one would be growing the tree too deep
        - verbose : Controls whether you want to output any text during the learning process. A value 
         of 0 suppresses all text while a value of 3 outputs the tree learning process at every iteration.
        """
        SEED = 0 ## For reproducibility 
        # Some useful parameters which will come in handy later on
        ntrain = self.train.shape[0]
        ntest = self.test.shape[0]
        NFOLDS = 5 # set folds for out-of-fold prediction
        kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=True) #*********************
        
        rf_params = {
            'n_jobs': -1,  # use all cores
            'n_estimators': 500, # 500 trees
            'warm_start': True, 
             #'max_features': 0.2,
            'max_depth': 6,  # depth of tree
            'min_samples_leaf': 2,
            'max_features' : 'sqrt',
            'verbose': 0
        }
        self.rfHelper = sklHelper.SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
        
        # Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
        self.yTrain = self.train[self.targetVar].ravel() # ravel is used to change a 2-dimensional array or a multi-dimensional array into a contiguous flattened array 
        train = self.train.drop([self.targetVar], axis=1)
        xTrain = train.values   # Creates an array of the train data
        xTest = self.test.values   # Creates an array of the test data
        # Need to convert array to float 32 not default float 62
        self.xTrain = np.nan_to_num(xTrain.astype(np.float32)) # This will convert everything to float 32 and if this results in inf they will be converted to max float 32
        self.xTest = np.nan_to_num(xTest.astype(np.float32)) 
        
        ## Train the model using the training sets
        self.rf = self.rfHelper.fit(self.xTrain, self.yTrain)
        ## Make predictions using the model
        self.yPred = self.rf.predict(self.xTest) 
        
    def modelAccuracy(self):
        ## After training, check the accuracy using actual and predicted values.
        return (metrics.accuracy_score(self.yTest, self.yPred)) #*************************************************
        
    def featureImportance(self):
        
        # Feature importances generated from the different classifiers
        # Now having learned our the first-level classifiers, we can utilise a very nifty feature of the 
        # Sklearn models and that is to output the importances of the various features in the training and
        # test sets with one very simple line of code.
        # As per the Sklearn documentation, most of the classifiers are built in with an attribute which returns feature importances by simply typing in .featureimportances.
       
        # The importance of a feature is computed as the (normalized) total reduction of the criterion 
        # brought by that feature. It is also known as the Gini importance.
        rfFeature = self.rfHelper.feature_importances(self.xTrain, self.yTrain)
        rfFeatures=list(rfFeature)

        train = self.train.drop([self.targetVar], axis=1)
        cols = train.columns.values
        # Create a dataframe with features
        featureDataframe = pd.DataFrame( {'features': cols,
             'Random Forest feature importances': rfFeatures
             })
        
        
        # Scatter plot 
        trace = go.Scatter(
            y = featureDataframe['Random Forest feature importances'].values,
            x = featureDataframe['features'].values,
            mode='markers',
            marker=dict(
                sizemode = 'diameter',
                sizeref = 1,
                size = 25,
                # size= feature_dataframe['AdaBoost feature importances'].values,
                # color = np.random.randn(500), #set color equal to a variable
                color = featureDataframe['Random Forest feature importances'].values,
                colorscale='Portland',
                showscale=True
            ),
            text = featureDataframe['features'].values
        )
        data = [trace]
        layout= go.Layout(
            autosize= True,
            title= 'Random Forest Feature Importance',
            hovermode= 'closest',
        #     xaxis= dict(
        #         title= 'Pop',
        #         ticklen= 5,
        #         zeroline= False,
        #         gridwidth= 2,
        #     ),
            yaxis=dict(
                title= 'Feature Importance',
                ticklen= 5,
                gridwidth= 2
            ),
            showlegend= False
        )
        fig = go.Figure(data=data, layout=layout)
        #py.iplot(fig,filename='scatter2010')
        
        # automatically open plot in local window (google chrome for me)
        fig.write_html('random_forest_feature_importance_figure.html', auto_open=True)
        # get plot to appear in plot window
        #fig.show(renderer="png")
        return (fig) #############******************************************Problem

