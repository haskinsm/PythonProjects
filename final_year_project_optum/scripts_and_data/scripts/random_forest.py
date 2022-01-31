# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:56:10 2021

@author: micha

Script containing the class Model which is used to create Random Forest models, and calculate the models 
accuracy when applied to the training and validation set.

Classes:
    Model 
"""
import pandas as pd
import numpy as np
import sys
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Files I've written
import sklearn_helper as sklHelper 

class Model():
    """
    A class for creating Random Forest models. Example of creating instance of this class:
    rf = Model(targetVar, train, xTest, yTest, xValid, yValid)
    
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
        Creates a Random Forest model (fitted to the training set)
    modelAccuracy():
        Gets decimal accuracy of model applied to the test set
    validAccuracy():
        Gets decimal accuracy of model applied to the valid set
    featureImportance():
        Returns a figure showcasing the importance of the datasets features to the model
    
     
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
                   " in the master script. E.g:  rf = Model(targetVar, train, xTest, yTest, xValid, yValid) ")
             print("\nProgram Execution aborted. Goodbye.")
             sys.exit() ## This will terminate the script execution
            
            
    def createModel(self, nTrees = 100):
        """
        Function for creating Random Forest models. The created model will be saved to this instance of the object and 
        nothing will be returned.

        Parameters
        ----------
        nTrees : int, optional
            DESCRIPTION. Optionally set the number of trees in the random forest model. The default is 100.

        Returns
        -------
        None.

        """
        rf_params = {
            'n_jobs': -1,  # use all processing cores
            'n_estimators': nTrees, # 500 trees
            #'warm_start': True, 
            #'max_depth': 8,  # depth of tree,(default = None)
            'min_samples_leaf': 50, # Too small a number will result in overfitting 
            'max_features' : 'sqrt', # Better than setting it as 0.2 due to size of datasets
            'verbose': 0 #suppresses all text while a value of 3 outputs the tree learning process at every iteration
        }
        self.rfHelper = sklHelper.SklearnHelper(clf=RandomForestClassifier, params=rf_params)
        
        ### Create Numpy arrays of train, test and target dataframes to feed into model 
        # create yTrain variable and remove target variable from train 
        self.yTrain = self.train[self.targetVar].ravel() # ravel is used to change a 2-dimensional array or a multi-dimensional array into a contiguous flattened array 
        train = self.train.drop([self.targetVar], axis=1) 
        
        # create arrays of training and test data
        xTrain = train.values   # Creates an array of the train data
        xTest = self.xTest.values   # Creates an array of the test data
        
        #### Need to convert array to float 32 not default float 62 for use in sklearn functions
        self.xTrain = np.nan_to_num(xTrain.astype(np.float32)) # This will convert everything to float 32 and if this results in inf they will be converted to max float 32
        self.xTest = np.nan_to_num(xTest.astype(np.float32)) 
        
        ## Train the model using the training sets
        self.rf = self.rfHelper.fit(self.xTrain, self.yTrain)
        
        
        ### Delete no longer needed vars from memory 
        del train 
        del xTrain
        del xTest
        del rf_params 
     
        
    def modelAccuracy(self):
        """
        Returns decimal accuracy of Random Forest model applied to the Test set. E.g. 0.9210...

        Returns
        -------
        float
            Decimal accuracy of the model applied to the test set.  E.g. 0.9210...

        """
        ## Make predictions using the model
        yTestPred = self.rf.predict(self.xTest) 
        ## Return the accuracy of the model when applied to the test set 
        return (metrics.accuracy_score(self.yTest, yTestPred)) 
    
    
    def validAccuracy(self):
        """
        Return decimal accuracy of Random Forest model when applied to the Validation set

        Returns
        -------
        float
            decimal accuracy of support vector machine model applied to validation set

        """ 
        ## Make predictions using the model 
        yValidPred = self.rf.predict(self.xValid)
        ## Return the accuracy of the model when applied ot the validation set 
        return (metrics.accuracy_score(self.yValid, yValidPred))
        
        
    def featureImportance(self):
        """
            Returns a figure showing the feature importances generated from the different classifiers.
            The importance of a feature is computed as the (normalized) total reduction of the criterion 
            brought by that feature. It is also known as the Gini importance.
        """
        # As per the Sklearn documentation, most of the classifiers are built in with an attribute which 
        # returns feature importances by simply typing in .featureimportances.
       
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
        # fig.write_html('random_forest_feature_importance_figure.html', auto_open=True) # this works
        # get plot to appear in plot window
        #fig.show(renderer="png")
        
        ### Delete no longer needed vars from memory 
        del train 
        del data 
        
        return (fig) 

