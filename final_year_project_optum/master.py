# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:02:43 2021

@author: micha
"""
import os
import numpy as np 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import cooks_distance
# Change directory to correctDir or script wont run correctly
CORRECTDIR = "C:\\Users\\micha\\Documents\\3rd year\\Software Applications\\PythonSpyder(Anaconda V)\\final_year_project_optum"
os.chdir(CORRECTDIR)

##### Import scripts I've written. Do this after changed to correct directory
import scripts_and_data


# Create shorter references for these classes
#rf = scripts_and_data.scripts.random_forest
data = scripts_and_data.data.titanic.Titanic
"""
rfObj = rf.RandomForest(data.TARGET_VAR_NAME, data.TRAIN, data.TEST, data.YTEST)
rfObj.createModel()
rfAccuracy = rfObj.modelAccuracy()
rfFeatureImpPlot = rfObj.featureImportance()
rfFeatureImpPlot.show(renderer="png")
"""
a=data.TRAIN
b=data.TEST
c=data.YTEST
d=data.FULLDATASET
##
def getCooksDistance(predictorData, targetData):
    """
    Function to get cooks distance for this dataset 
    src: https://www.scikit-yb.org/en/latest/api/regressor/influence.html
    """
    predictorData = np.nan_to_num(predictorData.astype(np.float64)) # This will convert everything to float 32 and if this results in inf they will be converted to max float 64
  
    # The below indented code creates a residuals plot, but this is fairly useless I think
        #lm = LinearRegression()
        ####lm.fit(predictorData, targetData)
        #visualizerResiduals = ResidualsPlot(lm)
        #visualizerResiduals.fit(predictorData, targetData)
        #visualizerResiduals.show()
    
    # Instantiate and fit the visualizer
    cooks_distance(
        predictorData, targetData,
        draw_threshold=True,
        linefmt="C0-", markerfmt=","
    )

predictors = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Has_Cabin']
f = getCooksDistance(data.FULLDATASET[predictors], data.FULLDATASET["Survived"])
