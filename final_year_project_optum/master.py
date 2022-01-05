# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:02:43 2021

@author: micha
"""
import os
import numpy as np 
#from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.stats.outliers_influence import OLSInfluence
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import cooks_distance
# Change directory to correctDir or script wont run correctly
CORRECTDIR = "C:\\Users\\micha\\Documents\\3rd year\\Software Applications\\PythonSpyder(Anaconda V)\\final_year_project_optum"
os.chdir(CORRECTDIR)

##### Import scripts I've written. Do this after changed to correct directory ###########################
import scripts_and_data


####### Titanic dataset ############
""" ********************************************* This ownt work as I've made changes to random_forest constructor
##### Random Forest
# create shorter refernce
rf = scripts_and_data.scripts.random_forest
data = scripts_and_data.data.titanic.Titanic
# create instance of random forest class 
rfObj = rf.RandomForest(data.TARGET_VAR_NAME, data.TRAIN, data.TEST, data.YTEST)
rfObj.createModel() # train the model
rfAccuracy = rfObj.modelAccuracy() # get model accuracy 
rfFeatureImpPlot = rfObj.featureImportance() # get plot of feature importance 
rfFeatureImpPlot.show(renderer="png") # render plot of feature importance 
"""

######### WaterPump (wp) Dataset ##########
##### Random Forest
# create shorter reference
wpRf = scripts_and_data.scripts.random_forest # reference of script WaterPump
wpData = scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump 
# create instance of random forest class 
wpRfObj = wpRf.RandomForest(wpData.TARGET_VAR_NAME, wpData.TRAIN, wpData.XTEST, wpData.YTEST, wpData.XVALID, wpData.YVALID)
wpRfObj.createModel() # train the model
wpRfAccuracy = wpRfObj.modelAccuracy() # get model accuracy 
wpRfFeatureImpPlot = wpRfObj.featureImportance() # get plot of feature importance 
wpRfFeatureImpPlot.show(renderer="png") # render plot of feature importance 

   
X, y = load_concrete()  #Test
    
def getCooksDistance(data):
    """
    Function to get cooks distance for this dataset 
    src: https://www.scikit-yb.org/en/latest/api/regressor/influence.html
    """
    targetVar = data.columns[-1]
    targetData = data[targetVar]
    predictors = list(data.columns[:-1])
    predictorData = data[predictors]
    
    #predictorData = np.nan_to_num(predictorData.astype(np.float64)) # This will convert everything to float 32 and if this results in inf they will be converted to max float 64
    
    # Instantiate and fit the visualizer
    cooksD = cooks_distance(
        predictorData, targetData,
        draw_threshold=True,
        linefmt="C0-", markerfmt=","
    )
    return cooksD

cooksDistPlot = getCooksDistance(wpData.TRAIN)



