# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 19:04:08 2022

@author: micha
"""

import os
import pandas as pd 
import matplotlib.pyplot as plt
from yellowbrick.regressor import cooks_distance
import statistics

# Change directory to correctDir or script won't run correctly
CORRECTDIR = "C:\\Users\\micha\\Documents\\3rd year\\Software Applications\\PythonSpyder(Anaconda V)\\final_year_project_optum"
os.chdir(CORRECTDIR)
##### Import scripts I've written. Do this after changed to correct directory ###########################
import scripts_and_data 
import master


################################# Basic system testing immediatly below #####################################

######### WaterPump (wp) Dataset ##########
wpData = scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump 

##### Random Forest
# create shorter reference
rf = scripts_and_data.scripts.random_forest # reference of script random_forest
# create instance of random forest class 
wpRfObj = rf.Model(wpData.TARGET_VAR_NAME, wpData.TRAIN, wpData.XTEST, wpData.YTEST, wpData.XVALID, wpData.YVALID)
wpRfObj.createModel() # train the model
wpRfTest = wpRfObj.modelAccuracy() # get model test accuracy 
wpRfVal = wpRfObj.validAccuracy() # get model valid accuracy 
wpRfFeatureImpPlot = wpRfObj.featureImportance() # get plot of feature importance 
wpRfFeatureImpPlot.show(renderer="png") # render plot of feature importance 

##### XGBoost
# create shorter reference
xgb = scripts_and_data.scripts.xgboost_script 
# create instance of xgboost class 
wpXgbObj = xgb.Model(wpData.TARGET_VAR_NAME, wpData.TRAIN, wpData.XTEST, wpData.YTEST, wpData.XVALID, wpData.YVALID)
wpXgbObj.createModel() # train the model
wpXgbTest = wpXgbObj.modelAccuracy() # get model test accuracy 
wpXgbVal =  wpXgbObj.validAccuracy() # get valid accuracy 


##### Decision Tree
# create shorter reference 
dt = scripts_and_data.scripts.decision_tree
# create instance of decsion tree class
wpDtObj = dt.Model(wpData.TARGET_VAR_NAME, wpData.TRAIN, wpData.XTEST, wpData.YTEST, wpData.XVALID, wpData.YVALID)
wpDtObj.createModel() # train the model
wpDtTest = wpDtObj.modelAccuracy() # get model test accuracy 
wpDtVal =  wpDtObj.validAccuracy() # get valid accuracy 


########## Census Income dataset #######################
cIData = scripts_and_data.data.census_income_dataset.CensusIncome

##### Random Forest
# create shorter reference
rf = scripts_and_data.scripts.random_forest # reference of script random_forest
# create instance of random forest class 
cIRfObj = rf.Model(cIData.TARGET_VAR_NAME, cIData.TRAIN, cIData.XTEST, cIData.YTEST, cIData.XVALID, cIData.YVALID)
cIRfObj.createModel() # train the model
cIRfTest = cIRfObj.modelAccuracy() # get model test accuracy 
cIRfVal = cIRfObj.validAccuracy() # get model valid accuracy
cIRfFeatureImpPlot = cIRfObj.featureImportance() # get plot of feature importance 
#cIRfFeatureImpPlot.show(renderer="png") # render plot of feature importance

########## Credit Card Default dataset ###################
cCDData = scripts_and_data.data.credit_card_default_dataset.CreditCardDefault

##### Random Forest 
# create shorter reference
rf = scripts_and_data.scripts.random_forest # reference of script random_forest
# create instance of random forest class 
cCDRfObj = rf.Model(cCDData.TARGET_VAR_NAME, cCDData.TRAIN, cCDData.XTEST, cCDData.YTEST, cCDData.XVALID, cCDData.YVALID)
cCDRfObj.createModel() # train the model
cCDRfTest = cCDRfObj.modelAccuracy() # get model test accuracy 
cCDRfVal = cCDRfObj.validAccuracy() # get model valid accuracy
cCDRfFeatureImpPlot = cCDRfObj.featureImportance() # get plot of feature importance 
#cCDRfFeatureImpPlot.show(renderer="png") # render plot of feature importance

##### Support vector machine
# create shorter reference 
svm = scripts_and_data.scripts.support_vector_machine
# create instance of class 
cCDSvmObj = svm.Model(cCDData.TARGET_VAR_NAME, cCDData.TRAIN, cCDData.XTEST, cCDData.YTEST, cCDData.XVALID, cCDData.YVALID)
cCDSvmObj.createModel()
cCDSvmTest = cCDSvmObj.modelAccuracy()
cCDSvmVal = cCDSvmObj.validAccuracy()

################################# End of basic system testing ##############################
