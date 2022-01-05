# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:02:43 2021

@author: micha
"""
import os
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import CooksDistance
import matplotlib.pyplot as plt
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


    
def getCooksDistance(data):
    """
    Function to get cooks distance for this dataset 
    src: https://www.scikit-yb.org/en/latest/api/regressor/influence.html
    imilar src: https://coderzcolumn.com/tutorials/machine-learning/yellowbrick-visualize-sklearn-classification-and-regression-metrics-in-python#regression_4
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
        linefmt="C0-", markerfmt=",",
        fig=plt.figure(figsize=(12,7))
    )
    
    ##### Seperate attempt #####
    #viz = CooksDistance(fig=plt.figure(figsize=(9,7)))
    #viz.fit(predictorData, targetData)
    
    return cooksD

cooksDist = getCooksDistance(wpData.TRAIN)
#cooksDist.show()


def insertingNoise(data, noisePerc):
    """
    Function to insert noise into a passed dataset. The percentage of noise from 0 -> 100 should be entered 
    If an invalid percentage is entred 0 will be returned 

    """
    if( noisePerc > 100 or noisePerc < 0):
        print("Error: You cant have a noise percentage in excess of 100. Please try again")
        return 
    
    # get number of obs target avlues to change
    numObs = len(data) 
    numToChange = round(numObs * noisePerc/100)
    # randomly select the number of rows from the dataframe (without replacement)
    toChangeDf = data.sample(n=numToChange) # add replace = True for replacement sampling 
    # Create a df of rest of the data 
    restDf = toChangeDf.merge(data, how = "right", indicator = True) # This will create a copy of data, with the added 
    # column _merge, which will have values right_only for values not in the toChangeDf. These are the values I want
    # as these represent values that are not in the subset toChangeDf
    restDf = restDf.loc[restDf['_merge'] == 'right_only', ].iloc[:,:-1]
    
    
    def changeTargetValue(row):
        targetValue = row[-1]
        if(targetValue == 0):
            return 1
        else: 
            return 0
    
    # Now interate through the toChangeDf and add noise to target variable 
    toChangeDf.iloc[:,-1] = toChangeDf.apply(lambda row: changeTargetValue(row), axis = 1)
    noiseDf = pd.DataFrame(toChangeDf)
    
    # Create df of rest of the data and then merge this with the noiseDf
    df = noiseDf.merge(restDf, how = "inner") #This will result in 
    
    return df  

newData = insertingNoise(wpData.TRAIN, 1)
#c = wpData.TRAIN 


#### Test
    
data_df1 = np.array([['Name','Unit','Attribute','Date'],['a','A',1,2014],['b','B',2,2015],['c','C',3,2016],\
                 ['d','D',4,2017],['e','E',5,2018]])
data_df2 = np.array([['Name','Unit','Attribute','Date'],['a','A',1,2014], ['e','E',5, 2018]])
df1 = pd.DataFrame(data=data_df1)
df2 = pd.DataFrame(data=data_df2)
df3 = df2.merge(df1, how = "right", indicator = True)    
a = df3.loc[df3['_merge'] == 'right_only', ].iloc[:,:-1]

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

