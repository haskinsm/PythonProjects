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
import statistics
import matplotlib.pyplot as plt


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
wpData = scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump 

##### Random Forest
# create shorter reference
rf = scripts_and_data.scripts.random_forest # reference of script random_forest
# create instance of random forest class 
wpRfObj = rf.Model(wpData.TARGET_VAR_NAME, wpData.TRAIN, wpData.XTEST, wpData.YTEST, wpData.XVALID, wpData.YVALID)
wpRfObj.createModel() # train the model
wpRfTestAccuracy = wpRfObj.modelAccuracy() # get model test accuracy 
wpRfValidAccuracy = wpRfObj.validAccuracy() # get model valid accuracy 
wpRfFeatureImpPlot = wpRfObj.featureImportance() # get plot of feature importance 
wpRfFeatureImpPlot.show(renderer="png") # render plot of feature importance 

##### XGBoost
# create shorter reference
xgb = scripts_and_data.scripts.xgboost_script # reference of script random_forest
# create instance of random forest class 
wpXgbObj = xgb.Model(wpData.TARGET_VAR_NAME, wpData.TRAIN, wpData.XTEST, wpData.YTEST, wpData.XVALID, wpData.YVALID)
wpXgbObj.createModel() # train the model
wpXgbTestAccuracy = wpXgbObj.modelAccuracy() # get model test accuracy 
wpXgbValidAccuracy =  wpXgbObj.validAccuracy() # get valid accuracy 

########## Census Income dataset #######################
cIData = scripts_and_data.data.census_income_dataset.CensusIncome

##### Random Forest
# create shorter reference
rf = scripts_and_data.scripts.random_forest # reference of script random_forest
# create instance of random forest class 
cIRfObj = rf.Model(cIData.TARGET_VAR_NAME, cIData.TRAIN, cIData.XTEST, cIData.YTEST, cIData.XVALID, cIData.YVALID)
cIRfObj.createModel() # train the model
cIRfTestAccuracy = cIRfObj.modelAccuracy() # get model test accuracy 
cIRfValidAccuracy = cIRfObj.validAccuracy() # get model valid accuracy
cIRfFeatureImpPlot = cIRfObj.featureImportance() # get plot of feature importance 
cIRfFeatureImpPlot.show(renderer="png") # render plot of feature importance


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
        print("Error: You cant have a noise percentage in excess of 100 or below 0. Please try again")
        return 0
    
    # get number of obs target avlues to change
    numObs = len(data) 
    numToChange = round(numObs * noisePerc/100)
    # randomly select the number of rows from the dataframe (without replacement)
    toChangeDf = data.sample(n=numToChange) # add replace = True for replacement sampling 
    
    ### Create a df of rest of the data 
    toChangeIndex = toChangeDf.index # get index of observations that noise will be inserted into 
    # remove observations with these indexes
    restDf = data.copy() # create copy so original is not overwitten 
    restDf.drop(toChangeIndex, inplace=True)
    
    def changeTargetValue(row):
        targetValue = row[-1]
        if(targetValue == 0):
            return 1
        else: 
            return 0
    
    # Now interate through the toChangeDf and add noise to target variable 
    toChangeDf.iloc[:,-1] = toChangeDf.apply(lambda row: changeTargetValue(row), axis = 1)
    noiseDf = pd.DataFrame(toChangeDf)
    
    # Concatenate unchanged df with the noiseDf to get the dataframe to be returned
    df = pd.concat([restDf, noiseDf],  axis = 0) #This will result in 
    
    return df  

def insertingNoiseTestSet(xTest, yTest, noisePerc):
    """ Function to handle inserting noise into test set. Returns noiseXTest and noiseYTest dataframes  """
    noiseDf = insertingNoise( pd.concat([xTest, yTest],  axis = 1), noisePerc)
    noiseXTest = noiseDf.iloc[:,:-1]
    noiseYTest = noiseDf.iloc[:, -1]
    return noiseXTest, noiseYTest

noiseTrain = insertingNoise(wpData.TRAIN, 1)
noiseXTest, noiseYTest = insertingNoiseTestSet(wpData.XTEST, wpData.YTEST, 1)



## Now write functions to add noise over a specified range, and then plot the resulting change in accuracy 

def noiseEffect(mlAlgoScriptRef, dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements):
    """
    This function should take in the dataRef (being a refernece to a data class, e.g. wpData is an abbrevated version of the ref to the WaterPump data class
    where wpData = scripts_and_data.data.water_pump_dataset.WaterPump), should take in the noiseStart and noiseEnd values, and the integer noise increments.
    If an error occurs 0 is returned.
    """
    if( noiseStartPerc > 100 or noiseStartPerc < 0 or noiseEndPerc > 100 or noiseEndPerc < 0):
        print("Error: You cant have a noise percentage in excess of 100 or below 0. Please try again")
        return 0
    elif(noiseStartPerc > noiseEndPerc):
        print("Error: You cant have a noise end percentage less than the noise start percentage. Please try again")
        return 0
    elif(isinstance(numNoiseIncrements, int) == False):
        print("Error: please enter an integer noiseIncrements value")
        return 0
   
    testAccuracy = []
    valAccuracy = []
    noiseLevelPerc = []
    train = dataRef.TRAIN 
    xTest = dataRef.XTEST
    yTest = dataRef.YTEST
    #rf = scripts_and_data.scripts.random_forest # reference to random_forest script
    
    noiseIncrements = round((noiseEndPerc - noiseStartPerc)/numNoiseIncrements)
    for x in range(noiseStartPerc, noiseEndPerc, noiseIncrements):
        
        # Get average accuracy at this perc noise interval
        testAccuaracyAtIncrement = []
        valAccuaracyAtIncrement = []
        for i in range(10):
            train = insertingNoise(dataRef.TRAIN, x)
            xTest, yTest = insertingNoiseTestSet(dataRef.XTEST, dataRef.YTEST, x)
            obj = mlAlgoScriptRef.Model(dataRef.TARGET_VAR_NAME, train, xTest, yTest, dataRef.XVALID, dataRef.YVALID)
            obj.createModel() # train the model
            # Get and append model test and validation accuracy
            rfTestAccuracy = obj.modelAccuracy() 
            testAccuaracyAtIncrement.append(rfTestAccuracy)
            rfValAccuaracy = obj.validAccuracy()
            valAccuaracyAtIncrement.append(rfValAccuaracy)
            
        testAccuracy.append(statistics.mean(testAccuaracyAtIncrement))
        valAccuracy.append(statistics.mean(valAccuaracyAtIncrement))
        noiseLevelPerc.append(x) # Append noise level perc
    
    return testAccuracy, valAccuracy, noiseLevelPerc
        
def rfNoiseEffect(dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements):
    rfScriptRef = scripts_and_data.scripts.random_forest
    return noiseEffect(rfScriptRef, dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements)

def xgbNoiseEffect(dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements):
    xgbScriptRef = scripts_and_data.scripts.xgboost_script
    return noiseEffect(xgbScriptRef, dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements)

def createSingleNoiseEffectPlot(testAccuracy, valAccuracy, noiseLevelPerc, datasetName, algorithmName):
    plt.figure() # Instantiate a new figure 
    plt.plot(noiseLevelPerc, testAccuracy,'r--', label = "Test")
    plt.plot(noiseLevelPerc, valAccuracy, 'g--', label = "Validation")
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Noise Effect on {} Accuracy for {}".format(algorithmName, datasetName), fontsize=18)
    plt.title("Note: Noise randomly inserted to target variable in training and test sets", fontsize=12)
    #plt.close() # Close current fig so nothing further will be overlayed on it 
    
def createMlAlgorithmNoiseEffectPlot(wpTestAccuracy, wpValAccuracy, ciTestAccuracy, ciValAccuracy, noiseLevelPerc, mlAlgorithmName):
    """
    Function to generate a plot depicting how the accuracy of machine learning alrgorithm Random forests is
    affected by adding noise to the target variable in the test and training sets for multiple datasets.  

    Parameters
    ----------
    wpTestAccuracy : []
        DESCRIPTION.
    wpValAccuracy : []
        DESCRIPTION.
    ciTestAccuracy : []
        DESCRIPTION.
    ciValAccuracy : []
        DESCRIPTION.
    noiseLevelPerc : []
        DESCRIPTION. Assumes that the noise increments is the same for all datasets 
    mlAlgorithmName : String
        DESCRIPTION. E.g. "Random Forest" or "XGBoost"

    Returns
    -------
    None.

    """
    plt.figure()
    plt.plot(noiseLevelPerc, wpTestAccuracy,'r--', label = "Test WaterPump Dataset")
    plt.plot(noiseLevelPerc, wpValAccuracy, 'g--', label = "Validation WaterPump Dataset")
    plt.plot(noiseLevelPerc, ciTestAccuracy,'r:', label = "Test Census Income Dataset")
    plt.plot(noiseLevelPerc, ciValAccuracy, 'g:', label = "Validation WaterPump Dataset")
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Noise Effect on {} Accuracy".format(mlAlgorithmName), fontsize=18)
    plt.title("Note: Noise randomly inserted to target variable in training and test sets", fontsize=12)
    
def createMultipleNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy, wpXgbTestAccuracy, wpXgbValAccuracy, ciRfTestAccuracy, ciRfValAccuracy, ciXgbTestAccuracy, ciXgbValAccuracy, noiseLevelPerc):
    plt.figure() # Instantiate a new figure 
    # rf
    plt.plot(noiseLevelPerc, wpRfTestAccuracy,'r--', label = "Rf Test WaterPump D.")
    plt.plot(noiseLevelPerc, wpRfValAccuracy, 'g--', label = "Rf Validation WaterPump D.")
    plt.plot(noiseLevelPerc, ciRfTestAccuracy,'r:', label = "Rf Test Census Income D.")
    plt.plot(noiseLevelPerc, ciRfValAccuracy, 'g:', label = "Rf Validation Census Income D.")
    # xgb
    plt.plot(noiseLevelPerc, wpXgbTestAccuracy,'b--', label = "Xgb Test WaterPump D.")
    plt.plot(noiseLevelPerc, wpXgbValAccuracy, 'k--', label = "Xgb Validation WaterPump D.")
    plt.plot(noiseLevelPerc, ciXgbTestAccuracy,'b:', label = "Xgb Test Census Income D.")
    plt.plot(noiseLevelPerc, ciXgbValAccuracy, 'k:', label = "Xgb Validation Census Income D.")
    
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Noise Effect on Machine Learning Algorithms Accuracy", fontsize=18)
    plt.title("Note: Noise randomly inserted to target variable in training and test sets", fontsize=12)
  
    
########### Get accuracy of specifc datasets and algorithms for specific noise levels ########
####### WaterPump dataset 
## rf     
wpRfTestAccuracy, wpRfValAccuracy, wpRfNoiseLevelPerc = rfNoiseEffect(wpData, 0, 100, 100)
## xgb
wpXgbTestAccuracy, wpXgbValAccuracy, wpXgbNoiseLevelPerc = xgbNoiseEffect(wpData, 0, 100, 100)

####### Census Income dataset 
## rf     
ciRfTestAccuracy, ciRfValAccuracy, ciRfNoiseLevelPerc = rfNoiseEffect(cIData, 0, 100, 100)
## xgb
ciXgbTestAccuracy, ciXgbValAccuracy, ciXgbNoiseLevelPerc = xgbNoiseEffect(cIData, 0, 100, 100)



############################ Plots ##############################
####### Generate Simple plots for each algorithm applied to each dataset #####
#### Waterpump dataset 
## rf
createSingleNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy, wpRfNoiseLevelPerc, "Water Pump dataset", "Random Forest")
## xgb
createSingleNoiseEffectPlot(wpXgbTestAccuracy, wpXgbValAccuracy, wpXgbNoiseLevelPerc, "Water Pump dataset", "XGBoost")

#### Census Income 
## rf
createSingleNoiseEffectPlot(ciRfTestAccuracy, ciRfValAccuracy, ciRfNoiseLevelPerc, "Census Income dataset", "Random Forest")
## xgb
createSingleNoiseEffectPlot(wpXgbTestAccuracy, wpXgbValAccuracy, wpXgbNoiseLevelPerc, "Water Pump dataset", "XGBoost")

    
#######  Noise Effect for specific ml algorithm on multiple datasets ######
#### rf
createMlAlgorithmNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy, ciRfTestAccuracy, ciRfValAccuracy, wpRfNoiseLevelPerc, "Random Forest")
#### xgb 
createMlAlgorithmNoiseEffectPlot(wpXgbTestAccuracy, wpXgbValAccuracy, ciXgbTestAccuracy, ciXgbValAccuracy, wpXgbNoiseLevelPerc, "XGBoost")


###### Noise Effect for multiple datasets #######
createMultipleNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy, wpXgbTestAccuracy, wpXgbValAccuracy, ciRfTestAccuracy, ciRfValAccuracy, ciXgbTestAccuracy, ciXgbValAccuracy, wpRfNoiseLevelPerc)


    

