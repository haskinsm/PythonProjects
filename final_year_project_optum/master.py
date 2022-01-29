# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:02:43 2021

@author: micha
"""
import os
#import numpy as np 
import pandas as pd 
#from sklearn.linear_model import LinearRegression
#from yellowbrick.regressor import CooksDistance
import matplotlib.pyplot as plt
from yellowbrick.regressor import cooks_distance
import statistics


# Change directory to correctDir or script wont run correctly
CORRECTDIR = "C:\\Users\\micha\\Documents\\3rd year\\Software Applications\\PythonSpyder(Anaconda V)\\final_year_project_optum"
os.chdir(CORRECTDIR)
##### Import scripts I've written. Do this after changed to correct directory ###########################
import scripts_and_data



################################# Basic system testing immediatly below #####################################
####### Titanic dataset ############
""" 
********************************************* This wownt work anymore as I've made changes to random_forest constructor
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
xgb = scripts_and_data.scripts.xgboost_script 
# create instance of xgboost class 
wpXgbObj = xgb.Model(wpData.TARGET_VAR_NAME, wpData.TRAIN, wpData.XTEST, wpData.YTEST, wpData.XVALID, wpData.YVALID)
wpXgbObj.createModel() # train the model
wpXgbTestAccuracy = wpXgbObj.modelAccuracy() # get model test accuracy 
wpXgbValidAccuracy =  wpXgbObj.validAccuracy() # get valid accuracy 


##### Decision Tree
# create shorter reference 
dt = scripts_and_data.scripts.decision_tree
# create instance of decsion tree class
wpDtObj = dt.Model(wpData.TARGET_VAR_NAME, wpData.TRAIN, wpData.XTEST, wpData.YTEST, wpData.XVALID, wpData.YVALID)
wpDtObj.createModel() # train the model
wpDtTestAccuracy = wpDtObj.modelAccuracy() # get model test accuracy 
wpDtValidAccuracy =  wpDtObj.validAccuracy() # get valid accuracy 


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
#cIRfFeatureImpPlot.show(renderer="png") # render plot of feature importance

########## Credit Card Default dataset ###################
cCDData = scripts_and_data.data.credit_card_default_dataset.CreditCardDefault

##### Random Forest 
# create shorter reference
rf = scripts_and_data.scripts.random_forest # reference of script random_forest
# create instance of random forest class 
cCDRfObj = rf.Model(cCDData.TARGET_VAR_NAME, cCDData.TRAIN, cCDData.XTEST, cCDData.YTEST, cCDData.XVALID, cCDData.YVALID)
cCDRfObj.createModel() # train the model
cCDRfTestAccuracy = cCDRfObj.modelAccuracy() # get model test accuracy 
cCDRfValidAccuracy = cCDRfObj.validAccuracy() # get model valid accuracy
cCDRfFeatureImpPlot = cCDRfObj.featureImportance() # get plot of feature importance 
#cCDRfFeatureImpPlot.show(renderer="png") # render plot of feature importance

################################# End of basic system testing ##############################

def getCooksDistance(data):
    """
    Function to get cooks distance for a pandas dataframe
    
    This function takes in a pandas dataframe and returns an object of yellowbrick.regressor.influence.
    A plot n then be called from this returned object using the method .show()

    
    src: https://www.scikit-yb.org/en/latest/api/regressor/influence.html
    Similar src: https://coderzcolumn.com/tutorials/machine-learning/yellowbrick-visualize-sklearn-classification-and-regression-metrics-in-python#regression_4
    Parameters
    ----------
    data : pandas dataframe
        DESCRIPTION. This should be a pandas dataframe 

    Returns
    -------
    cooksD : Object of yellowbrick.regressor.influence
        DESCRIPTION. cooksDist = getCooksDistance(wpData.TRAIN)   
                     cooksDist.show() # show the plot of cooks distance    
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
cooksDist.show()


def insertingNoise(data, noisePerc):
    """
    Function to insert a specifc percentage of noise into a pandas dataframe. The target variable is assumed
    to be the last column in the dataframe. A pandas dataframe with the noise inserted will be returned 
    
    If an invalid percentage is entred 0 will be returned  
    
    Parameters
    ----------
    data : pandas dataframe
        DESCRIPTION. Dataframe to inserrt noise to. It is assumed the last column is the binary target var to insert 
        noise to. 
    noisePerc : float
        DESCRIPTION. This is the percentage of noise to insert into the target variable. Enter 40.5 for 40.5% noise.
        Should be a float ranging from 0-100. If it is not 0 will be returned

    Returns
    -------
    pandas dataframe
        DESCRIPTION.

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
        """
        To be used in a Lambda function embedded in an apply function. Takes a row of observations and 
        returns the correct value so that the binary target variable will be swapped to its opposite value
        in the apply function

        Parameters
        ----------
        row : TYPE
            DESCRIPTION.

        Returns
        -------
        int
            DESCRIPTION.

        """
        targetValue = row[-1]
        if(targetValue == 0):
            return 1
        # else
        return 0 
    
    # Now interate through the toChangeDf and add noise to target variable 
    toChangeDf.iloc[:,-1] = toChangeDf.apply(lambda row: changeTargetValue(row), axis = 1)
    noiseDf = pd.DataFrame(toChangeDf)
    
    # Concatenate unchanged df with the noiseDf to get the dataframe to be returned
    df = pd.concat([restDf, noiseDf],  axis = 0) #This will result in 
    
    return df  

def insertingNoiseTestSet(xTest, yTest, noisePerc):
    """
    Function to handle inserting noise into the test set. Calls the function inserting noise which returns a dataframe
    with the noise inserted. This function splits up this dataframe into the x (i.e. explanatory variables) and
    the y (i.e. target variable). 

    Parameters
    ----------
    xTest : pandas dataframe
        DESCRIPTION. The noise free X variables of the test set
    yTest : pandas dataframe
        DESCRIPTION. The noise free Y variables of the test set
    noisePerc : float 
        DESCRIPTION. This is the percentage of noise to insert into the target variable. Enter 40.5 for 40.5% noise.
        Should be a float ranging from 0-100. If it is not 0 will be returned

    Returns
    -------
    noiseXTest : pandas dataframe
        DESCRIPTION.
    noiseYTest : pandas dataframe
        DESCRIPTION.

    """
    noiseDf = insertingNoise( pd.concat([xTest, yTest],  axis = 1), noisePerc)
    noiseXTest = noiseDf.iloc[:,:-1]
    noiseYTest = noiseDf.iloc[:, -1]
    return noiseXTest, noiseYTest

noiseTrain = insertingNoise(wpData.TRAIN, 1)
noiseXTest, noiseYTest = insertingNoiseTestSet(wpData.XTEST, wpData.YTEST, 1)



## Now write functions to add noise over a specified range, and then plot the resulting change in accuracy 

def noiseEffect(mlAlgoScriptRef, dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements):
    """
    This function adds noise at increments over a specified range to a dataset and calculates the acccuracy of a Machine Learning algorithm when 
    applied to the dataset. 
    
    If an error occurs 0 is returned.

    Parameters
    ----------
    mlAlgoScriptRef : Class Reference
        DESCRIPTION.
    dataRef : Class Reference 
        DESCRIPTION.  Reference to the class of a dataset E.g.: scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump
    noiseStartPerc : float
        DESCRIPTION. % to start incrementing noise from 
    noiseEndPerc : float
        DESCRIPTION. % to stop incrementing noise at 
    numNoiseIncrements : int
        DESCRIPTION. Number of noise increments over the range of noiseStartPerc and noieEndPerc

    Returns
    -------
    [] 
        DESCRIPTION. List of test Accuaracys of random forest model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuaracys of random forest model when noise is applied 
    [] 
        DESCRIPTION. List of % noise increments that have been applied 

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
    """
    Helper Function for general noise Effect function. This will call the noiseEffect function 
    will add noise to data used to train random forest models

    Parameters
    ----------
    dataRef : Class Reference
        DESCRIPTION. Refernece to the class of a dataset E.g.: scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump
    noiseStartPerc : Float 
        DESCRIPTION. % to start incrementing noise from 
    noiseEndPerc : Float 
        DESCRIPTION. % to stop incrementing noise at 
    numNoiseIncrements : Int
        DESCRIPTION. Number of noise increments over the range of noiseStartPerc and noieEndPerc

    Returns
    -------
    [] 
        DESCRIPTION. List of test Accuaracys of random forest model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuaracys of random forest model when noise is applied 
    [] 
        DESCRIPTION. List of % noise increments that have been applied 

    """
    rfScriptRef = scripts_and_data.scripts.random_forest
    return noiseEffect(rfScriptRef, dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements)

def xgbNoiseEffect(dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements):
    """
    Helper Function for general noise Effect function. This will call the noiseEffect function 
    will add noise to data used to train xgboost models

    Parameters
    ----------
    dataRef : Class Reference
        DESCRIPTION. Refernece to the class of a dataset E.g.: scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump
    noiseStartPerc : Float 
        DESCRIPTION. % to start incrementing noise from 
    noiseEndPerc : Float 
        DESCRIPTION. % to stop incrementing noise at 
    numNoiseIncrements : Int
        DESCRIPTION. Number of noise increments over the range of noiseStartPerc and noieEndPerc

    Returns
    -------
    [] 
        DESCRIPTION. List of test Accuaracys of xgboost model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuaracys of xgboost model when noise is applied 
    [] 
        DESCRIPTION. List of % noise increments that have been applied 

    """
    xgbScriptRef = scripts_and_data.scripts.xgboost_script
    return noiseEffect(xgbScriptRef, dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements)

def dtNoiseEffect(dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements):
    """
    Helper Function for general noise Effect function. This will call the noiseEffect function 
    will add noise to data used to train decision tree models

    Parameters
    ----------
    dataRef : Class Reference
        DESCRIPTION. Refernece to the class of a dataset E.g.: scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump
    noiseStartPerc : Float 
        DESCRIPTION. % to start incrementing noise from 
    noiseEndPerc : Float 
        DESCRIPTION. % to stop incrementing noise at 
    numNoiseIncrements : Int
        DESCRIPTION. Number of noise increments over the range of noiseStartPerc and noieEndPerc

    Returns
    -------
    [] 
        DESCRIPTION. List of test Accuaracys of decision tree model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuaracys of decision tree model when noise is applied 
    [] 
        DESCRIPTION. List of % noise increments that have been applied 

    """
    dtScriptRef = scripts_and_data.scripts.decision_tree
    return noiseEffect(dtScriptRef, dataRef, noiseStartPerc, noiseEndPerc, numNoiseIncrements)

def createSingleNoiseEffectPlot(testAccuracy, valAccuracy, noiseLevelPerc, datasetName, algorithmName):
    """
    This is a simple plotting fucntion which plots the test and validation accuracy of a specific ML algorithm
    appied to a specific dataset
    
    Note: This function does not keep dataset clours consitent, and any graphs generated from this will not be used
    in my final report.

    Parameters
    ----------
    testAccuracy : [float]
        DESCRIPTION.
    valAccuracy : [float]
        DESCRIPTION.
    noiseLevelPerc : [int]
        DESCRIPTION.
    datasetName : String
        DESCRIPTION.
    algorithmName : String
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plt.figure() # Instantiate a new figure 
    plt.plot(noiseLevelPerc, testAccuracy,'r--', label = "Test")
    plt.plot(noiseLevelPerc, valAccuracy, 'g--', label = "Validation")
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Noise Effect on {} Accuracy for {}".format(algorithmName, datasetName), fontsize=18)
    plt.title("Note: Noise randomly inserted to binary target variable in training and test sets", fontsize=12)
    #plt.close() # Close current fig so nothing further will be overlayed on it 
    
def createMlAlgorithmNoiseEffectPlot(wpTestAccuracy, wpValAccuracy, 
                                     cITestAccuracy, cIValAccuracy,
                                     cCDTestAccuracy, cCDValAccuracy, 
                                     noiseLevelPerc, mlAlgorithmName):
    """
    Function to generate a plot depicting how the accuracy of machine learning alrgorithm Random forests is
    affected by adding noise to the target variable in the test and training sets for multiple datasets.  

    Parameters
    ----------
    wpTestAccuracy : []
        DESCRIPTION. List of Test accuracys for waterpump dataset
    wpValAccuracy : []
        DESCRIPTION. List of Validation accuracys for waterpump dataset
    ciTestAccuracy : []
        DESCRIPTION. List of Test accuracys for Census Income dataset
    ciValAccuracy : []
        DESCRIPTION. List of Validation accuracys for Census Income dataset
    noiseLevelPerc : []
        DESCRIPTION. **Assumes that the noise increments is the same for all datasets**. List of noise levels which
                     should correspond to the test and validation accuracys 
    mlAlgorithmName : String
        DESCRIPTION. E.g. "Random Forest" or "XGBoost" or "Decsion Tree"

    Returns
    -------
    None.

    """
    plt.figure()
    plt.plot(noiseLevelPerc, wpTestAccuracy, color = 'red', ls = '--', label = "Test WaterPump Dataset")
    plt.plot(noiseLevelPerc, wpValAccuracy, color = 'red', ls = ':', label = "Validation WaterPump Dataset")
    plt.plot(noiseLevelPerc, cITestAccuracy, color = 'green', ls = '--', label = "Test Census Income Dataset")
    plt.plot(noiseLevelPerc, cIValAccuracy, color = 'green', ls = ':', label = "Validation Census Income Dataset")
    plt.plot(noiseLevelPerc, cCDTestAccuracy, color = 'royalblue', ls ='--', label = "Test Credit Card Default Dataset")
    plt.plot(noiseLevelPerc, cCDValAccuracy, color = 'royalblue', ls = ':', label = "Validation Credit Card Default Dataset")
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Noise Effect on {} Accuracy".format(mlAlgorithmName), fontsize=18)
    plt.title("Note: Noise randomly inserted to binary target variable in training and test sets", fontsize=12)
    
def createMultipleNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy, wpXgbTestAccuracy, wpXgbValAccuracy, wpDtTestAccuracy, wpDtValAccuracy,
                                  cIRfTestAccuracy, cIRfValAccuracy, cIXgbTestAccuracy, cIXgbValAccuracy, cIDtTestAccuracy, cIDtValAccuracy, 
                                  cCDRfTestAccuracy, cCDRfValAccuracy, cCDXgbTestAccuracy, cCDXgbValAccuracy, cCDDtTestAccuracy, cCDDtValAccuracy,
                                  noiseLevelPerc):
    """
    Generate a matplotlib plot showcasing the noise effect on the average accuracy of Random Forest, 
    XGBoost and decision tree models applied to 3 datasets. 

    Parameters
    ----------
    wpRfTestAccuracy : TYPE
        DESCRIPTION.
    wpRfValAccuracy : TYPE
        DESCRIPTION.
    wpXgbTestAccuracy : TYPE
        DESCRIPTION.
    wpXgbValAccuracy : TYPE
        DESCRIPTION.
    wpDtTestAccuracy : TYPE
        DESCRIPTION.
    wpDtValAccuracy : TYPE
        DESCRIPTION.
    cIRfTestAccuracy : TYPE
        DESCRIPTION.
    cIRfValAccuracy : TYPE
        DESCRIPTION.
    cIXgbTestAccuracy : TYPE
        DESCRIPTION.
    cIXgbValAccuracy : TYPE
        DESCRIPTION.
    cIDtTestAccuracy : TYPE
        DESCRIPTION.
    cIDtValAccuracy : TYPE
        DESCRIPTION.
    cCDRfTestAccuracy : TYPE
        DESCRIPTION.
    cCDRfValAccuracy : TYPE
        DESCRIPTION.
    cCDXgbTestAccuracy : TYPE
        DESCRIPTION.
    cCDXgbValAccuracy : TYPE
        DESCRIPTION.
    cCDDtTestAccuracy : TYPE
        DESCRIPTION.
    cCDDtValAccuracy : TYPE
        DESCRIPTION.
    noiseLevelPerc : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plt.figure() # Instantiate a new figure 
    
    # Get mean accuracy for each ml algorithm
    meanRfTestAccuracy, meanXgbTestAccuracy, meanDtTestAccuracy = [], [], []
    meanRfValAccuracy, meanXgbValAccuracy, meanDtValAccuracy = [], [], []
    
    for i in range(len(wpRfTestAccuracy)):
        meanRfTestAccuracy.append( wpRfTestAccuracy[i] + cIRfTestAccuracy[i] + cCDRfTestAccuracy[i] /3)
        meanXgbTestAccuracy.append( wpXgbTestAccuracy[i] + cIXgbTestAccuracy[i] + cCDXgbTestAccuracy[i] /3)
        meanDtTestAccuracy.append( wpDtTestAccuracy[i] + cIDtTestAccuracy[i] + cCDDtTestAccuracy[i] /3)
        meanRfValAccuracy.append( wpRfValAccuracy[i] + cIRfValAccuracy[i] + cCDRfValAccuracy[i] /3)
        meanXgbValAccuracy.append( wpXgbValAccuracy[i] + cIXgbValAccuracy[i] + cCDXgbValAccuracy[i] /3)
        meanDtValAccuracy.append( wpDtValAccuracy[i] + cIDtValAccuracy[i] + cCDDtValAccuracy[i] /3)
    
    
    # rf
    plt.plot(noiseLevelPerc, meanRfTestAccuracy, color = 'brown', linestyle = 'dotted', label = "Random Forest Test")
    plt.plot(noiseLevelPerc, meanRfValAccuracy, color = 'brown', ls='-', label = "Random Forest Validation")
    # xgb
    plt.plot(noiseLevelPerc, meanXgbTestAccuracy, color = 'cyan', linestyle = 'dotted', label = "XGBoost Test")
    plt.plot(noiseLevelPerc, meanXgbValAccuracy, color = 'cyan', ls='-', label = "XGBoost Validation")
    ## dt 
    plt.plot(noiseLevelPerc, meanDtTestAccuracy, color = 'gold', linestyle = 'dotted', label = "Decsion Tree Test")
    plt.plot(noiseLevelPerc, meanDtValAccuracy, color = 'gold', ls='-', label = "Decsion Tree Validation")
    
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Average Noise Effect on Machine Learning Algorithms Accuracy", fontsize=18)
    plt.title("Note: Noise randomly inserted to binary target variable in training and test sets", fontsize=12)
  
    
  
    
########### Get accuracy of specifc datasets and algorithms for specific noise levels ########
####### WaterPump dataset 
## rf     
wpRfTestAccuracy, wpRfValAccuracy, wpRfNoiseLevelPerc = rfNoiseEffect(wpData, 0, 50, 51)
## xgb
wpXgbTestAccuracy, wpXgbValAccuracy, wpXgbNoiseLevelPerc = xgbNoiseEffect(wpData, 0, 50, 51)
## dt 
wpDtTestAccuracy, wpDtValAccuracy, wpDtNoiseLevelPerc = dtNoiseEffect(wpData, 0, 50, 51)

####### Census Income dataset 
## rf     
cIRfTestAccuracy, cIRfValAccuracy, cIRfNoiseLevelPerc = rfNoiseEffect(cIData, 0, 50, 51)
## xgb
cIXgbTestAccuracy, cIXgbValAccuracy, cIXgbNoiseLevelPerc = xgbNoiseEffect(cIData, 0, 50, 51)
## dt 
cIDtTestAccuracy, cIDtValAccuracy, cIDtNoiseLevelPerc = dtNoiseEffect(cIData, 0, 50, 51)

###### Credit Card Default dataset 
## rf
cCDRfTestAccuracy, cCDRfValAccuracy, cCDRfNoiseLevelPerc = rfNoiseEffect(cCDData, 0, 50, 51)
## xgb
cCDXgbTestAccuracy, cCDXgbValAccuracy, cCDXgbNoiseLevelPerc = xgbNoiseEffect(cCDData, 0, 50, 51)
## dt 
cCDDtTestAccuracy, cCDDtValAccuracy, cCDDtNoiseLevelPerc = dtNoiseEffect(cCDData, 0, 50, 51)



############################ Plots ##############################
############# Generate Simple plots for each algorithm applied to each dataset #####
#### Waterpump dataset 
## rf
createSingleNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy, wpRfNoiseLevelPerc, 
                            "Water Pump dataset", "Random Forest")
## xgb
createSingleNoiseEffectPlot(wpXgbTestAccuracy, wpXgbValAccuracy, wpXgbNoiseLevelPerc,
                            "Water Pump dataset", "XGBoost")
## dt
createSingleNoiseEffectPlot(wpDtTestAccuracy, wpDtValAccuracy, wpDtNoiseLevelPerc, 
                            "Water Pump dataset", "Decsion Tree")


#### Census Income 
## rf
createSingleNoiseEffectPlot(cIRfTestAccuracy, cIRfValAccuracy, cIRfNoiseLevelPerc, 
                            "Census Income dataset", "Random Forest")
## xgb
createSingleNoiseEffectPlot(cIXgbTestAccuracy, cIXgbValAccuracy, cIXgbNoiseLevelPerc, 
                            "Census Income dataset", "XGBoost")
## dt
createSingleNoiseEffectPlot(cIDtTestAccuracy, cIDtValAccuracy, cIDtNoiseLevelPerc, 
                            "Census Income dataset", "Decison Tree")
   
#### Credit Card Default  
## rf
createSingleNoiseEffectPlot(cCDRfTestAccuracy, cCDRfValAccuracy, cCDRfNoiseLevelPerc, 
                            "Credit Card Default dataset", "Random Forest")
## xgb
createSingleNoiseEffectPlot(cCDXgbTestAccuracy, cCDXgbValAccuracy, cCDXgbNoiseLevelPerc, 
                            "Credit Card Default dataset", "XGBoost")
## dt
createSingleNoiseEffectPlot(cCDDtTestAccuracy, cCDDtValAccuracy, cCDDtNoiseLevelPerc, 
                            "Credit Card Default dataset", "Decison Tree")
   

#############  Noise Effect for specific ml algorithm on multiple datasets ######
#### rf
createMlAlgorithmNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy,
                                 cIRfTestAccuracy, cIRfValAccuracy,
                                 cCDRfTestAccuracy, cCDRfValAccuracy,
                                 wpRfNoiseLevelPerc, "Random Forest")
#### xgb 
createMlAlgorithmNoiseEffectPlot(wpXgbTestAccuracy, wpXgbValAccuracy,
                                 cIXgbTestAccuracy, cIXgbValAccuracy,
                                 cCDXgbTestAccuracy, cCDXgbValAccuracy,
                                 wpXgbNoiseLevelPerc, "XGBoost")
#### dt 
createMlAlgorithmNoiseEffectPlot(wpDtTestAccuracy, wpDtValAccuracy,
                                 cIDtTestAccuracy, cIDtValAccuracy,
                                 cCDDtTestAccuracy, cCDDtValAccuracy,
                                 wpDtNoiseLevelPerc, "Decison Tree")

############## Average Ml Accuarcy for multiple datasets when noise is inserted #######
createMultipleNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy, wpXgbTestAccuracy, wpXgbValAccuracy, wpDtTestAccuracy, wpDtValAccuracy,
                              cIRfTestAccuracy, cIRfValAccuracy, cIXgbTestAccuracy, cIXgbValAccuracy, cIDtTestAccuracy, cIDtValAccuracy,
                              cCDRfTestAccuracy, cCDRfValAccuracy, cCDXgbTestAccuracy, cCDXgbValAccuracy, cCDDtTestAccuracy, cCDDtValAccuracy,
                              wpRfNoiseLevelPerc)
   
     
    

