# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:02:43 2021

@author: micha

This is the master script of the module from which instances can be created of 
machine learning algorthm classes such as Random forest, support vector machine, 
decision trees, and XGBoost

Functions:
    getCooksDistance()
    insertingNoise()
        Helper functions of this function:
            insertingNoiseTestSet()
    noiseEffect()
        Helper functions of this function:
            rfNoiseEffect()
            xgbNoiseEffect()
            dtNoiseEffect()
            svmNoiseEffect()
    createSingleNoiseEffectPlot()
    createMlAlgorithmNoiseEffectPlot()
    createMultipleNoiseEffectPlot()
"""
############# External Imports
import os
import pandas as pd 
import matplotlib.pyplot as plt
from yellowbrick.regressor import CooksDistance #cooks_distance
import statistics

############ Internal Imports (i.e. Self written)
# Change directory to correctDir or script won't run correctly
CORRECTDIR = "C:\\Users\\micha\\Documents\\3rd year\\Software Applications\\PythonSpyder(Anaconda V)\\final_year_project_optum"
os.chdir(CORRECTDIR)
##### Import scripts I've written. Do this after changed to correct directory 
import scripts_and_data



################   Create references to the classes of the ML Algorithms and the datasets ################

######### Datasets 
#### WaterPump (wp) Dataset 
wpData = scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump 

#### Census Income dataset 
cIData = scripts_and_data.data.census_income_dataset.CensusIncome

#### Credit Card Default dataset 
cCDData = scripts_and_data.data.credit_card_default_dataset.CreditCardDefault


######### Machine Learning Algorithms 
##### Random Forest
rf = scripts_and_data.scripts.random_forest 

##### XGBoost
xgb = scripts_and_data.scripts.xgboost_script 

##### Decision Tree
dt = scripts_and_data.scripts.decision_tree

##### Support vector machine
svm = scripts_and_data.scripts.support_vector_machine


######################################################################################################################
######################################### Examine noise effect on ML algortihms ######################################

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


## Now write functions to add noise over a specified range, and then plot the resulting change in accuracy 

def noiseEffect(mlAlgoScriptRef, dataRef, noisePercLevels, nTrees = 100):
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
    noisePercLevels : [float]
        DESCRIPTION. List of the percentages of noise that will be inserted into the test and training sets.

    Returns
    -------
    [] 
        DESCRIPTION. List of test Accuaracys of random forest model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuaracys of random forest model when noise is applied 
    [] 
        DESCRIPTION. List of % noise increments that have been applied 

    """
   
    testAccuracy, valAccuracy, valF1, valAuc = [], [], [], []
    train = dataRef.TRAIN 
    xTest = dataRef.XTEST
    yTest = dataRef.YTEST
    
    for noisePerc in noisePercLevels:
        
        # Get average accuracy at this perc noise interval
        testAccuaracyAtIncrement = []
        valAccuaracyAtIncrement = []
        valF1AtIncrement = []
        valAucAtIncrement = []
        
        # Get average of only 10 models to speed up system 
        for i in range(10): 
            train = insertingNoise(dataRef.TRAIN, noisePerc)
            xTest, yTest = insertingNoiseTestSet(dataRef.XTEST, dataRef.YTEST, noisePerc)
            obj = mlAlgoScriptRef.Model(dataRef.TARGET_VAR_NAME, train, xTest, yTest, dataRef.XVALID, dataRef.YVALID)
            obj.createModel(nTrees) # train the model (If an argument is passed to a function that takes no arguments, the argument will be ignored)
            # Get and append model test and validation accuracy
            modelTestAccuracy = obj.modelAccuracy() 
            testAccuaracyAtIncrement.append(modelTestAccuracy)
            modelValAccuaracy = obj.validAccuracy()
            valAccuaracyAtIncrement.append(modelValAccuaracy)
            modelValF1 = obj.F1ScoreValid()
            valF1AtIncrement.append(modelValF1)
            
            # delete variables that take up a lot of space
            del train, xTest, yTest, obj
            
        testAccuracy.append(statistics.mean(testAccuaracyAtIncrement))
        valAccuracy.append(statistics.mean(valAccuaracyAtIncrement))
        valF1.append(statistics.mean(valF1AtIncrement))
    
    return testAccuracy, valAccuracy, valF1
        
def rfNoiseEffect(dataRef, noisePercLevels, nTrees = 100):
    """
    Helper Function for general noise Effect function. This will call the noiseEffect function 
    will add noise to data used to train random forest models

    Parameters
    ----------
    dataRef : Class Reference
        DESCRIPTION. Refernece to the class of a dataset E.g.: scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump
    noisePercLevels : [float]
        DESCRIPTION. List of the percentages of noise that will be inserted into the test and training sets.
        
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
    return noiseEffect(rfScriptRef, dataRef, noisePercLevels, nTrees)

def xgbNoiseEffect(dataRef, noisePercLevels, nTrees = 100):
    """
    Helper Function for general noise Effect function. This will call the noiseEffect function 
    will add noise to data used to train xgboost models

    Parameters
    ----------
    dataRef : Class Reference
        DESCRIPTION. Refernece to the class of a dataset E.g.: scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump
    noisePercLevels : [float]
        DESCRIPTION. List of the percentages of noise that will be inserted into the test and training sets.

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
    return noiseEffect(xgbScriptRef, dataRef, noisePercLevels, nTrees)

def dtNoiseEffect(dataRef, noisePercLevels):
    """
    Helper Function for general noise Effect function. This will call the noiseEffect function 
    will add noise to data used to train decision tree models

    Parameters
    ----------
    dataRef : Class Reference
        DESCRIPTION. Refernece to the class of a dataset E.g.: scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump
    noisePercLevels : [float]
        DESCRIPTION. List of the percentages of noise that will be inserted into the test and training sets.

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
    return noiseEffect(dtScriptRef, dataRef, noisePercLevels)

def svmNoiseEffect(dataRef, noisePercLevels):
    """
    Helper Function for general noise Effect function. This will call the noiseEffect function 
    will add noise to data used to train support vector machine models

    Parameters
    ----------
    dataRef : Class Reference
        DESCRIPTION. Refernece to the class of a dataset E.g.: scripts_and_data.data.water_pump_dataset.WaterPump # reference of class WaterPump
    noisePercLevels : [float]
        DESCRIPTION. List of the percentages of noise that will be inserted into the test and training sets.

    Returns
    -------
    [] 
        DESCRIPTION. List of test Accuaracys of decision tree model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuaracys of decision tree model when noise is applied 
    [] 
        DESCRIPTION. List of % noise increments that have been applied 

    """
    dtScriptRef = scripts_and_data.scripts.support_vector_machine
    return noiseEffect(dtScriptRef, dataRef, noisePercLevels)

def createSingleNoiseEffectPlot(testAccuracy, valAccuracy, noisePercLevels, datasetName, algorithmName):
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
    noisePercLevels : [float]
        DESCRIPTION. List of the percentages of noise that have been inserted into the test and training sets.
        This should correspond to the test and val accuracy ranges of noise. 
    datasetName : String
        DESCRIPTION.
    algorithmName : String
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plt.figure() # Instantiate a new figure 
    plt.plot(noisePercLevels, testAccuracy,'r--', label = "Test")
    plt.plot(noisePercLevels, valAccuracy, 'g--', label = "Validation")
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Noise Effect on {} Accuracy for {}".format(algorithmName, datasetName), fontsize=18)
    plt.title("Note: Noise randomly inserted to binary target variable in training and test sets", fontsize=12)
    #plt.close() # Close current fig so nothing further will be overlayed on it 
    
def createMlAlgorithmNoiseEffectPlot(wpTestAccuracy, wpValAccuracy, 
                                     cITestAccuracy, cIValAccuracy,
                                     cCDTestAccuracy, cCDValAccuracy, 
                                     noisePercLevels, mlAlgorithmName):
    """
    Function to generate a plot depicting how the accuracy of machine learning alrgorithm Random forests is
    affected by adding noise to the target variable in the test and training sets for multiple datasets.  

    Parameters
    ----------
    wpTestAccuracy : [float]
        DESCRIPTION. List of Test accuracys for waterpump dataset
    wpValAccuracy : [float]
        DESCRIPTION. List of Validation accuracys for waterpump dataset
    ciTestAccuracy : [float]
        DESCRIPTION. List of Test accuracys for Census Income dataset
    ciValAccuracy : [float]
        DESCRIPTION. List of Validation accuracys for Census Income dataset
    cCDTestAccuracy : [float]
        DESCRIPTION. List of Test accuracys for Credit Card Default dataset
    cCDValAccuracy : [float]
        DESCRIPTION. List of Validation accuracys for Credit Card Default dataset
    noisePercLevels : []
        DESCRIPTION. **Assumes that the noise increments is the same for all datasets**. List of noise levels which
                     should correspond to the test and validation accuracys 
    mlAlgorithmName : String
        DESCRIPTION. E.g. "Random Forest" or "XGBoost" or "Decsion Tree" or "Support Vector Machine". This 
        will only be used to insert the machine learning algorithm name into the title of the plot
                    

    Returns
    -------
    None.

    """
    plt.figure()
    plt.plot(noisePercLevels, wpTestAccuracy, color = 'red', ls = '--', label = "Test WaterPump Dataset")
    plt.plot(noisePercLevels, wpValAccuracy, color = 'red', ls = '-', label = "Validation WaterPump Dataset")
    plt.plot(noisePercLevels, cITestAccuracy, color = 'green', ls = '--', label = "Test Census Income Dataset")
    plt.plot(noisePercLevels, cIValAccuracy, color = 'green', ls = '-', label = "Validation Census Income Dataset")
    plt.plot(noisePercLevels, cCDTestAccuracy, color = 'royalblue', ls ='--', label = "Test Credit Card Default Dataset")
    plt.plot(noisePercLevels, cCDValAccuracy, color = 'royalblue', ls = '-', label = "Validation Credit Card Default Dataset")
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Noise Effect on {} Accuracy".format(mlAlgorithmName), fontsize=18)
    plt.title("Note: Noise randomly inserted to binary target variable in training and test sets", fontsize=12)
    
def createMultipleNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy, wpXgbTestAccuracy, wpXgbValAccuracy, wpDtTestAccuracy, wpDtValAccuracy,
                                  cIRfTestAccuracy, cIRfValAccuracy, cIXgbTestAccuracy, cIXgbValAccuracy, cIDtTestAccuracy, cIDtValAccuracy, 
                                  cCDRfTestAccuracy, cCDRfValAccuracy, cCDXgbTestAccuracy, cCDXgbValAccuracy, cCDDtTestAccuracy, cCDDtValAccuracy,
                                  noisePercLevels):
    """
    Generate a matplotlib plot showcasing the noise effect on the average accuracy of Random Forest, 
    XGBoost and decision tree models applied to 3 datasets. 

    Parameters
    ----------
    wpRfTestAccuracy : [float]
        DESCRIPTION.
    wpRfValAccuracy : [float]
        DESCRIPTION.
    wpXgbTestAccuracy : [float]
        DESCRIPTION.
    wpXgbValAccuracy : [float]
        DESCRIPTION.
    wpDtTestAccuracy : [float]
        DESCRIPTION.
    wpDtValAccuracy : [float]
        DESCRIPTION.
    cIRfTestAccuracy : [float]
        DESCRIPTION.
    cIRfValAccuracy : [float]
        DESCRIPTION.
    cIXgbTestAccuracy : [float]
        DESCRIPTION.
    cIXgbValAccuracy : [float]
        DESCRIPTION.
    cIDtTestAccuracy : [float]
        DESCRIPTION.
    cIDtValAccuracy : [float]
        DESCRIPTION.
    cCDRfTestAccuracy : [float]
        DESCRIPTION.
    cCDRfValAccuracy : [float]
        DESCRIPTION.
    cCDXgbTestAccuracy : [float]
        DESCRIPTION.
    cCDXgbValAccuracy : [float]
        DESCRIPTION.
    cCDDtTestAccuracy : [float]
        DESCRIPTION.
    cCDDtValAccuracy : [float]
        DESCRIPTION.
    noisePercLevels : [float]
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
    plt.plot(noisePercLevels, meanRfTestAccuracy, color = 'brown', linestyle = 'dotted', label = "Random Forest Test")
    plt.plot(noisePercLevels, meanRfValAccuracy, color = 'brown', ls='-', label = "Random Forest Validation")
    # xgb
    plt.plot(noisePercLevels, meanXgbTestAccuracy, color = 'cyan', linestyle = 'dotted', label = "XGBoost Test")
    plt.plot(noisePercLevels, meanXgbValAccuracy, color = 'cyan', ls='-', label = "XGBoost Validation")
    ## dt 
    plt.plot(noisePercLevels, meanDtTestAccuracy, color = 'gold', linestyle = 'dotted', label = "Decsion Tree Test")
    plt.plot(noisePercLevels, meanDtValAccuracy, color = 'gold', ls='-', label = "Decsion Tree Validation")
    
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Average Noise Effect on Machine Learning Algorithms Accuracy", fontsize=18)
    plt.title("Note: Noise randomly inserted to binary target variable in training and test sets", fontsize=12)
  

"""
########### Get accuracy of specifc datasets and algorithms for specific noise levels ########
#### Constants 
## Create the noiseLevelPerc list for the % of noise to insert and for subsequent graphing purposes.
NOISEPERCLEVELS = list(range(0, 51, 1)) # 0,1,2,....,50 
# (Note: It doesnt make theoretical sense to insert noise greater than 50%, as you would see a reflection)
# To change simply change range in the format range(startAt, stopBefore, incrementBy)

####### WaterPump dataset 
## rf     
wpRfTestAccuracy, wpRfValAccuracy, wpRfValF1 = rfNoiseEffect(wpData, NOISEPERCLEVELS)
## xgb
wpXgbTestAccuracy, wpXgbValAccuracy, wpXgbValF1 = xgbNoiseEffect(wpData, NOISEPERCLEVELS)
## dt 
wpDtTestAccuracy, wpDtValAccuracy, wpDtValF1 = dtNoiseEffect(wpData, NOISEPERCLEVELS)
## svm 
wpSvmTestAccuracy, wpSvmValAccuracy, wpSvmValF1 = svmNoiseEffect(wpData, NOISEPERCLEVELS)

####### Census Income dataset 
## rf     
cIRfTestAccuracy, cIRfValAccuracy = rfNoiseEffect(cIData, NOISEPERCLEVELS)
## xgb
cIXgbTestAccuracy, cIXgbValAccuracy = xgbNoiseEffect(cIData, NOISEPERCLEVELS)
## dt 
cIDtTestAccuracy, cIDtValAccuracy = dtNoiseEffect(cIData, NOISEPERCLEVELS)
## svm 
cISvmTestAccuracy, cISvmValAccuracy = svmNoiseEffect(cIData, NOISEPERCLEVELS)

###### Credit Card Default dataset 
## rf
cCDRfTestAccuracy, cCDRfValAccuracy = rfNoiseEffect(cCDData, NOISEPERCLEVELS)
## xgb
cCDXgbTestAccuracy, cCDXgbValAccuracy = xgbNoiseEffect(cCDData, NOISEPERCLEVELS)
## dt 
cCDDtTestAccuracy, cCDDtValAccuracy = dtNoiseEffect(cCDData, NOISEPERCLEVELS)
## svm
cCDSvmTestAccuracy, cCDSvmValAccuracy = svmNoiseEffect(cCDData, NOISEPERCLEVELS)


############################ Plots ##############################

############# Generate Simple plots for each algorithm applied to each dataset #####
#### Waterpump dataset 
## rf
createSingleNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy, NOISEPERCLEVELS, 
                            "Water Pump dataset", "Random Forest")
## xgb
createSingleNoiseEffectPlot(wpXgbTestAccuracy, wpXgbValAccuracy, NOISEPERCLEVELS,
                            "Water Pump dataset", "XGBoost")
## dt
createSingleNoiseEffectPlot(wpDtTestAccuracy, wpDtValAccuracy, NOISEPERCLEVELS, 
                            "Water Pump dataset", "Decsion Tree")
## svm
createSingleNoiseEffectPlot(wpSvmTestAccuracy, wpSvmValAccuracy, NOISEPERCLEVELS, 
                            "Water Pump dataset", "Support Vector Machine")

#### Census Income 
## rf
createSingleNoiseEffectPlot(cIRfTestAccuracy, cIRfValAccuracy, NOISEPERCLEVELS, 
                            "Census Income dataset", "Random Forest")
## xgb
createSingleNoiseEffectPlot(cIXgbTestAccuracy, cIXgbValAccuracy, NOISEPERCLEVELS, 
                            "Census Income dataset", "XGBoost")
## dt
createSingleNoiseEffectPlot(cIDtTestAccuracy, cIDtValAccuracy, NOISEPERCLEVELS, 
                            "Census Income dataset", "Decison Tree")
## svm 
createSingleNoiseEffectPlot(cISvmTestAccuracy, cISvmValAccuracy, NOISEPERCLEVELS, 
                            "Census Income dataset", "Support Vector Machine")
   
#### Credit Card Default  
## rf
createSingleNoiseEffectPlot(cCDRfTestAccuracy, cCDRfValAccuracy, NOISEPERCLEVELS, 
                            "Credit Card Default dataset", "Random Forest")
## xgb
createSingleNoiseEffectPlot(cCDXgbTestAccuracy, cCDXgbValAccuracy, NOISEPERCLEVELS, 
                            "Credit Card Default dataset", "XGBoost")
## dt
createSingleNoiseEffectPlot(cCDDtTestAccuracy, cCDDtValAccuracy, NOISEPERCLEVELS, 
                            "Credit Card Default dataset", "Decison Tree")
## svm 
#createSingleNoiseEffectPlot(cCDSvmTestAccuracy, cCDSvmValAccuracy, NOISEPERCLEVELS, 
#                            "Credit Card Default dataset", "Support Vector Machine")


#############  Noise Effect for specific ml algorithm on multiple datasets ######
#### rf
createMlAlgorithmNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy,
                                 cIRfTestAccuracy, cIRfValAccuracy,
                                 cCDRfTestAccuracy, cCDRfValAccuracy,
                                 NOISEPERCLEVELS, "Random Forest")
#### xgb 
createMlAlgorithmNoiseEffectPlot(wpXgbTestAccuracy, wpXgbValAccuracy,
                                 cIXgbTestAccuracy, cIXgbValAccuracy,
                                 cCDXgbTestAccuracy, cCDXgbValAccuracy,
                                 NOISEPERCLEVELS, "XGBoost")
#### dt 
createMlAlgorithmNoiseEffectPlot(wpDtTestAccuracy, wpDtValAccuracy,
                                 cIDtTestAccuracy, cIDtValAccuracy,
                                 cCDDtTestAccuracy, cCDDtValAccuracy,
                                 NOISEPERCLEVELS, "Decison Tree")

#### svm
createMlAlgorithmNoiseEffectPlot(wpSvmTestAccuracy, wpSvmValAccuracy,
                                 cISvmTestAccuracy, cISvmValAccuracy,
                                 cCDSvmTestAccuracy, cCDSvmValAccuracy,
                                 NOISEPERCLEVELS, "Support Vector Machine")


############## Average Ml Accuarcy for multiple datasets when noise is inserted #######
createMultipleNoiseEffectPlot(wpRfTestAccuracy, wpRfValAccuracy, wpXgbTestAccuracy, wpXgbValAccuracy, wpDtTestAccuracy, wpDtValAccuracy,
                              cIRfTestAccuracy, cIRfValAccuracy, cIXgbTestAccuracy, cIXgbValAccuracy, cIDtTestAccuracy, cIDtValAccuracy,
                              cCDRfTestAccuracy, cCDRfValAccuracy, cCDXgbTestAccuracy, cCDXgbValAccuracy, cCDDtTestAccuracy, cCDDtValAccuracy,
                              NOISEPERCLEVELS)
  
      


################## Saving results ###########
##### Due to long running time 
# Store Results in Dict  
dataDict = {"rfWpTest": wpRfTestAccuracy, "rfWpVal": wpRfValAccuracy, "xgbWpTest": wpXgbTestAccuracy, "xgbWpVal": wpXgbValAccuracy,
            "dtWpTest": wpDtTestAccuracy, "dtWpVal": wpDtValAccuracy, "svmWpTest": wpSvmTestAccuracy, "svmWpVal": wpSvmValAccuracy,
            "rfCiTest": cIRfTestAccuracy, "rfCiVal": cIRfValAccuracy, "xgbCiTest": cIXgbTestAccuracy, "xgbCiVal": cIXgbValAccuracy,
            "dtCiTest": cIDtTestAccuracy, "dtCiVal": cIDtValAccuracy, "svmCiTest": cISvmTestAccuracy, "svmCiVal": cISvmValAccuracy,
            "rfCCdTest": cCDRfTestAccuracy, "rfCCdVal": cCDRfValAccuracy, "xgbCCdTest": cCDXgbTestAccuracy, "xgbCCdVal": cCDXgbValAccuracy,
            "dtCCdTest": cCDDtTestAccuracy, "dtCCdVal": cCDDtValAccuracy, "svmCCdTest": cCDSvmTestAccuracy, "svmCCdVal": cCDSvmValAccuracy}
dataItems = dataDict.items()
dataList = list(dataItems)
df = pd.DataFrame(dataList)
csvFilePath = os.path.join(CORRECTDIR, "\\accuracy_results.csv")
# Save results as a csv file
df.to_csv(csvFilePath, header = True)
"""


###################################################################################################################################
################################ Trialing potential Methods of Mitigating Noise ###################################################

########################### Using cooks distance to remove influential datapoints #####################
def getCooksDistance(data):
    """
    Function to get cooks distance for a pandas dataframe
    
    This function takes in a pandas dataframe and returns an object of yellowbrick.regressor.influence.
    A plot n then be called from this returned object using the method .show()

    src documenation: https://www.scikit-yb.org/en/latest/api/regressor/influence.html
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
    
    # Instantiate and fit the cooks distance object
    cooksD = CooksDistance()
    cooksD.fit(predictorData, targetData)
     
    """ 
    # Instantiate and fit the visualizer
    cooksD = cooks_distance(
        predictorData, targetData,
        draw_threshold = True,
        linefmt = "C0-", markerfmt = ","
    )
    """
    
    # Get outlier percentage (% where Cooks Distance is greater than the influence threshold)
    #cooksD.outlier_percentage_

    # Get distances and influence threshold 
    distances, inflThreshold = cooksD.distance_, cooksD.influence_threshold_
    
    # Free up memory by deleting object 
    del cooksD 
    
    return distances, inflThreshold


#cooksDists, cooksThreshold = getCooksDistance(wpData.TRAIN)


def swapPercInfluenentialPoints(train, xTest, yTest, influentialTrain, influentialSwapPerc): 
    if( influentialSwapPerc > 100 or influentialSwapPerc < 0):
        print("Error: You cant have a swap percentage in excess of 100 or below 0. Please try again")
        return 0
    
    # Concat the xTest and yTest together
    test = pd.concat([xTest, yTest], axis = 1)
    
    # get number of obs to change from the training set to the test
    numObs = len(influentialTrain) 
    numToChange = round(numObs * influentialSwapPerc/100)
    # randomly select the number of rows from the dataframe (without replacement)
    toChangeDists = influentialTrain.sample(n = numToChange) # add replace = True for replacement sampling 
    
    ### Remove the observations to change from the training set
    toChangeIndexes = toChangeDists.index # get index of observations to change #******** might need to add [0]
    # get observations with these indexes from the training set
    influentialPoints = train.loc[toChangeIndexes, : ] #************ Problem with this as it assumes index goes from 1->maxlength-1
    # and remove observations with these indexes from the training set
    newTrain = train.copy() # create copy so original is not overwitten 
    newTrain.drop(toChangeIndexes, inplace = True)
    
    ### Now add the toChange df to the test set 
    newTest = pd.concat([test, influentialPoints], axis = 0) #This will result in 
    ### Split back into xTest and yTest
    newXTest, newYTest = newTest.iloc[:, :-1], newTest.iloc[:, -1]
    
    return newTrain, newXTest, newYTest, numToChange, len(train) 

def influentialPointEffect(mlAlgoScriptRef, dataRef, noisePercLevels, nTrees = 100):
   
    # Create list of percentages of most influential points that will be swapped from the training to test set
    influentialSwapPercLevels = [0, 20, 40, 60, 80, 100]
    
    # Create copy of training set so original is not overwitten 
    train = dataRef.TRAIN
    xTest = dataRef.XTEST 
    yTest = dataRef.YTEST
   
    # Create dict to store results 
    results = {}
    
    for swapPerc in influentialSwapPercLevels:
        # Create lists to store the results
        testAccuracy = []
        valAccuracy = []
        
        numSwappeddddd = []
        lenTrainnnn = []
        
        for noisePerc in noisePercLevels:
            # Repeat 10 times and get the average accuracy 
            testAccuaracyAtIncrement = []
            valAccuaracyAtIncrement = []
            valF1AtIncrement = []
            
            numSwapped = 0 
            lenTrain = 0
            
            for i in range(5): # Get average of only 5 models to speed up system
                # Insert % noise into the test and training sets
                noiseTrain = insertingNoise(train, noisePerc)
                noiseXTest, noiseYTest = insertingNoiseTestSet(xTest, yTest, noisePerc)
                
                ### Swapping the top % of most influential points from the training to test set
                # Get influential points in the training set using cooks distance 
                cooksDists, cooksThreshold = getCooksDistance(noiseTrain)
                # Get the influential points df
                cooksDists = pd.DataFrame(cooksDists, columns = ['distances'])
                influentialTrainDf = cooksDists[cooksDists['distances'] > cooksThreshold]
                influentialTrainDf.sort_values(by = 'distances', ascending = False, inplace = True)
                # Now get the new training and test setafter the influential points have been swapped 
                swappedTrain, swappedXTest, swappedYTest, numSwapped, lenTrain = swapPercInfluenentialPoints(
                                                                                    noiseTrain, noiseXTest, noiseYTest, 
                                                                                    influentialTrainDf, swapPerc)
    
                ### Now create an object of the relevant machine learning algorithm class 
                obj = mlAlgoScriptRef.Model(dataRef.TARGET_VAR_NAME, swappedTrain, swappedXTest, swappedYTest,
                                            dataRef.XVALID, dataRef.YVALID)
                obj.createModel(nTrees) # train the model 
                # (If an argument is passed to a function that takes no arguments, the argument will be ignored)
                # Get and append model test and validation accuracy
                modelTestAccuracy = obj.modelAccuracy() 
                testAccuaracyAtIncrement.append(modelTestAccuracy)
                modelValAccuaracy = obj.validAccuracy()
                valAccuaracyAtIncrement.append(modelValAccuaracy)
                
                
                # Free up memory by deleting object
                del obj 
            
            # Get the average accuracy and add it to the test and val accuracy lists
            testAccuracy.append(statistics.mean(testAccuaracyAtIncrement))
            valAccuracy.append(statistics.mean(valAccuaracyAtIncrement))
            
            lenTrainnnn.append(lenTrain)
            numSwappeddddd.append(numSwapped)
        
        ## Add the accuracys to results 
        results['{}%Test'.format(swapPerc)] = testAccuracy
        results['{}%Val'.format(swapPerc)] = valAccuracy
        results['{}%NumSwapped'.format(swapPerc)] = numSwappeddddd
        results['{}%LengthTrain'.format(swapPerc)] = lenTrainnnn
    
    return results


def createCooksDistNoiseMitigationPlot(wpResults, cIResults, cCDResults, mlAlgorithmName, noiseLevelPerc):
    plt.figure() # Instantiate a new figure 
    
    # Get mean accuracy for each 
    mean0PercSwapTestAccuracy,  mean0PercSwapValAccuracy = [], []
    mean20PercSwapTestAccuracy,  mean20PercSwapValAccuracy = [], []
    mean40PercSwapTestAccuracy,  mean40PercSwapValAccuracy = [], []
    mean60PercSwapTestAccuracy,  mean60PercSwapValAccuracy = [], []
    mean80PercSwapTestAccuracy,  mean80PercSwapValAccuracy = [], []
    mean100PercSwapTestAccuracy, mean100PercSwapValAccuracy = [], []

    # Get length of noise level percentages list (to ascertain how long each list in the results dicts are)
    for i in range( len(noiseLevelPerc) ): #len(next(iter(wpRfNoiseMitigationResults.values()))) ):
        mean0PercSwapTestAccuracy.append( (wpResults['0%Test'][i] + cIResults['0%Test'][i] + cCDResults['0%Test'][i])/3 )
        mean0PercSwapValAccuracy.append( (wpResults['0%Val'][i] + cIResults['0%Val'][i] + cCDResults['0%Val'][i])/3 )
       
        mean20PercSwapTestAccuracy.append( (wpResults['20%Test'][i] + cIResults['20%Test'][i] + cCDResults['20%Test'][i])/3 )
        mean20PercSwapValAccuracy.append( (wpResults['20%Val'][i] + cIResults['20%Val'][i] + cCDResults['20%Val'][i])/3 )
       
        mean40PercSwapTestAccuracy.append( (wpResults['40%Test'][i] + cIResults['40%Test'][i] + cCDResults['40%Test'][i])/3 )
        mean40PercSwapValAccuracy.append( (wpResults['40%Val'][i] + cIResults['40%Val'][i] + cCDResults['40%Val'][i])/3 )
        
        mean60PercSwapTestAccuracy.append( (wpResults['60%Test'][i] + cIResults['60%Test'][i] + cCDResults['60%Test'][i])/3 )
        mean60PercSwapValAccuracy.append( (wpResults['60%Val'][i] + cIResults['60%Val'][i] + cCDResults['60%Val'][i])/3 )
       
        mean80PercSwapTestAccuracy.append( (wpResults['80%Test'][i] + cIResults['80%Test'][i] + cCDResults['80%Test'][i])/3 )
        mean80PercSwapValAccuracy.append( (wpResults['80%Val'][i] + cIResults['80%Val'][i] + cCDResults['80%Val'][i])/3 )
       
        mean100PercSwapTestAccuracy.append( (wpResults['100%Test'][i] + cIResults['100%Test'][i] + cCDResults['100%Test'][i])/3 )
        mean100PercSwapValAccuracy.append( (wpResults['100%Val'][i] + cIResults['100%Val'][i] + cCDResults['100%Val'][i])/3 )
       
    
    # 0%
    plt.plot(noiseLevelPerc, mean0PercSwapTestAccuracy, color = 'pink', linestyle = 'dotted', label = "0% Test")
    plt.plot(noiseLevelPerc, mean0PercSwapValAccuracy, color = 'pink', ls='-', label = "0% Validation")
    # 20%
    plt.plot(noiseLevelPerc, mean20PercSwapTestAccuracy, color = 'coral', linestyle = 'dotted', label = "20% Test")
    plt.plot(noiseLevelPerc, mean20PercSwapValAccuracy, color = 'coral', ls='-', label = "20% Validation")
    # 40% 
    plt.plot(noiseLevelPerc, mean40PercSwapTestAccuracy, color = 'orangered', linestyle = 'dotted', label = "40% Test")
    plt.plot(noiseLevelPerc, mean40PercSwapValAccuracy, color = 'orangered', ls='-', label = "40% Validation")
    # 60%
    plt.plot(noiseLevelPerc, mean60PercSwapTestAccuracy, color = 'indianred', linestyle = 'dotted', label = "60% Test")
    plt.plot(noiseLevelPerc, mean60PercSwapValAccuracy, color = 'indianred', ls='-', label = "60% Validation")
    # 80%
    plt.plot(noiseLevelPerc, mean0PercSwapTestAccuracy, color = 'maroon', linestyle = 'dotted', label = "80% Test")
    plt.plot(noiseLevelPerc, mean0PercSwapValAccuracy, color = 'maroon', ls='-', label = "80% Validation")
    # 100%
    plt.plot(noiseLevelPerc, mean0PercSwapTestAccuracy, color = 'black', linestyle = 'dotted', label = "100% Test")
    plt.plot(noiseLevelPerc, mean0PercSwapValAccuracy, color = 'black', ls='-', label = "100% Validation")
    
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Using Cooks Distance to Mitigate the Impact of Noise on {} Accuracy".format(mlAlgorithmName), fontsize=18)
    plt.title("Note: -The top X percentage of the ordered most influential points in the training set were put into te test set"
              + "\n-Noise randomly inserted to binary target variable in training and test sets", fontsize=12)


################# Get the results of mitigation of noise effect using cooks distance  
##### Constants
## Create the noiseLevelPerc list for the % of noise to insert and for subsequent graphing purposes.
NOISEPERCLEVELSMITIGATIONEXP = list(range(0, 51, 2)) # 0,1,2,....,50
# To change simply change range in the format range(startAt, stopBefore, incrementBy)

### Temp Testing
wpRfNMRTesting = influentialPointEffect(rf, wpData, NOISEPERCLEVELSMITIGATIONEXP, 100)


######## Water pump dataset 
## rf
wpRfNoiseMitigationResults = influentialPointEffect(rf, wpData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)
## xgb 
wpXgbNoiseMitigationResults = influentialPointEffect(xgb, wpData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)
## svm
wpSvmNoiseMitigationResults = influentialPointEffect(svm, wpData, NOISEPERCLEVELSMITIGATIONEXP)
## dt 
wpDtNoiseMitigationResults = influentialPointEffect(dt, wpData, NOISEPERCLEVELSMITIGATIONEXP)

######## Census Income dataset 
## rf
cIRfNoiseMitigationResults = influentialPointEffect(rf, cIData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)
## xgb 
cIXgbNoiseMitigationResults = influentialPointEffect(xgb, cIData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)
## svm
cISvmNoiseMitigationResults = influentialPointEffect(svm, cIData, NOISEPERCLEVELSMITIGATIONEXP)
## dt 
cIDtNoiseMitigationResults = influentialPointEffect(dt, cIData, NOISEPERCLEVELSMITIGATIONEXP)

######## Credit Card Default dataset 
## rf
cCDRfNoiseMitigationResults = influentialPointEffect(rf, cCDData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)
## xgb 
cCDXgbNoiseMitigationResults = influentialPointEffect(xgb, cCDData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)
## svm
cCDSvmNoiseMitigationResults = influentialPointEffect(svm, cCDData, NOISEPERCLEVELSMITIGATIONEXP)
## dt 
cCDDtNoiseMitigationResults = influentialPointEffect(dt, cCDData, NOISEPERCLEVELSMITIGATIONEXP)


#### Generate general plot where the average accuracy across the 3 datasets for each noise % is calculated
# rf
createCooksDistNoiseMitigationPlot(wpRfNoiseMitigationResults, cIRfNoiseMitigationResults, 
                                   cCDRfNoiseMitigationResults, "Random Forest", NOISEPERCLEVELSMITIGATIONEXP)
# xgb
createCooksDistNoiseMitigationPlot(wpXgbNoiseMitigationResults, cIXgbNoiseMitigationResults, 
                                   cCDXgbNoiseMitigationResults, "XGBoost", NOISEPERCLEVELSMITIGATIONEXP)
# svm
createCooksDistNoiseMitigationPlot(wpSvmNoiseMitigationResults, cISvmNoiseMitigationResults, 
                                   cCDSvmNoiseMitigationResults, "Support Vector Machine", NOISEPERCLEVELSMITIGATIONEXP)
# dt
createCooksDistNoiseMitigationPlot(wpDtNoiseMitigationResults, cIDtNoiseMitigationResults, 
                                   cCDDtNoiseMitigationResults, "Decision Tree", NOISEPERCLEVELSMITIGATIONEXP)



############################ Test if the number of trees provides insulation against noise ######################
def createTreesNoiseInsulationPlot(test20T, val20T, test60T, val60T, test100T, val100T,
                                   test200T, val200T, test500T, val500T,
                                   noisePercLevels, mlAlgoName):
    plt.plot(noisePercLevels, test20T, color = 'pink', ls = '-', label = "20 Trees Test")
    plt.plot(noisePercLevels, val20T, color = 'pink', ls = ':', label = "20 Trees Val")
    plt.plot(noisePercLevels, test60T, color = 'coral', ls = '-', label = "60 Trees Test")
    plt.plot(noisePercLevels, val60T, color = 'coral', ls = ':', label = "60 Trees Val")
    plt.plot(noisePercLevels, test100T, color = 'orangered', ls = '-', label = "100 Trees Test")
    plt.plot(noisePercLevels, val100T, color = 'orangered', ls = ':', label = "100 Trees Val")
    plt.plot(noisePercLevels, test200T, color = 'indianred', ls = '-', label = "200 Trees Test")
    plt.plot(noisePercLevels, val200T, color = 'indianred', ls = ':', label = "200 Trees Val")
    plt.plot(noisePercLevels, test500T, color = 'maroon', ls = '-', label = "500 Trees Test")
    plt.plot(noisePercLevels, val500T, color = 'maroon', ls = ':', label = "500 Trees Val")
    
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Accuracy %")
    plt.suptitle("Insualtion against Noise Effect on {} Accuracy Provided by Number of Trees".format(mlAlgoName), fontsize=18)
    plt.title("Note: Noise randomly inserted to binary target variable in training and test sets", fontsize=12)


############### Get Accuracys to see if increasing the number of trees provides insulation agasints impact of noise ###################
##### Constants
## Create the noiseLevelPerc list for the % of noise to insert and for subsequent graphing purposes.
NOISEPERCLEVELSMITIGATIONEXP = list(range(0, 51, 1)) # 0,1,2,....,50
# To change simply change range in the format range(startAt, stopBefore, incrementBy)


######## Random Forest
## 20 trees
wpRfTest20T, wpRfVal20T = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 20)
## 60 trees
wpRfTest60T, wpRfVal60T = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 60)
## 100 trees
# This is the deafult which has already been run. Results stored in vars: wpRfTestAccuracy, wpRfValAccuracy
# wpRfTest100T, wpRfVal100T = wpRfTestAccuracy, wpRfValAccuracy
wpRfTest100T, wpRfVal100T = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 100) # alternatively run code again 
## 200 trees
wpRfTest200T, wpRfVal200T = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 200)
## 500 trees
wpRfTest500T, wpRfVal500T = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 500)

## Generate plot
createTreesNoiseInsulationPlot(wpRfTest20T, wpRfVal20T, wpRfTest60T, wpRfVal60T, wpRfTest100T, wpRfVal100T,
                               wpRfTest200T, wpRfVal200T, wpRfTest500T, wpRfVal500T, 
                               NOISEPERCLEVELSMITIGATIONEXP, "Random Forest")


######## XGBoost
## 20 trees
wpXgbTest20T, wpXgbVal20T = xgbNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 20)
## 60 trees
wpXgbTest60T, wpXgbVal60T = xgbNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 60)
## 100 trees
# This is the deafult which has already been run. Results stored in vars: wpXgbTestAccuracy, wpXgbValAccuracy, wpXgbNoiseLevelPerc
# wpXgbTest100T, wpXgbVal100T = wpXgbTestAccuracy, wpXgbValAccuracy
wpXgbTest100T, wpXgbVal100T = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 100) # alternatively run code again 
## 200 trees
wpXgbTest200T, wpXgbVal200T = xgbNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 200)
## 500 trees
wpXgbTest500T, wpXgbVal500T = xgbNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 500)

##### Generate plot
createTreesNoiseInsulationPlot(wpXgbTest20T, wpXgbVal20T, wpXgbTest60T, wpXgbVal60T, wpXgbTest100T, wpXgbVal100T,
                               wpXgbTest200T, wpXgbVal200T, wpXgbTest500T, wpXgbVal500T, 
                               NOISEPERCLEVELSMITIGATIONEXP, "Xgboost")




