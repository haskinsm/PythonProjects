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
import numpy as np
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
        DESCRIPTION. List of test Accuracys of random forest model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuracys of random forest model when noise is applied 
    [] 
        DESCRIPTION. List of % noise increments that have been applied 

    """
   
    testAccuracy, valAccuracy, valF1, valAUC = [], [], [], []
    
    for noisePerc in noisePercLevels:
        
        # Get average accuracy at this perc noise interval
        testAccuracyAtIncrement = []
        valAccuracyAtIncrement = []
        valF1AtIncrement = []
        valAUCAtIncrement = []
        
        # Get average of only 10 models to speed up system 
        for i in range(10): 
            train = insertingNoise(dataRef.TRAIN, noisePerc)
            xTest, yTest = insertingNoiseTestSet(dataRef.XTEST, dataRef.YTEST, noisePerc)
            obj = mlAlgoScriptRef.Model(dataRef.TARGET_VAR_NAME, train, xTest, yTest, dataRef.XVALID, dataRef.YVALID)
            obj.createModel(nTrees) # train the model (If an argument is passed to a function that takes no arguments, the argument will be ignored)
            
            # Get and append model perfromance measures to relevant lists
            modelTestAccuracy = obj.modelAccuracy() 
            testAccuracyAtIncrement.append(modelTestAccuracy)
            
            modelValAccuracy = obj.validAccuracy()
            valAccuracyAtIncrement.append(modelValAccuracy)
            
            modelValF1 = obj.validF1Score()
            valF1AtIncrement.append(modelValF1)
            
            modelValAUC = obj.validAUC()
            valAUCAtIncrement.append(modelValAUC)
            
            # delete object as no loner needed 
            del obj
            
        testAccuracy.append(statistics.mean(testAccuracyAtIncrement))
        valAccuracy.append(statistics.mean(valAccuracyAtIncrement))
        valF1.append(statistics.mean(valF1AtIncrement))
        valAUC.append(statistics.mean(valAUCAtIncrement))
    
   
    ## Add the results to a dict
    results = {}
    results['TestAccuracy'] = testAccuracy
    results['ValAccuracy'] = valAccuracy
    results['ValF1'] = valF1
    results['ValAUC'] = valAUC	
    
    return results

        
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
        DESCRIPTION. List of test Accuracys of random forest model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuracys of random forest model when noise is applied 
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
        DESCRIPTION. List of test Accuracys of xgboost model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuracys of xgboost model when noise is applied 
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
        DESCRIPTION. List of test Accuracys of decision tree model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuracys of decision tree model when noise is applied 
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
        DESCRIPTION. List of test Accuracys of decision tree model when noise is applied   
    [] 
        DESCRIPTION. List of validation Accuracys of decision tree model when noise is applied 
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
    
def createMlAlgorithmNoiseEffectPlots(wpResults, 
                                     cIResults,
                                     cCDResults, 
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
    # Create Test Accuracy Plot 
    plt.figure()
    
    plt.plot(noisePercLevels, wpResults['TestAccuracy'], color = 'red', ls = '-', label = "WaterPump (Pump it Up)")
    plt.plot(noisePercLevels, cIResults['TestAccuracy'], color = 'green', ls = '-', label = "Census Income Dataset")
    plt.plot(noisePercLevels, cCDResults['TestAccuracy'], color = 'royalblue', ls ='-', label = "Credit Card Default Dataset")
   
    plt.legend(title = "Dataset")
    plt.xlabel("Noise %")
    plt.ylabel("Average Test Accuracy %")
    plt.suptitle("Noise Effect on {} ".format(mlAlgorithmName) + r"$\bf{Test \ Accuracy}$", fontsize=18)
    #plt.title("Note: Noise randomly inserted to binary target variable in training and test sets", fontsize=12)
    
    
    # Create Val Accuracy Plot 
    plt.figure()
    
    plt.plot(noisePercLevels, wpResults['ValAccuracy'], color = 'red', ls = '-', label = "WaterPump (Pump it Up)")
    plt.plot(noisePercLevels, cIResults['ValAccuracy'], color = 'green', ls = '-', label = "Census Income")
    plt.plot(noisePercLevels, cCDResults['ValAccuracy'], color = 'royalblue', ls = '-', label = "Credit Card Default")
    
    plt.legend(title = "Dataset")
    plt.xlabel("Noise %")
    plt.ylabel("Average Validation Accuracy %")
    plt.suptitle("Noise Effect on {} ".format(mlAlgorithmName) + r"$\bf{Validation \ Accuracy}$", fontsize=18)
    #plt.title("Note: Noise randomly inserted to binary target variable in training set", fontsize=12)
    
    
    # Create F1 Score Plot 
    plt.figure()
    
    plt.plot(noisePercLevels, wpResults['ValF1'], color = 'red', ls = '-', label = "WaterPump (Pump it Up)")
    plt.plot(noisePercLevels, cIResults['ValF1'], color = 'green', ls = '-', label = "Census Income")
    plt.plot(noisePercLevels, cCDResults['ValF1'], color = 'royalblue', ls = '-', label = "Credit Card Default")
    
    plt.legend(title = "Dataset")
    plt.xlabel("Noise %")
    plt.ylabel("Average F1 Score")
    plt.suptitle("Noise Effect on {} ".format(mlAlgorithmName) + r"$\bf{Validation \ F1 \ Score}$", fontsize=18)
    #plt.title("Note: Noise randomly inserted to binary target variable in training set", fontsize=12)
    
    
    # Create AUC Plot 
    plt.figure()
    
    plt.plot(noisePercLevels, wpResults['ValAUC'], color = 'red', ls = '-', label = "WaterPump (Pump it Up)")
    plt.plot(noisePercLevels, cIResults['ValAUC'], color = 'green', ls = '-', label = "Census Income")
    plt.plot(noisePercLevels, cCDResults['ValAUC'], color = 'royalblue', ls = '-', label = "Credit Card Default")
    
    plt.legend(title = "Dataset")
    plt.xlabel("Noise %")
    plt.ylabel("Average AUC Value")
    plt.suptitle("Noise Effect on {} ".format(mlAlgorithmName) + r"$\bf{Validation \ AUC}$", fontsize=18)
    #plt.title("Note: Noise randomly inserted to binary target variable in training set", fontsize=12)
    
    
    
def createMultipleNoiseEffectPlot(wpRfResults, wpXgbResults, wpDtResults, wpSvmResults,
                                  cIRfResults, cIXgbResults, cIDtResults, cISvmResults,
                                  cCDRfResults, cCDXgbResults, cCDDtResults, cCDSvmResults,
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
    
    # Get mean accuracy for each ml algorithm
    meanRfTestAccuracy, meanXgbTestAccuracy, meanDtTestAccuracy, meanSvmTestAccuracy = [], [], [], []
    meanRfValAccuracy, meanXgbValAccuracy, meanDtValAccuracy, meanSvmValAccuracy = [], [], [], []
    meanRfValF1, meanXgbValF1, meanDtValF1, meanSvmValF1 = [], [], [], []
    meanRfValAUC, meanXgbValAUC, meanDtValAUC, meanSvmValAUC = [], [], [], []
    
    # Get length of noise level percentages list (to ascertain how long each list in the results dicts are)
    # Then iterate through each item in the lists and get the average value for each algorithm
    for i in range( len(noisePercLevels) ): 
        # Get mean Test accuracys for each algorithm 
        meanRfTestAccuracy.append( (wpRfResults['TestAccuracy'][i] + cIRfResults['TestAccuracy'][i] + cCDRfResults['TestAccuracy'][i])/3)
        meanXgbTestAccuracy.append( (wpXgbResults['TestAccuracy'][i] + cIXgbResults['TestAccuracy'][i] + cCDXgbResults['TestAccuracy'][i])/3)
        meanDtTestAccuracy.append( (wpDtResults['TestAccuracy'][i] + cIDtResults['TestAccuracy'][i] + cCDDtResults['TestAccuracy'][i])/3)
        meanSvmTestAccuracy.append( (wpSvmResults['TestAccuracy'][i] + cISvmResults['TestAccuracy'][i] + cCDSvmResults['TestAccuracy'][i])/3)

        # Get mean Val accuracys for each algorithm 
        meanRfValAccuracy.append( (wpRfResults['ValAccuracy'][i] + cIRfResults['ValAccuracy'][i] + cCDRfResults['ValAccuracy'][i])/3)
        meanXgbValAccuracy.append( (wpXgbResults['ValAccuracy'][i] + cIXgbResults['ValAccuracy'][i] + cCDXgbResults['ValAccuracy'][i])/3)
        meanDtValAccuracy.append( (wpDtResults['ValAccuracy'][i] + cIDtResults['ValAccuracy'][i] + cCDDtResults['ValAccuracy'][i])/3)
        meanSvmValAccuracy.append( (wpSvmResults['ValAccuracy'][i] + cISvmResults['ValAccuracy'][i] + cCDSvmResults['ValAccuracy'][i])/3)

        # Get mean F1 Score for each algorithm 
        meanRfValF1.append( (wpRfResults['ValF1'][i] + cIRfResults['ValF1'][i] + cCDRfResults['ValF1'][i])/3)
        meanXgbValF1.append( (wpXgbResults['ValF1'][i] + cIXgbResults['ValF1'][i] + cCDXgbResults['ValF1'][i])/3)
        meanDtValF1.append( (wpDtResults['ValF1'][i] + cIDtResults['ValF1'][i] + cCDDtResults['ValF1'][i])/3)
        meanSvmValF1.append( (wpSvmResults['ValF1'][i] + cISvmResults['ValF1'][i] + cCDSvmResults['ValF1'][i])/3)

        # Get mean AUC for each algorithm 
        meanRfValAUC.append( (wpRfResults['ValAUC'][i] + cIRfResults['ValAUC'][i] + cCDRfResults['ValAUC'][i])/3)
        meanXgbValAUC.append( (wpXgbResults['ValAUC'][i] + cIXgbResults['ValAUC'][i] + cCDXgbResults['ValAUC'][i])/3)
        meanDtValAUC.append( (wpDtResults['ValAUC'][i] + cIDtResults['ValAUC'][i] + cCDDtResults['ValAUC'][i])/3)
        meanSvmValAUC.append( (wpSvmResults['ValAUC'][i] + cISvmResults['ValAUC'][i] + cCDSvmResults['ValAUC'][i])/3)

    
    # Create Test Accuracy Plot 
    plt.figure()
    
    plt.plot(noisePercLevels, meanRfTestAccuracy, color = 'teal', ls = '-', label = "Random Forest")
    plt.plot(noisePercLevels, meanXgbTestAccuracy, color = 'darkviolet', ls = '-', label = "XGBoost")
    plt.plot(noisePercLevels, meanDtTestAccuracy, color = 'orange', ls ='-', label = "Decision Tree")
    plt.plot(noisePercLevels, meanSvmTestAccuracy, color = 'darkslategrey', ls ='-', label = "Support Vector Machine")
   
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Test Accuracy %")
    plt.suptitle("Noise Effect on Machine Learning Algortihms " + r"$\bf{Test \ Accuracy}$", fontsize=18)
    #plt.title("Note: Noise randomly inserted to binary target variable in training and test sets", fontsize=12)
    
    
    # Create Val Accuracy Plot 
    plt.figure()
    
    plt.plot(noisePercLevels, meanRfValAccuracy, color = 'teal', ls = '-', label = "Random Forest")
    plt.plot(noisePercLevels, meanXgbValAccuracy, color = 'darkviolet', ls = '-', label = "XGBoost")
    plt.plot(noisePercLevels, meanDtValAccuracy, color = 'orange', ls ='-', label = "Decision Tree")
    plt.plot(noisePercLevels, meanSvmValAccuracy, color = 'darkslategrey', ls ='-', label = "Support Vector Machine")
    
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average Validation Accuracy %")
    plt.suptitle("Noise Effect on Machine Learning Algortihms " + r"$\bf{Validation \ Accuracy}$", fontsize=18)
    #plt.title("Note: Noise randomly inserted to binary target variable in training set", fontsize=12)
    
    
    # Create F1 Score Plot 
    plt.figure()
    
    plt.plot(noisePercLevels, meanRfValF1, color = 'teal', ls = '-', label = "Random Forest")
    plt.plot(noisePercLevels, meanXgbValF1, color = 'darkviolet', ls = '-', label = "XGBoost")
    plt.plot(noisePercLevels, meanDtValF1, color = 'orange', ls ='-', label = "Decision Tree")
    plt.plot(noisePercLevels, meanSvmValF1, color = 'darkslategrey', ls ='-', label = "Support Vector Machine")
    
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average F1 Score")
    plt.suptitle("Noise Effect on Machine Learning Algortihms " + r"$\bf{Validation \ F1 \ Score}$", fontsize=18)
    #plt.title("Note: Noise randomly inserted to binary target variable in training set", fontsize=12)
    
    
    # Create AUC Plot 
    plt.figure()
    
    plt.plot(noisePercLevels, meanRfValAUC, color = 'teal', ls = '-', label = "Random Forest")
    plt.plot(noisePercLevels, meanXgbValAUC, color = 'darkviolet', ls = '-', label = "XGBoost")
    plt.plot(noisePercLevels, meanDtValAUC, color = 'orange', ls ='-', label = "Decision Tree")
    plt.plot(noisePercLevels, meanSvmValAUC, color = 'darkslategrey', ls ='-', label = "Support Vector Machine")
    
    plt.legend()
    plt.xlabel("Noise %")
    plt.ylabel("Average AUC Value")
    plt.suptitle("Noise Effect on Machine Learning Algortihms " + r"$\bf{Validation \ AUC}$", fontsize=18)
    #plt.title("Note: Noise randomly inserted to binary target variable in training set", fontsize=12)
    


########### Get accuracy of specifc datasets and algorithms for specific noise levels ########
#### Constants 
## Create the noiseLevelPerc list for the % of noise to insert and for subsequent graphing purposes.
NOISEPERCLEVELS = list(range(0, 51, 1)) # 0,1,2,....,50 
# (Note: It doesnt make theoretical sense to insert noise greater than 50%, as you would see a reflection)
# To change simply change range in the format range(startAt, stopBefore, incrementBy)

####### WaterPump dataset 
## rf     
wpRfNoiseResults = rfNoiseEffect(wpData, NOISEPERCLEVELS)
## xgb
wpXgbNoiseResults = xgbNoiseEffect(wpData, NOISEPERCLEVELS)
## dt 
wpDtNoiseResults = dtNoiseEffect(wpData, NOISEPERCLEVELS)
## svm 
wpSvmNoiseResults = svmNoiseEffect(wpData, NOISEPERCLEVELS)

####### Census Income dataset 
## rf     
cIRfNoiseResults = rfNoiseEffect(cIData, NOISEPERCLEVELS)
## xgb
cIXgbNoiseResults = xgbNoiseEffect(cIData, NOISEPERCLEVELS)
## dt 
cIDtNoiseResults = dtNoiseEffect(cIData, NOISEPERCLEVELS)
## svm 
cISvmNoiseResults = svmNoiseEffect(cIData, NOISEPERCLEVELS)

###### Credit Card Default dataset 
## rf
cCDRfNoiseResults = rfNoiseEffect(cCDData, NOISEPERCLEVELS)
## xgb
cCDXgbNoiseResults = xgbNoiseEffect(cCDData, NOISEPERCLEVELS)
## dt 
cCDDtNoiseResults = dtNoiseEffect(cCDData, NOISEPERCLEVELS)
## svm
cCDSvmNoiseResults = svmNoiseEffect(cCDData, NOISEPERCLEVELS)


############################ Plots ##############################

#############  Noise Effect for specific ml algorithm on multiple datasets ######
#### rf
createMlAlgorithmNoiseEffectPlots(wpRfNoiseResults,
                                 cIRfNoiseResults,
                                 cCDRfNoiseResults,
                                 NOISEPERCLEVELS, "Random Forest")
#### xgb 
createMlAlgorithmNoiseEffectPlots(wpXgbNoiseResults,
                                 cIXgbNoiseResults,
                                 cCDXgbNoiseResults,
                                 NOISEPERCLEVELS, "XGBoost")
#### dt 
createMlAlgorithmNoiseEffectPlots(wpDtNoiseResults,
                                 cIDtNoiseResults,
                                 cCDDtNoiseResults,
                                 NOISEPERCLEVELS, "Decison Tree")

#### svm
createMlAlgorithmNoiseEffectPlots(wpSvmNoiseResults,
                                 cISvmNoiseResults,
                                 cCDSvmNoiseResults,
                                 NOISEPERCLEVELS, "Support Vector Machine")


############## Average Ml Accuarcy for multiple datasets when noise is inserted #######
createMultipleNoiseEffectPlot(wpRfNoiseResults, wpXgbNoiseResults, wpDtNoiseResults, wpSvmNoiseResults,
                              cIRfNoiseResults, cIXgbNoiseResults, cIDtNoiseResults, cISvmNoiseResults,
                              cCDRfNoiseResults, cCDXgbNoiseResults, cCDDtNoiseResults, cCDSvmNoiseResults,
                              NOISEPERCLEVELS)
  
      

"""
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
    toChangeIndexes = toChangeDists.index # get index of observations to change             #******** might need to add [0]
    # Get observations with these indexes from the training set
    influentialPoints = train.loc[toChangeIndexes, : ]                      #************ Problem with this as it assumes index goes from 1->maxlength-1
    # Remove observations with these indexes from the training set
    train.drop(toChangeIndexes, inplace = True) 
    # No need to create copy of train as copy is already created when noise is inserted so original will not be overwritten
    
    ### Now add the toChange df to the test set 
    newTest = pd.concat([test, influentialPoints], axis = 0)  
    ### Split back into xTest and yTest
    newXTest, newYTest = newTest.iloc[:, :-1], newTest.iloc[:, -1]
    
    ## Delete unneeded large variables from memory 
    del test, newTest
    
    return train, newXTest, newYTest, numToChange 


def influentialPointEffect(mlAlgoScriptRef, dataRef, noisePercLevels, nTrees = 100):
   
    # Create list of percentages of most influential points that will be swapped from the training to test set
    influentialSwapPercLevels = [0, 50, 100] #[0, 20, 40, 60, 80, 100]

    # Create dict to store results 
    results = {}
        
    for swapPerc in influentialSwapPercLevels:
        # Create lists to store the results
        testAccuracy, valAccuracy, valF1, valAUC = [], [], [], []
        numPointsSwapped = []
    
        
        for noisePerc in noisePercLevels:
            # Repeat 10 times and get the average accuracy 
            testAccuracyAtIncrement, valAccuracyAtIncrement = [], []
            valF1AtIncrement, valAUCAtIncrement = [], []
            numPointsSwappedAtIncrement = []
            numSwapped = 0
            
            for i in range(5): # Get average of only 5 models to speed up system
                # Insert % noise into the test and training sets
                noiseTrain = insertingNoise(dataRef.TRAIN, noisePerc)
                noiseXTest, noiseYTest = insertingNoiseTestSet(dataRef.XTEST, dataRef.YTEST, noisePerc)
                
                #### Swapping the top % of most influential points from the training to test set
                # Get influential points in the training set using cooks distance 
                cooksDists, cooksThreshold = getCooksDistance(noiseTrain)
                # Get the influential points df
                cooksDists = pd.DataFrame(cooksDists, columns = ['distances'])
                influentialTrainDf = cooksDists[cooksDists['distances'] > cooksThreshold]
                #influentialTrainDf.sort_values(by = 'distances', ascending = False, inplace = True) ##*******Not sure if this is needed 
                
                # Now get the new training and test setafter the influential points have been swapped 
                swappedTrain, swappedXTest, swappedYTest, numSwapped = swapPercInfluenentialPoints(
                                                                                    noiseTrain, noiseXTest, noiseYTest, 
                                                                                    influentialTrainDf, swapPerc)
    
                #### Now create an object of the relevant machine learning algorithm class 
                obj = mlAlgoScriptRef.Model(dataRef.TARGET_VAR_NAME, swappedTrain, swappedXTest, swappedYTest,
                                            dataRef.XVALID, dataRef.YVALID)
                obj.createModel(nTrees) # train the model 
                
                # Get and append model perfromance measures to relevant lists
                modelTestAccuracy = obj.modelAccuracy() 
                testAccuracyAtIncrement.append(modelTestAccuracy)
                
                modelValAccuracy = obj.validAccuracy()
                valAccuracyAtIncrement.append(modelValAccuracy)
                
                modelValF1 = obj.validF1Score()
                valF1AtIncrement.append(modelValF1)
                
                modelValAUC = obj.validAUC()
                valAUCAtIncrement.append(modelValAUC)
                
                # Append number of points swapped at this increment of noise
                numPointsSwappedAtIncrement.append(numSwapped)
                
                # Free up memory by deleting object
                del obj 
            
            # Get the average value of perfromance measures and add it to relevant list
            testAccuracy.append(statistics.mean(testAccuracyAtIncrement))
            valAccuracy.append(statistics.mean(valAccuracyAtIncrement))
            valF1.append(statistics.mean(valF1AtIncrement))
            valAUC.append(statistics.mean(valAUCAtIncrement))
            numPointsSwapped.append(round(statistics.mean(numPointsSwappedAtIncrement)))
        
        
        ## Add the perfromance measures to results dictionary
        results['{}%Test'.format(swapPerc)] = testAccuracy
        results['{}%Val'.format(swapPerc)] = valAccuracy
        results['{}%ValF1'.format(swapPerc)] = valF1
        results['{}%ValAUC'.format(swapPerc)] = valAUC
        results['{}%NumSwapped'.format(swapPerc)] = numPointsSwapped
        
    
    return results


def createCooksDistNoiseMitigationPlot(wpResults, cIResults, cCDResults, mlAlgorithmName, noisePercLevels):
    
    # Get mean accuracy for each 
    mean0PercSwapTestAccuracy, mean0PercSwapValAccuracy, mean0PercSwapValF1, mean0PercSwapValAUC = [], [], [], []
    #mean20PercSwapTestAccuracy, mean20PercSwapValAccuracy, mean20PercSwapValF1, mean20PercSwapValAUC = [], [], [], []
    #mean40PercSwapTestAccuracy, mean40PercSwapValAccuracy, mean40PercSwapValF1, mean40PercSwapValAUC = [], [], [], []
    mean50PercSwapTestAccuracy, mean50PercSwapValAccuracy, mean50PercSwapValF1, mean50PercSwapValAUC = [], [], [], []
    #mean60PercSwapTestAccuracy, mean60PercSwapValAccuracy, mean60PercSwapValF1, mean60PercSwapValAUC = [], [], [], []
    #mean80PercSwapTestAccuracy, mean80PercSwapValAccuracy, mean80PercSwapValF1, mean80PercSwapValAUC = [], [], [], []
    mean100PercSwapTestAccuracy, mean100PercSwapValAccuracy, mean100PercSwapValF1, mean100PercSwapValAUC = [], [], [], []
    
    # Get mean length for each 
    #mean0PercSwapNumPointsSwapped, mean20PercSwapNumPointsSwapped, mean40PercSwapNumPointsSwapped = [], [], []
    #mean60PercSwapNumPointsSwapped, mean80PercSwapNumPointsSwapped, mean100PercSwapNumPointsSwapped  = [], [], []
    mean0PercSwapNumPointsSwapped, mean50PercSwapNumPointsSwapped, mean100PercSwapNumPointsSwapped = [], [], []

    # Get length of noise level percentages list (to ascertain how long each list in the results dicts are)
    for i in range( len(noisePercLevels) ): #len(next(iter(wpRfNoiseMitigationResults.values()))) ):
        # 0%
        mean0PercSwapTestAccuracy.append( (wpResults['0%Test'][i] + cIResults['0%Test'][i] + cCDResults['0%Test'][i])/3 )
        mean0PercSwapValAccuracy.append( (wpResults['0%Val'][i] + cIResults['0%Val'][i] + cCDResults['0%Val'][i])/3 )
        mean0PercSwapValF1.append( (wpResults['0%ValF1'][i] + cIResults['0%ValF1'][i] + cCDResults['0%ValF1'][i])/3 )
        mean0PercSwapValAUC.append( (wpResults['0%ValAUC'][i] + cIResults['0%ValAUC'][i] + cCDResults['0%ValAUC'][i])/3 )
       
        """
        # 20%
        mean20PercSwapTestAccuracy.append( (wpResults['20%Test'][i] + cIResults['20%Test'][i] + cCDResults['20%Test'][i])/3 )
        mean20PercSwapValAccuracy.append( (wpResults['20%Val'][i] + cIResults['20%Val'][i] + cCDResults['20%Val'][i])/3 )
        mean20PercSwapValF1.append( (wpResults['20%ValF1'][i] + cIResults['20%ValF1'][i] + cCDResults['20%ValF1'][i])/3 )
        mean20PercSwapValAUC.append( (wpResults['20%ValAUC'][i] + cIResults['20%ValAUC'][i] + cCDResults['20%ValAUC'][i])/3 )
       
        # 40%
        mean40PercSwapTestAccuracy.append( (wpResults['40%Test'][i] + cIResults['40%Test'][i] + cCDResults['40%Test'][i])/3 )
        mean40PercSwapValAccuracy.append( (wpResults['40%Val'][i] + cIResults['40%Val'][i] + cCDResults['40%Val'][i])/3 )
        mean40PercSwapValF1.append( (wpResults['40%ValF1'][i] + cIResults['40%ValF1'][i] + cCDResults['40%ValF1'][i])/3 )
        mean40PercSwapValAUC.append( (wpResults['40%ValAUC'][i] + cIResults['40%ValAUC'][i] + cCDResults['40%ValAUC'][i])/3 )
       
        # 60%
        mean60PercSwapTestAccuracy.append( (wpResults['60%Test'][i] + cIResults['60%Test'][i] + cCDResults['60%Test'][i])/3 )
        mean60PercSwapValAccuracy.append( (wpResults['60%Val'][i] + cIResults['60%Val'][i] + cCDResults['60%Val'][i])/3 )
        mean60PercSwapValF1.append( (wpResults['60%ValF1'][i] + cIResults['60%ValF1'][i] + cCDResults['60%ValF1'][i])/3 )
        mean60PercSwapValAUC.append( (wpResults['60%ValAUC'][i] + cIResults['60%ValAUC'][i] + cCDResults['60%ValAUC'][i])/3 )
       
        # 80%
        mean80PercSwapTestAccuracy.append( (wpResults['80%Test'][i] + cIResults['80%Test'][i] + cCDResults['80%Test'][i])/3 )
        mean80PercSwapValAccuracy.append( (wpResults['80%Val'][i] + cIResults['80%Val'][i] + cCDResults['80%Val'][i])/3 )
        mean80PercSwapValF1.append( (wpResults['80%ValF1'][i] + cIResults['80%ValF1'][i] + cCDResults['80%ValF1'][i])/3 )
        mean80PercSwapValAUC.append( (wpResults['80%ValAUC'][i] + cIResults['80%ValAUC'][i] + cCDResults['80%ValAUC'][i])/3 )
        """
        
        # 50%
        mean50PercSwapTestAccuracy.append( (wpResults['50%Test'][i] + cIResults['50%Test'][i] + cCDResults['50%Test'][i])/3 )
        mean50PercSwapValAccuracy.append( (wpResults['50%Val'][i] + cIResults['50%Val'][i] + cCDResults['50%Val'][i])/3 )
        mean50PercSwapValF1.append( (wpResults['50%ValF1'][i] + cIResults['50%ValF1'][i] + cCDResults['50%ValF1'][i])/3 )
        mean50PercSwapValAUC.append( (wpResults['50%ValAUC'][i] + cIResults['50%ValAUC'][i] + cCDResults['50%ValAUC'][i])/3 )
        
        
        # 100%
        mean100PercSwapTestAccuracy.append( (wpResults['100%Test'][i] + cIResults['100%Test'][i] + cCDResults['100%Test'][i])/3 )
        mean100PercSwapValAccuracy.append( (wpResults['100%Val'][i] + cIResults['100%Val'][i] + cCDResults['100%Val'][i])/3 )
        mean100PercSwapValF1.append( (wpResults['100%ValF1'][i] + cIResults['100%ValF1'][i] + cCDResults['100%ValF1'][i])/3 )
        mean100PercSwapValAUC.append( (wpResults['100%ValAUC'][i] + cIResults['100%ValAUC'][i] + cCDResults['100%ValAUC'][i])/3 )
        
        # number of points swapped 
        mean0PercSwapNumPointsSwapped.append( (wpResults['0%NumSwapped'][i] + cIResults['0%NumSwapped'][i] + cCDResults['0%NumSwapped'][i])/3 )
        #mean20PercSwapNumPointsSwapped.append( (wpResults['20%NumSwapped'][i] + cIResults['20%NumSwapped'][i] + cCDResults['20%NumSwapped'][i])/3 )
        #mean40PercSwapNumPointsSwapped.append( (wpResults['40%NumSwapped'][i] + cIResults['40%NumSwapped'][i] + cCDResults['40%NumSwapped'][i])/3 )
        mean50PercSwapNumPointsSwapped.append( (wpResults['50%NumSwapped'][i] + cIResults['50%NumSwapped'][i] + cCDResults['50%NumSwapped'][i])/3 )
        #mean60PercSwapNumPointsSwapped.append( (wpResults['60%NumSwapped'][i] + cIResults['60%NumSwapped'][i] + cCDResults['60%NumSwapped'][i])/3 )
        #mean80PercSwapNumPointsSwapped.append( (wpResults['80%NumSwapped'][i] + cIResults['80%NumSwapped'][i] + cCDResults['80%NumSwapped'][i])/3 )
        mean100PercSwapNumPointsSwapped.append( (wpResults['100%NumSwapped'][i] + cIResults['100%NumSwapped'][i] + cCDResults['100%NumSwapped'][i])/3 )
       
    
    #### Create Test Accuracy Plot 
    plt.figure() 
    # 0%
    plt.plot(noisePercLevels, mean0PercSwapTestAccuracy, color = 'lightgreen', ls = '-', label = "0%")
    """
    # 20%
    plt.plot(noisePercLevels, mean20PercSwapTestAccuracy, color = 'lawngreen', ls = '-', label = "20%")
    # 40% 
    plt.plot(noisePercLevels, mean40PercSwapTestAccuracy, color = 'limegreen', ls = '-', label = "40%")
    # 60%
    plt.plot(noisePercLevels, mean60PercSwapTestAccuracy, color = 'mediumseagreen', ls = '-', label = "60%")
    # 80%
    plt.plot(noisePercLevels, mean80PercSwapTestAccuracy, color = 'darkgreen', ls = '-', label = "80%")
    """
    # 50%
    plt.plot(noisePercLevels, mean50PercSwapTestAccuracy, color = 'mediumseagreen', ls = '-', label = "50%")
    # 100%
    plt.plot(noisePercLevels, mean100PercSwapTestAccuracy, color = 'black', ls = '-', label = "100%")
    
    plt.legend(title = "Influential Points Swapped")
    plt.xlabel("Noise %")
    plt.ylabel("Average Test Accuracy %")
    plt.suptitle("Using Cooks Distance to Mitigate Impact of Noise on \n{} ".format(mlAlgorithmName) + r"$\bf{Test \ Accuracy}$", fontsize=18)
    
    #### Create Val Accuracy Plot 
    plt.figure() 
    # 0%
    plt.plot(noisePercLevels, mean0PercSwapValAccuracy, color = 'lightgreen', ls = '-', label = "0%")
    """
    # 20%
    plt.plot(noisePercLevels, mean20PercSwapValAccuracy, color = 'lawngreen', ls = '-', label = "20%")
    # 40% 
    plt.plot(noisePercLevels, mean40PercSwapValAccuracy, color = 'limegreen', ls = '-', label = "40%")
    # 60%
    plt.plot(noisePercLevels, mean60PercSwapValAccuracy, color = 'mediumseagreen', ls = '-', label = "60%")
    # 80%
    plt.plot(noisePercLevels, mean80PercSwapValAccuracy, color = 'darkgreen', ls = '-', label = "80%")
    """
    # 50%
    plt.plot(noisePercLevels, mean50PercSwapValAccuracy, color = 'mediumseagreen', ls = '-', label = "50%")
    # 100%
    plt.plot(noisePercLevels, mean100PercSwapValAccuracy, color = 'black', ls = '-', label = "100%")
    
    plt.legend(title = "Influential Points Swapped")
    plt.xlabel("Noise %")
    plt.ylabel("Average Val Accuracy %")
    plt.suptitle("Using Cooks Distance to Mitigate Impact of Noise on \n{} ".format(mlAlgorithmName) + r"$\bf{Validation \ Accuracy}$", fontsize=18)
    
    
    #### Create Val F1 Score Plot 
    plt.figure() 
    # 0%
    plt.plot(noisePercLevels, mean0PercSwapValF1, color = 'lightgreen', ls = '-', label = "0%")
    """
    # 20%
    plt.plot(noisePercLevels, mean20PercSwapValF1, color = 'lawngreen', ls = '-', label = "20%")
    # 40% 
    plt.plot(noisePercLevels, mean40PercSwapValF1, color = 'limegreen', ls = '-', label = "40%")
    # 60%
    plt.plot(noisePercLevels, mean60PercSwapValF1, color = 'mediumseagreen', ls = '-', label = "60%")
    # 80%
    plt.plot(noisePercLevels, mean80PercSwapValF1, color = 'darkgreen', ls = '-', label = "80%")
    """
    # 50%
    plt.plot(noisePercLevels, mean50PercSwapValF1, color = 'mediumseagreen', ls = '-', label = "50%")
    # 100%
    plt.plot(noisePercLevels, mean100PercSwapValF1, color = 'black', ls = '-', label = "100%")
    
    plt.legend(title = "Influential Points Swapped")
    plt.xlabel("Noise %")
    plt.ylabel("Average F1 Score")
    plt.suptitle("Using Cooks Distance to Mitigate Impact of Noise on \n{} ".format(mlAlgorithmName) + r"$\bf{Validation \ F1 \ Score}$", fontsize=18)
    
    
    #### Create Val AUC Plot 
    plt.figure() 
    # 0%
    plt.plot(noisePercLevels, mean0PercSwapValAUC, color = 'lightgreen', ls = '-', label = "0%")
    """
    # 20%
    plt.plot(noisePercLevels, mean20PercSwapValAUC, color = 'lawngreen', ls = '-', label = "20%")
    # 40% 
    plt.plot(noisePercLevels, mean40PercSwapValAUC, color = 'limegreen', ls = '-', label = "40%")
    # 60%
    plt.plot(noisePercLevels, mean60PercSwapValAUC, color = 'mediumseagreen', ls = '-', label = "60%")
    # 80%
    plt.plot(noisePercLevels, mean80PercSwapValAUC, color = 'darkgreen', ls = '-', label = "80%")
    """
    # 50%
    plt.plot(noisePercLevels, mean50PercSwapValAUC, color = 'mediumseagreen', ls = '-', label = "50%")
    # 100%
    plt.plot(noisePercLevels, mean100PercSwapValAUC, color = 'black', ls = '-', label = "100%")
    
    plt.legend(title = "Influential Points Swapped")
    plt.xlabel("Noise %")
    plt.ylabel("Average AUC Score")
    plt.suptitle("Using Cooks Distance to Mitigate Impact of Noise on \n{} ".format(mlAlgorithmName) + r"$\bf{Validation \ AUC}$", fontsize=18)
    
    
    #### Create histogram of number of influential points swapped 
    plt.figure() 
    xPositions = np.arange(len(mean0PercSwapNumPointsSwapped))
    # 0% (Intuitively this will be 0, so it will appear like its missing)
    #plt.bar(xPositions + 0.00, mean0PercSwapNumPointsSwapped, width = 0.25, color = 'lightgreen', ls = '-', label = "0%")
    """
    # 20%
    plt.bar(mean20PercSwapNumPointsSwapped, color = 'lawngreen', ls = '-', label = "20%")
    # 40% 
    plt.bar(mean40PercSwapNumPointsSwapped, color = 'limegreen', ls = '-', label = "40%")
    # 60%
    plt.bar(mean60PercSwapNumPointsSwapped, color = 'mediumseagreen', ls = '-', label = "60%")
    # 80%
    plt.bar(mean80PercSwapNumPointsSwapped, color = 'darkgreen', ls = '-', label = "80%")
    """
    # 50%
    plt.bar(xPositions + 0.25, mean50PercSwapNumPointsSwapped, width = 0.25, color = 'mediumseagreen', ls = '-', label = "50%")
    # 100%
    plt.bar(xPositions + 0.50, mean100PercSwapNumPointsSwapped, width = 0.25, color = 'black', ls = '-', label = "100%")
    
    #plt.axis(noisePercLevels)
    plt.xticks(ticks = xPositions, labels = noisePercLevels)
    plt.legend(title = "Influential Points Swapped")
    plt.xlabel("Noise %")
    plt.ylabel("Number of points swapped")
    plt.suptitle("Number of Influential Points Swapped from Training to Test set", fontsize=18)
   


################# Get the results of mitigation of noise effect using cooks distance  
##### Constants
## Create the noiseLevelPerc list for the % of noise to insert and for subsequent graphing purposes.
NOISEPERCLEVELSMITIGATIONEXP = list(range(0, 51, 2)) # 0,2,4,....,50
# To change simply change range in the format range(startAt, stopBefore, incrementBy)


##### Rf
## Waterpump dataset
wpRfNoiseMitigationResults = influentialPointEffect(rf, wpData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)
## Census Income
cIRfNoiseMitigationResults = influentialPointEffect(rf, cIData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)
## Credit Card Default dataset 
cCDRfNoiseMitigationResults = influentialPointEffect(rf, cCDData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)

#### xgb   
## waterpump
wpXgbNoiseMitigationResults = influentialPointEffect(xgb, wpData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)
## Census Income
cIXgbNoiseMitigationResults = influentialPointEffect(xgb, cIData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)
## Credit Card Default 
cCDXgbNoiseMitigationResults = influentialPointEffect(xgb, cCDData, NOISEPERCLEVELSMITIGATIONEXP, 100) # Use default nTrees (=100)

#### svm
## waterpump
wpSvmNoiseMitigationResults = influentialPointEffect(svm, wpData, NOISEPERCLEVELSMITIGATIONEXP)
## Census Income 
cISvmNoiseMitigationResults = influentialPointEffect(svm, cIData, NOISEPERCLEVELSMITIGATIONEXP)
## Credit Card Default
cCDSvmNoiseMitigationResults = influentialPointEffect(svm, cCDData, NOISEPERCLEVELSMITIGATIONEXP)


#### dt 
wpDtNoiseMitigationResults = influentialPointEffect(dt, wpData, NOISEPERCLEVELSMITIGATIONEXP)
## Census Income dataset 
cIDtNoiseMitigationResults = influentialPointEffect(dt, cIData, NOISEPERCLEVELSMITIGATIONEXP)
## Credit Card Default dataset 
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





###### Additional sub experiement with Cooks Distance 
# examining if swapping the 5% most influential points from the training set when 5% noise has been inserted increases accuracy
def cooksDistNoiseMitigationExp2(mlAlgoScriptRef, dataRef, noisePercLevels, nTrees = 100):
    # Create dict to store results 
    results = {}
        
    # Create lists to store the results of models trained with original training data 
    # (not swapping any points from the training to the test)
    noSwapTestAccuracy, noSwapValAccuracy, noSwapValF1, noSwapValAUC = [], [], [], []
    # and results of models trained with data that has been altered (swapping points from training to test)
    testAccuracy, valAccuracy, valF1, valAUC = [], [], [], []
    numPointsSwapped = []

    
    for noisePerc in noisePercLevels:
        # Repeat 10 times and get the average accuracy 
        noSwapTestAccuracyAtIncrement, noSwapValAccuracyAtIncrement = [], []
        noSwapValF1AtIncrement, noSwapValAUCAtIncrement = [], []
        
        testAccuracyAtIncrement, valAccuracyAtIncrement = [], []
        valF1AtIncrement, valAUCAtIncrement = [], []
        numPointsSwappedAtIncrement = []
        
        for i in range(5): # Get average of only 5 models to speed up system
           
            # Insert % noise into the test and training sets
            noiseTrain = insertingNoise(dataRef.TRAIN, noisePerc)
            noiseXTest, noiseYTest = insertingNoiseTestSet(dataRef.XTEST, dataRef.YTEST, noisePerc)
            
            #####  Train basic model with no influential points swapped  
            obj = mlAlgoScriptRef.Model(dataRef.TARGET_VAR_NAME, noiseTrain, noiseXTest, noiseYTest,
                                        dataRef.XVALID, dataRef.YVALID)
            obj.createModel(nTrees) # train the model 
            
            # Get and append model perfromance measures to relevant lists
            noSwapModelTestAccuracy = obj.modelAccuracy() 
            noSwapTestAccuracyAtIncrement.append(noSwapModelTestAccuracy)
            
            noSwapModelValAccuracy = obj.validAccuracy()
            noSwapValAccuracyAtIncrement.append(noSwapModelValAccuracy)
            
            noSwapModelValF1 = obj.validF1Score()
            noSwapValF1AtIncrement.append(noSwapModelValF1)
            
            noSwapModelValAUC = obj.validAUC()
            noSwapValAUCAtIncrement.append(noSwapModelValAUC)
            
            # Free up memory by deleting object
            del obj 
            
            
            #### Swapping the top x% of most influential points from the training to test set, 
            # where x is the noise perc level that has been inserted into the training set 
            
            # Get influential points in the training set using cooks distance 
            cooksDists, cooksThreshold = getCooksDistance(noiseTrain)
            # Get the influential points df
            cooksDists = pd.DataFrame(cooksDists, columns = ['distances'])
            cooksDists.sort_values(by = 'distances', ascending = False, inplace = True) 
            
            # Concat the xTest and yTest together
            test = pd.concat([noiseXTest, noiseYTest], axis = 1)
            
            # get number of obs to change from the training set to the test
            numObs = len(noiseTrain) 
            numToChange = round(numObs * noisePerc/100)
            # Select the points to change (points are already in order of cooks distance) 
            toChangeDists = cooksDists[ :numToChange]
            
            ### Remove the observations to change from the training set
            toChangeIndexes = toChangeDists.index # get index of observations to change             
            # Get observations with these indexes from the training set
            mostInfluentialPointsToSwap = noiseTrain.loc[toChangeIndexes, : ]                      
            # Remove observations with these indexes from the training set
            swappedTrain = noiseTrain.drop(toChangeIndexes)
            
            ### Now add the most influential points df to the test set 
            swappedTest = pd.concat([test, mostInfluentialPointsToSwap], axis = 0)  
            ### Split back into xTest and yTest
            swappedXTest, swappedYTest = swappedTest.iloc[:, :-1], swappedTest.iloc[:, -1]
            
            
            #### Now get results of a model trained using the now altered train and test sets
            obj = mlAlgoScriptRef.Model(dataRef.TARGET_VAR_NAME, swappedTrain, swappedXTest, swappedYTest,
                                        dataRef.XVALID, dataRef.YVALID)
            obj.createModel(nTrees) # train the model 
            
            # Get and append model perfromance measures to relevant lists
            modelTestAccuracy = obj.modelAccuracy() 
            testAccuracyAtIncrement.append(modelTestAccuracy)
            
            modelValAccuracy = obj.validAccuracy()
            valAccuracyAtIncrement.append(modelValAccuracy)
            
            modelValF1 = obj.validF1Score()
            valF1AtIncrement.append(modelValF1)
            
            modelValAUC = obj.validAUC()
            valAUCAtIncrement.append(modelValAUC)
            
            # Append number of points swapped at this increment of noise
            numPointsSwappedAtIncrement.append(len(mostInfluentialPointsToSwap))
            
            # Free up memory by deleting object
            del obj 
        
        # Get the average value of perfromance measures and add it to relevant list
        # Noise only perfromance results 
        noSwapTestAccuracy.append(statistics.mean(noSwapTestAccuracyAtIncrement))
        noSwapValAccuracy.append(statistics.mean(noSwapValAccuracyAtIncrement))
        noSwapValF1.append(statistics.mean(noSwapValF1AtIncrement))
        noSwapValAUC.append(statistics.mean(noSwapValAUCAtIncrement))
       
        # Influential piints swapped results
        testAccuracy.append(statistics.mean(testAccuracyAtIncrement))
        valAccuracy.append(statistics.mean(valAccuracyAtIncrement))
        valF1.append(statistics.mean(valF1AtIncrement))
        valAUC.append(statistics.mean(valAUCAtIncrement))
        numPointsSwapped.append(round(statistics.mean(numPointsSwappedAtIncrement)))
    
    
    ## Add the perfromance measures to results dictionary
    # The noise only perfroamnce reuslts
    results['Test'] = noSwapTestAccuracy
    results['Val'] = noSwapValAccuracy
    results['ValF1'] = noSwapValF1
    results['ValAUC'] = noSwapValAUC
    
    # The noise, and swapped influential points from train to test perfromance results
    results['IPSTest'] = testAccuracy
    results['IPSVal'] = valAccuracy
    results['IPSValF1'] = valF1
    results['IPSValAUC'] = valAUC
    results['NumSwapped'] = numPointsSwapped
    
    return results
    
def mergeExp2Results(wpResults, cIResults, cCDResults, noisePercLevels):
    IPSTest, IPSVal, IPSValAUC, IPSValF1 = [], [], [], []
    test, val, valAUC, valF1 = [], [], [], []
    
    results = {}
    # Get length of noise level percentages list (to ascertain how long each list in the results dicts are)
    for i in range( len(noisePercLevels) ): #len(next(iter(wpRfNoiseMitigationResults.values()))) ):
        # Influential Points Swapped (IPS)
        IPSTest.append( round((wpResults['IPSTest'][i] + cIResults['IPSTest'][i] + cCDResults['IPSTest'][i])/3, 3) )
        IPSVal.append( round((wpResults['IPSVal'][i] + cIResults['IPSVal'][i] + cCDResults['IPSVal'][i])/3, 3) )
        IPSValF1.append( round((wpResults['IPSValF1'][i] + cIResults['IPSValF1'][i] + cCDResults['IPSValF1'][i])/3, 3) )
        IPSValAUC.append( round((wpResults['IPSValAUC'][i] + cIResults['IPSValAUC'][i] + cCDResults['IPSValAUC'][i])/3, 3) )
    
        # Just noise
        test.append( round((wpResults['Test'][i] + cIResults['Test'][i] + cCDResults['Test'][i])/3, 3) )
        val.append( round((wpResults['Val'][i] + cIResults['Val'][i] + cCDResults['Val'][i])/3, 3) )
        valF1.append( round((wpResults['ValF1'][i] + cIResults['ValF1'][i] + cCDResults['ValF1'][i])/3, 3) )
        valAUC.append( round((wpResults['ValAUC'][i] + cIResults['ValAUC'][i] + cCDResults['ValAUC'][i])/3, 3) )
         
    ## Add the perfromance measures to results dictionary
    # The noise only perfroamnce reuslts
    results['Test'] = test
    results['Val'] = val
    results['ValF1'] = valF1
    results['ValAUC'] = valAUC
    
    # The noise, and swapped influential points from train to test perfromance results
    results['IPSTest'] = IPSTest
    results['IPSVal'] = IPSVal
    results['IPSValF1'] = IPSValF1
    results['IPSValAUC'] = IPSValAUC
    
    # Add number of Points swapped for each
    results['NumSwappedWP'] = wpResults['NumSwapped']
    results['NumSwappedCI'] = cIResults['NumSwapped']
    results['NumSwappedCCD'] = cCDResults['NumSwapped']
    
    return results 
        
## Constant 
NOISEPERCLEVELSMITIGATIONEXP2 = [0,1,2,3,4,5]

##### Rf
## Waterpump dataset
wpRfCooksDistSubExp = cooksDistNoiseMitigationExp2(rf, wpData, NOISEPERCLEVELSMITIGATIONEXP2, 100) # Use default nTrees (=100)
## Census Income
cIRfCooksDistSubExp = cooksDistNoiseMitigationExp2(rf, cIData, NOISEPERCLEVELSMITIGATIONEXP2, 100) # Use default nTrees (=100)
## Credit Card Default dataset 
cCDRfCooksDistSubExp = cooksDistNoiseMitigationExp2(rf, cCDData, NOISEPERCLEVELSMITIGATIONEXP2, 100) # Use default nTrees (=100)

##### XGB
## Waterpump dataset
wpXgbCooksDistSubExp = cooksDistNoiseMitigationExp2(xgb, wpData, NOISEPERCLEVELSMITIGATIONEXP2, 100) # Use default nTrees (=100)
## Census Income
cIXgbCooksDistSubExp = cooksDistNoiseMitigationExp2(xgb, cIData, NOISEPERCLEVELSMITIGATIONEXP2, 100) # Use default nTrees (=100)
## Credit Card Default dataset 
cCDXgbCooksDistSubExp = cooksDistNoiseMitigationExp2(xgb, cCDData, NOISEPERCLEVELSMITIGATIONEXP2, 100) # Use default nTrees (=100)

##### SVM
## Waterpump dataset
wpSvmCooksDistSubExp = cooksDistNoiseMitigationExp2(svm, wpData, NOISEPERCLEVELSMITIGATIONEXP2)
## Census Income
cISvmCooksDistSubExp = cooksDistNoiseMitigationExp2(svm, cIData, NOISEPERCLEVELSMITIGATIONEXP2)
## Credit Card Default dataset 
cCDSvmCooksDistSubExp = cooksDistNoiseMitigationExp2(svm, cCDData, NOISEPERCLEVELSMITIGATIONEXP2)

##### Decision Trees
## Waterpump dataset
wpDtCooksDistSubExp = cooksDistNoiseMitigationExp2(dt, wpData, NOISEPERCLEVELSMITIGATIONEXP2)
## Census Income
cIDtCooksDistSubExp = cooksDistNoiseMitigationExp2(dt, cIData, NOISEPERCLEVELSMITIGATIONEXP2)
## Credit Card Default dataset 
cCDDtCooksDistSubExp = cooksDistNoiseMitigationExp2(dt, cCDData, NOISEPERCLEVELSMITIGATIONEXP2)


### Merge results 
rfCooksDistSubExpResults = mergeExp2Results(wpRfCooksDistSubExp, cIRfCooksDistSubExp, cCDRfCooksDistSubExp, NOISEPERCLEVELSMITIGATIONEXP2)
xgbCooksDistSubExpResults = mergeExp2Results(wpXgbCooksDistSubExp, cIXgbCooksDistSubExp, cCDXgbCooksDistSubExp, NOISEPERCLEVELSMITIGATIONEXP2)
svmCooksDistSubExpResults = mergeExp2Results(wpSvmCooksDistSubExp, cISvmCooksDistSubExp, cCDSvmCooksDistSubExp, NOISEPERCLEVELSMITIGATIONEXP2)
dtCooksDistSubExpResults = mergeExp2Results(wpDtCooksDistSubExp, cIDtCooksDistSubExp, cCDDtCooksDistSubExp, NOISEPERCLEVELSMITIGATIONEXP2)



############################ Test if the number of trees provides insulation against noise ######################
def createTreesNoiseInsulationPlot(wp20TResults, wp60TResults, wp100TResults,
                                   wp200TResults, wp500TResults,
                                   cI20TResults, cI60TResults, cI100TResults,
                                   cI200TResults, cI500TResults,
                                   cCD20TResults, cCD60TResults, cCD100TResults,
                                   cCD200TResults, cCD500TResults,
                                   noisePercLevels, mlAlgoName):
    
    
    # Get mean accuracy for each 
    mean20TreeTestAccuracy,  mean20TreeValAccuracy, mean20TreeValF1, mean20TreeValAUC = [], [], [], []
    mean60TreeTestAccuracy,  mean60TreeValAccuracy, mean60TreeValF1, mean60TreeValAUC = [], [], [], []
    mean100TreeTestAccuracy, mean100TreeValAccuracy, mean100TreeValF1, mean100TreeValAUC = [], [], [], []
    mean200TreeTestAccuracy, mean200TreeValAccuracy, mean200TreeValF1, mean200TreeValAUC = [], [], [], []
    mean500TreeTestAccuracy, mean500TreeValAccuracy, mean500TreeValF1, mean500TreeValAUC = [], [], [], []
   

    # Get length of noise level percentages list (to ascertain how long each list in the results dicts are)
    for i in range( len(noisePercLevels) ): 
        # 20 Trees
        mean20TreeTestAccuracy.append( (wp20TResults['TestAccuracy'][i] + cI20TResults['TestAccuracy'][i] + cCD20TResults['TestAccuracy'][i])/3 )
        mean20TreeValAccuracy.append( (wp20TResults['ValAccuracy'][i] + cI20TResults['ValAccuracy'][i] + cCD20TResults['ValAccuracy'][i])/3 )
        mean20TreeValF1.append( (wp20TResults['ValF1'][i] + cI20TResults['ValF1'][i] + cCD20TResults['ValF1'][i])/3 )
        mean20TreeValAUC.append( (wp20TResults['ValAUC'][i] + cI20TResults['ValAUC'][i] + cCD20TResults['ValAUC'][i])/3 )
        # 60 Trees   
        mean60TreeTestAccuracy.append( (wp60TResults['TestAccuracy'][i] + cI60TResults['TestAccuracy'][i] + cCD60TResults['TestAccuracy'][i])/3 )
        mean60TreeValAccuracy.append( (wp60TResults['ValAccuracy'][i] + cI60TResults['ValAccuracy'][i] + cCD60TResults['ValAccuracy'][i])/3 )
        mean60TreeValF1.append( (wp60TResults['ValF1'][i] + cI60TResults['ValF1'][i] + cCD60TResults['ValF1'][i])/3 )
        mean60TreeValAUC.append( (wp60TResults['ValAUC'][i] + cI60TResults['ValAUC'][i] + cCD60TResults['ValAUC'][i])/3 )
        # 100 Trees
        mean100TreeTestAccuracy.append( (wp100TResults['TestAccuracy'][i] + cI100TResults['TestAccuracy'][i] + cCD100TResults['TestAccuracy'][i])/3 )
        mean100TreeValAccuracy.append( (wp100TResults['ValAccuracy'][i] + cI100TResults['ValAccuracy'][i] + cCD100TResults['ValAccuracy'][i])/3 )
        mean100TreeValF1.append( (wp100TResults['ValF1'][i] + cI100TResults['ValF1'][i] + cCD100TResults['ValF1'][i])/3 )
        mean100TreeValAUC.append( (wp100TResults['ValAUC'][i] + cI100TResults['ValAUC'][i] + cCD100TResults['ValAUC'][i])/3 )
        # 200 Trees
        mean200TreeTestAccuracy.append( (wp200TResults['TestAccuracy'][i] + cI200TResults['TestAccuracy'][i] + cCD200TResults['TestAccuracy'][i])/3 )
        mean200TreeValAccuracy.append( (wp200TResults['ValAccuracy'][i] + cI200TResults['ValAccuracy'][i] + cCD200TResults['ValAccuracy'][i])/3 )
        mean200TreeValF1.append( (wp200TResults['ValF1'][i] + cI200TResults['ValF1'][i] + cCD200TResults['ValF1'][i])/3 )
        mean200TreeValAUC.append( (wp200TResults['ValAUC'][i] + cI200TResults['ValAUC'][i] + cCD200TResults['ValAUC'][i])/3 )
        # 500 Trees
        mean500TreeTestAccuracy.append( (wp500TResults['TestAccuracy'][i] + cI500TResults['TestAccuracy'][i] + cCD500TResults['TestAccuracy'][i])/3 )
        mean500TreeValAccuracy.append( (wp500TResults['ValAccuracy'][i] + cI500TResults['ValAccuracy'][i] + cCD500TResults['ValAccuracy'][i])/3 )
        mean500TreeValF1.append( (wp500TResults['ValF1'][i] + cI500TResults['ValF1'][i] + cCD500TResults['ValF1'][i])/3 )
        mean500TreeValAUC.append( (wp500TResults['ValAUC'][i] + cI500TResults['ValAUC'][i] + cCD500TResults['ValAUC'][i])/3 )
        
    
    #### Test Accuracy Plot 
    plt.figure() # Instantiate a new figure 

    # 20 Trees
    plt.plot(noisePercLevels, mean20TreeTestAccuracy, color = 'lightgreen', ls = '-', label = "20")
    # 60 Trees
    plt.plot(noisePercLevels, mean60TreeTestAccuracy, color = 'lawngreen', ls = '-', label = "60")
    # 100 Trees 
    plt.plot(noisePercLevels, mean100TreeTestAccuracy, color = 'limegreen', ls = '-', label = "100")
    # 200 Trees
    plt.plot(noisePercLevels, mean200TreeTestAccuracy, color = 'mediumseagreen', ls = '-', label = "200")
    # 500 Trees
    plt.plot(noisePercLevels, mean500TreeTestAccuracy, color = 'darkgreen', ls = '-', label = "500")
    
    plt.legend(title = "Number of Trees")
    plt.xlabel("Noise %")
    plt.ylabel("Average Test Accuracy %")
    plt.suptitle("Tree Insualtion against Noise Impact on {} ".format(mlAlgoName) + r"$\bf{Test \ Accuracy}$", fontsize=18)


    #### Val Accuracy Plot 
    plt.figure() # Instantiate a new figure 

    # 20 Trees
    plt.plot(noisePercLevels, mean20TreeValAccuracy, color = 'lightgreen', ls = '-', label = "20")
    # 60 Trees
    plt.plot(noisePercLevels, mean60TreeValAccuracy, color = 'lawngreen', ls = '-', label = "60")
    # 100 Trees 
    plt.plot(noisePercLevels, mean100TreeValAccuracy, color = 'limegreen', ls = '-', label = "100")
    # 200 Trees
    plt.plot(noisePercLevels, mean200TreeValAccuracy, color = 'mediumseagreen', ls = '-', label = "200")
    # 500 Trees
    plt.plot(noisePercLevels, mean500TreeValAccuracy, color = 'darkgreen', ls = '-', label = "500")
    
    plt.legend(title = "Number of Trees")
    plt.xlabel("Noise %")
    plt.ylabel("Average Val Accuracy %")
    plt.suptitle("Tree Insualtion against Noise Impact on {} ".format(mlAlgoName) + r"$\bf{Validation \ Accuracy}$", fontsize=18)


    #### Validation F1 Score Plot 
    plt.figure() # Instantiate a new figure 

    # 20 Trees
    plt.plot(noisePercLevels, mean20TreeValF1, color = 'lightgreen', ls = '-', label = "20")
    # 60 Trees
    plt.plot(noisePercLevels, mean60TreeValF1, color = 'lawngreen', ls = '-', label = "60")
    # 100 Trees 
    plt.plot(noisePercLevels, mean100TreeValF1, color = 'limegreen', ls = '-', label = "100")
    # 200 Trees
    plt.plot(noisePercLevels, mean200TreeValF1, color = 'mediumseagreen', ls = '-', label = "200")
    # 500 Trees
    plt.plot(noisePercLevels, mean500TreeValF1, color = 'darkgreen', ls = '-', label = "500")
    
    plt.legend(title = "Number of Trees")
    plt.xlabel("Noise %")
    plt.ylabel("Average Test Accuracy %")
    plt.suptitle("Tree Insualtion against Noise Impact on {} ".format(mlAlgoName) + r"$\bf{Validation \ F1 \ Score}$", fontsize=18)


    #### Validation AUC Value Plot 
    plt.figure() # Instantiate a new figure 

    # 20 Trees
    plt.plot(noisePercLevels, mean20TreeValAUC, color = 'lightgreen', ls = '-', label = "20")
    # 60 Trees
    plt.plot(noisePercLevels, mean60TreeValAUC, color = 'lawngreen', ls = '-', label = "60")
    # 100 Trees 
    plt.plot(noisePercLevels, mean100TreeValAUC, color = 'limegreen', ls = '-', label = "100")
    # 200 Trees
    plt.plot(noisePercLevels, mean200TreeValAUC, color = 'mediumseagreen', ls = '-', label = "200")
    # 500 Trees
    plt.plot(noisePercLevels, mean500TreeValAUC, color = 'darkgreen', ls = '-', label = "500")
    
    plt.legend(title = "Number of Trees")
    plt.xlabel("Noise %")
    plt.ylabel("Average Test Accuracy %")
    plt.suptitle("Tree Insualtion against Noise Impact on {} ".format(mlAlgoName) + r"$\bf{Validation \ AUC}$", fontsize=18)


############### Get Accuracys to see if increasing the number of trees provides insulation agasints impact of noise ###################
##### Constants
## Create the noiseLevelPerc list for the % of noise to insert and for subsequent graphing purposes.
NOISEPERCLEVELSMITIGATIONEXP = list(range(0, 51, 1)) # 0,1,2,....,50
# To change simply change range in the format range(startAt, stopBefore, incrementBy)


######## Random Forest
#### Waterpump data
## 20 trees
wpRf20TResults = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 20)
## 60 trees
wpRf60TResults = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 60)
## 100 trees
# This is the deafult which has already been run. Results stored in vars: wpRfTestAccuracy, wpRfValAccuracy
# wpRfTest100T, wpRfVal100T = wpRfTestAccuracy, wpRfValAccuracy
wpRf100TResults = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 100) # alternatively run code again 
## 200 trees
wpRf200TResults = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 200)
## 500 trees
wpRf500TResults = rfNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 500)

#### Census Income data
## 20 trees
cIRf20TResults = rfNoiseEffect(cIData, NOISEPERCLEVELSMITIGATIONEXP, 20)
## 60 trees
cIRf60TResults = rfNoiseEffect(cIData, NOISEPERCLEVELSMITIGATIONEXP, 60)
## 100 trees
# This is the deafult which has already been run. Results stored in vars: cIRfTestAccuracy, cIRfValAccuracy
# cIRfTest100T, cIRfVal100T = cIRfTestAccuracy, cIRfValAccuracy
cIRf100TResults = rfNoiseEffect(cIData, NOISEPERCLEVELSMITIGATIONEXP, 100) # alternatively run code again 
## 200 trees
cIRf200TResults = rfNoiseEffect(cIData, NOISEPERCLEVELSMITIGATIONEXP, 200)
## 500 trees
cIRf500TResults = rfNoiseEffect(cIData, NOISEPERCLEVELSMITIGATIONEXP, 500)

#### Credit Card Default Dataset
## 20 trees
cCDRf20TResults = rfNoiseEffect(cCDData, NOISEPERCLEVELSMITIGATIONEXP, 20)
## 60 trees
cCDRf60TResults = rfNoiseEffect(cCDData, NOISEPERCLEVELSMITIGATIONEXP, 60)
## 100 trees
# This is the deafult which has already been run. Results stored in vars: cCDRfTestAccuracy, cCDRfValAccuracy
# cCDRfTest100T, cCDRfVal100T = cCDRfTestAccuracy, cCDRfValAccuracy
cCDRf100TResults = rfNoiseEffect(cCDData, NOISEPERCLEVELSMITIGATIONEXP, 100) # alternatively run code again 
## 200 trees
cCDRf200TResults = rfNoiseEffect(cCDData, NOISEPERCLEVELSMITIGATIONEXP, 200)
## 500 trees
cCDRf500TResults = rfNoiseEffect(cCDData, NOISEPERCLEVELSMITIGATIONEXP, 500)


## Generate plot
createTreesNoiseInsulationPlot(wpRf20TResults, wpRf60TResults, wpRf100TResults,
                               wpRf200TResults, wpRf500TResults,
                               cIRf20TResults, cIRf60TResults, cIRf100TResults,
                               cIRf200TResults, cIRf500TResults,
                               cCDRf20TResults, cCDRf60TResults, cCDRf100TResults,
                               cCDRf200TResults, cCDRf500TResults,
                               NOISEPERCLEVELSMITIGATIONEXP, "Random Forest")


######## XGBoost
#### Waterpump data
## 20 trees
wpXgb20TResults = xgbNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 20)
## 60 trees
wpXgb60TResults = xgbNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 60)
## 100 trees
# This is the deafult which has already been run. Results stored in vars: wpXgbTestAccuracy, wpXgbValAccuracy
# wpXgbTest100T, wpXgbVal100T = wpXgbTestAccuracy, wpXgbValAccuracy
wpXgb100TResults = xgbNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 100) # alternatively run code again 
## 200 trees
wpXgb200TResults = xgbNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 200)
## 500 trees
wpXgb500TResults = xgbNoiseEffect(wpData, NOISEPERCLEVELSMITIGATIONEXP, 500)

#### Census Income data
## 20 trees
cIXgb20TResults = xgbNoiseEffect(cIData, NOISEPERCLEVELSMITIGATIONEXP, 20)
## 60 trees
cIXgb60TResults = xgbNoiseEffect(cIData, NOISEPERCLEVELSMITIGATIONEXP, 60)
## 100 trees
# This is the deafult which has already been run. Results stored in vars: cIXgbTestAccuracy, cIXgbValAccuracy
# cIXgbTest100T, cIXgbVal100T = cIXgbTestAccuracy, cIXgbValAccuracy
cIXgb100TResults = xgbNoiseEffect(cIData, NOISEPERCLEVELSMITIGATIONEXP, 100) # alternatively run code again 
## 200 trees
cIXgb200TResults = xgbNoiseEffect(cIData, NOISEPERCLEVELSMITIGATIONEXP, 200)
## 500 trees
cIXgb500TResults = xgbNoiseEffect(cIData, NOISEPERCLEVELSMITIGATIONEXP, 500)

#### Credit Card Default Dataset
## 20 trees
cCDXgb20TResults = xgbNoiseEffect(cCDData, NOISEPERCLEVELSMITIGATIONEXP, 20)
## 60 trees
cCDXgb60TResults = xgbNoiseEffect(cCDData, NOISEPERCLEVELSMITIGATIONEXP, 60)
## 100 trees
# This is the deafult which has already been run. Results stored in vars: cCDXgbTestAccuracy, cCDXgbValAccuracy
# cCDXgbTest100T, cCDXgbVal100T = cCDXgbTestAccuracy, cCDXgbValAccuracy
cCDXgb100TResults = xgbNoiseEffect(cCDData, NOISEPERCLEVELSMITIGATIONEXP, 100) # alternatively run code again 
## 200 trees
cCDXgb200TResults = xgbNoiseEffect(cCDData, NOISEPERCLEVELSMITIGATIONEXP, 200)
## 500 trees
cCDXgb500TResults = xgbNoiseEffect(cCDData, NOISEPERCLEVELSMITIGATIONEXP, 500)


## Generate plot
createTreesNoiseInsulationPlot(wpXgb20TResults, wpXgb60TResults, wpXgb100TResults,
                               wpXgb200TResults, wpXgb500TResults,
                               cIXgb20TResults, cIXgb60TResults, cIXgb100TResults,
                               cIXgb200TResults, cIXgb500TResults,
                               cCDXgb20TResults, cCDXgb60TResults, cCDXgb100TResults,
                               cCDXgb200TResults, cCDXgb500TResults,
                               NOISEPERCLEVELSMITIGATIONEXP, "XGBoost")

