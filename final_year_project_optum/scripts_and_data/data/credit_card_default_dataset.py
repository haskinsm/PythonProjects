# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:52:43 2022

@author: micha
"""

import pandas as pd 
import os
from sklearn.model_selection import train_test_split

class CreditCardDefault():
    ############################ Read in data #################################
    #get the current working directory 
    currentDir = os.getcwd()
   
    # Append name of file to be read to the current directory 
    dataFilesDir = os.path.join(currentDir, "scripts_and_data\\data\\datasets\\credit_card_default_data\\processed_data.csv")
    data = pd.read_csv(dataFilesDir)
    
    ########################### Split into training, test and validation sets ################
    # First split into training set and then will split remaining terms again to get the test and validation sets
    xTrain, xRem, yTrain, yRem = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], train_size = 0.6)
    xValid, xTest, yValid, yTest = train_test_split(xRem, yRem, test_size = 0.5) #Test = 20% and validation = 20%

    
    ###################### Set vars that will be accessed later as constants #######################
    TARGET_VAR_NAME = "default" # name of target variable column
    TRAIN = pd.concat([xTrain, yTrain],  axis = 1)
    XTEST = xTest 
    YTEST = yTest
    XVALID = xValid
    YVALID = yValid
    #FULLDATASET = fullDataset
    
    ##################### Use del to delete local vairables from memory ###########################
    # Need to do this as my laptop does not have enough 
    del xTrain
    del yTrain
    del xTest
    del yTest
    del xValid
    del yValid
    del data 