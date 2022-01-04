# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 13:28:41 2021

@author: micha

description: This where the already saved processed dataset will be read in and split up into training, test and validation sets. 
            The data has already been prepped for use in machine learning algortihms in water_pump_dataset_manip and has been saved as
            processedData.csv in data/datasets/water_pump_data/processed_data.csv
"""

import pandas as pd 
import os
from sklearn.model_selection import train_test_split

class WaterPump():
    ############################ Read in data #################################
    #get the current working directory 
    currentDir = os.getcwd()
   
    # Append name of file to be read to the current directory 
    dataFilesDir = os.path.join(currentDir, "scripts_and_data\\data\\datasets\\water_pump_data\\processed_data.csv")
    data = pd.read_csv(dataFilesDir, index_col = 'id')
    
    ########################### Split into training, test and validation sets ################
    # First split into training set and then will split remaining terms again to get the test and validation sets
    xTrain, xRem, yTrain, yRem = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], train_size = 0.6)
    xValid, xTest, yValid, yTest = train_test_split(xRem, yRem, test_size = 0.5) #Test = 20% and validation = 20%

    
    ###################### Set vars that will be accessed later as constants #######################
    TARGET_VAR_NAME = "status_group" # name of target variable column
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
    
