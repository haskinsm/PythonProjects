# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 13:28:41 2021

@author: micha

description: This where the already saved processed dataset will be read in and split up into training, test and validation sets. 
            The data has already been prepped for use in machine learning algortihms in water_pump_dataset_manip and has been saved as
            processedData.csv in data/datasets/water_pump_data/processed_data.csv
"""

import pandas as pd 
import numpy as np
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt

class WaterPump():
    ############################ Read in data #################################
    #get the current working directory 
    currentDir = os.getcwd()
   
    # Append name of file to be read to the current directory 
    testFilesDir = os.path.join(currentDir, "scripts_and_data\\data\\datasets\\water_pump_data\\processed_data.csv")
    test = pd.read_csv(testFilesDir)
    
