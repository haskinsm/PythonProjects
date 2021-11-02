# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:02:43 2021

@author: micha
"""
## import scripts I've written
import scripts_and_data
import os

# Make sure youre in the right current working directory should be  
# C:\Users\micha\Documents\3rd year\Software Applications\PythonSpyder(Anaconda V)\final_year_project_optum
currentDir = os.getcwd()

rf = scripts_and_data.scripts.random_forest
data = scripts_and_data.data.titanic
x = rf.RandomForest(data.TARGET_VAR_NAME, data.train, data.test)

data.TARGET_VAR_NAME()

import scripts
scripts.random_forest
dir(scripts)



