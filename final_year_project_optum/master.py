# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:02:43 2021

@author: micha
"""
import os

# Change directory to correctDir or script wont run correctly
correctDir = "C:\\Users\\micha\\Documents\\3rd year\\Software Applications\\PythonSpyder(Anaconda V)\\final_year_project_optum"
os.chdir(correctDir)



##### Import scripts I've written
import scripts_and_data
# Create shorter references for these classes
rf = scripts_and_data.scripts.random_forest
data = scripts_and_data.data.titanic.Titanic

rfObj = rf.RandomForest(data.TARGET_VAR_NAME, data.TRAIN, data.TEST)
model = rfObj.createModel()




