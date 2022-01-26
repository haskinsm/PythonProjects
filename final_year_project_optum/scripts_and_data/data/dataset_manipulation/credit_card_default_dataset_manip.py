# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:43:33 2022

@author: micha
"""

import pandas as pd 
import os

############################ Read in data #################################
# Change to correct directory 
CORRECTDIR = "C:/Users/micha/Documents/3rd year/Software Applications/PythonSpyder(Anaconda V)/final_year_project_optum"
os.chdir(CORRECTDIR)

# Append name of file you want to read to the current directory
datasFilePath = os.path.join(CORRECTDIR, "scripts_and_data\\data\\datasets\\credit_card_default_data\\original_data.csv")

data = pd.read_csv(datasFilePath, skiprows = 1, index_col = "ID")  #First row is useless information so skip it 

# Change name of target var from 'default payment next month' 
data['default'] = data.iloc[:, -1]
data.drop('default payment next month', axis = 1, inplace = True)

############################ Check data ##############################
#### Get number of levels of variable
# Cant have nominal variables with large number of unique values as my laptop is not able to handle OneHotEncoding these vars
levels = {}
for i in data.columns:
    levels[i] = data[i].value_counts()
#print(levels)

summary = data.describe()
### All variables seem to be ok, there is no missing data 



######################### Convert categorical variables to be ordinal using OrdinalEncoder or nominal using OneHotEncoder ####################
#### using get_dummies for Nominal data and OrdinalEncoder for ordianl data

# While the variable quantity appears to be semi ordinal there is not enough information to warrent encoding
# it as ordinal as it is unclear what order to put 'seasonal', 'unknown', 'dry', 'enough' , 'insufficient' in

XNom = data.iloc[:, [1,2,3,5,6,7,8,9,10,]].astype(str)
XCont = data.iloc[:, [0,4,11,12,13,14,15,16,17,18,19,20,21,22]].astype(str)
y = pd.DataFrame(data.iloc[:, -1].astype(str))

XNomDumDf = pd.get_dummies(XNom, columns = XNom.columns)
# Now merge everything back together to get the final encoded dataset
encData = pd.concat([pd.concat([XCont, XNomDumDf], axis = 1), y], axis = 1) 


#################### Now save as a csv file ####################
# This csv file will be read in in water_pump_dataset.py 
csvFilePath = os.path.join(CORRECTDIR, "scripts_and_data\\data\\datasets\\credit_card_default_data\\processed_data.csv")
encData.to_csv(csvFilePath, header = True, index = True)

