# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:37:00 2022

@author: micha
"""

import pandas as pd 
import os

############################ Read in data #################################
# Change to correct directory 
CORRECTDIR = "C:/Users/micha/Documents/3rd year/Software Applications/PythonSpyder(Anaconda V)/final_year_project_optum"
os.chdir(CORRECTDIR)

# Append name of file you want to read to the current directory
datasFilePath = os.path.join(CORRECTDIR, "scripts_and_data\\data\\datasets\\income_data\\adult_original.csv")
# Note: This csv file does not have headers 
columnNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(datasFilePath, header = 0, names = columnNames)  

############################ Drop Variables ##############################
#### Get number of levels of variable
# Cant have variables with large number of unique values as my laptop is not able to handle OneHotEncoding these vars
levels = {}
for i in data.columns:
    levels[i] = data[i].value_counts()
#print(levels)

# Drop education as it is just the label version of education-num 
data.drop('education', axis = 1, inplace = True)
# Drop fnlwgt as its value has different meaning depending on the state it was recorded in.
# "People with similar demographic characteristics should have similar weights.  There is one important caveat to remember
# about this statement.  That is that since the CPS sample is actually a collection of 51 state samples, each with its own
# probability of selection, the statement only applies within state." src: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
data.drop('fnlwgt', axis = 1, inplace = True)


############################ Missing data ################################
data.describe()
data.info()
data.isnull().sum() #Missing data appears to be represented as '?' in a few of the vars 
levels # Can see the count of each different variables unique values

#### Replace missing values in any nominal variables with 'unknown'
data['native-country'].replace('?', 'unknown', inplace = True) 
data['occupation'].replace('?', 'unknown', inplace = True) 
data['workclass'].replace('?', 'unknown', inplace = True) 


### Not known if following variables contain missing data
# 44,806 observations have a cpaital gains of 0. It is unknown if this is missing data or not. 
# 46,559 obeservations have a capital loss of 0. It is unknown if this is missing data or not. 


############################ Further data manipulation ########################
#### Change response variable 'income' values to 1 if income is greater than or equal to 50 and 0 if not
def responseLabels(row):
    if row['income'] == '<=50K':
        return 1
    else:
        return 0
data['income'] = data.apply(lambda row: responseLabels(row), axis = 1)



######################### Convert categorical variables to be ordinal using OrdinalEncoder or nominal using OneHotEncoder ####################
#### using get_dummies for Nominal data and OrdinalEncoder for ordianl data

# While the variable quantity appears to be semi ordinal there is not enough information to warrent encoding
# it as ordinal as it is unclear what order to put 'seasonal', 'unknown', 'dry', 'enough' , 'insufficient' in

XNom = data.iloc[:, [1,2,3,4,5,6,7,11]].astype(str)
XCont = data.iloc[:, [0,8,9,10]].astype(str)
y = pd.DataFrame(data.iloc[:, -1].astype(str))

XNomDumDf = pd.get_dummies(XNom, columns = XNom.columns)
# Now merge everything back together to get the final encoded dataset
encData = pd.concat([pd.concat([XCont, XNomDumDf], axis = 1), y], axis = 1) 


#################### Now save as a csv file ####################
# This csv file will be read in in water_pump_dataset.py 
csvFilePath = os.path.join(CORRECTDIR, "scripts_and_data\\data\\datasets\\income_data\\processed_data.csv")
encData.to_csv(csvFilePath, header = True, index = True)












