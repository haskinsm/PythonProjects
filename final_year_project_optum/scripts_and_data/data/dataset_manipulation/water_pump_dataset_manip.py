# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:34:03 2021

@author: micha

description: This is where the orginal water pump dataset is read in and data manipulation occurs
"""

import pandas as pd 
import numpy as np
import os

############################ Read in data #################################
#get the current working directory 
currentDir = os.getcwd()
## Check if in correct w. directory (C:\Users\micha\Documents\3rd year\Software Applications\PythonSpyder(Anaconda V)\final_year_project_optum)


# Append name of file you want to read to the current directory
datasFilePath = os.path.join(currentDir, "scripts_and_data\\data\\datasets\\water_pump_data\\training_set.csv")
# read in data (in this case the labels could only be downloaded seperately)
dataValues = pd.read_csv(datasFilePath, parse_dates = ['date_recorded'])  
# read in labels
labelsFilePath = os.path.join(currentDir, "scripts_and_data\\data\\datasets\\water_pump_data\\training_set_labels.csv")
dataLabels = pd.read_csv(labelsFilePath) # this contains the vars ID and the target variable status_group 


########################## Now merge the data and dataLabels (Joining them by ID) ###########################
## This will perform an inner join based on the common variable id (So if their is an id that is in one dataframe and not the 
# other this observation will be left out, as it is of no value)
data = pd.merge(left = dataValues, right = dataLabels, left_on = 'id', right_on = 'id')

# Set the vairable id to be the index of the dataframe
data = data.set_index('id')

# 59,000 rows, 40 columns 
data.info()
data.describe()

########################## Deal with missing data ######################################
# Check for missing data
data.isnull().sum()
# drop the column scheme name as too many missing values and not useful
data.drop('scheme_name', axis = 1, inplace = True)

# give nan values the value of zero 
data['funder'] = data['funder'].fillna(0)
data['installer'] = data['installer'].fillna(0)
data['subvillage'] = data['subvillage'].fillna(0)
data['public_meeting'] = data['installer'].fillna(0)
data['scheme_management'] = data['scheme_management'].fillna(0)
data['permit'] = data['permit'].fillna(0)

### Other instances of missing data
## The variable wpt_name has observations with value of 'none', but this is accpetable 
## The variable population has a number of observations with the value of zero. It is unclear whether this 
   # is missing data or is real data **************************************************** Ask Peter 
## The variable construction_year contains missing data with the value of zero. The first 'real' recording of construction year is 
   # 1960. So I will reset construction year so that 1960 has value of 1, 1961 has value of 2, etc and the missing data
   # has the median value of 27 
temp = data
data['construction_year'].replace(0, data['construction_year'].median(), inplace = True) 
data['construction_year'] = data['construction_year'] - 1959
# now construction year has missing data with median value of 27 and all the other datapoints values range from 1-> 54

# Many of the variables contain the value 'unknown', but this is acceptable 

# The variables payment & payment_type, quantity & quantity_group, waterpoint_type & waterpoint_type_group
# appear to be displaying the same information so drop one variable in each pair 
data.drop('payment', axis = 1, inplace = True)

   


######################## Further data manipultaion ############################
# The variable recorded_by only contains one value, so it is useless information 
data.drop('recorded_by', axis = 1, inplace = True)

# The target variable contrains 3 levels:P functional, non-functional and needs repair 
# I will create another variable called repair which will indicate if a water pump needs to be repaired
# 0 will indicate no repair needed, 1 will indicate repair needed. I will give non-functional water pumps 
# a repair value of 1 
## Think I may have made this too influential on the outcome so may need to be changed*****************************************
def repair_labels(row):
    if row['status_group'] == 'functional':
        return 0
    if row['status_group'] == 'functional needs repair':
        return 1
    if row['status_group'] == 'non functional':
        return 1
    return 'Other'
data['repair'] = data.apply(lambda row: repair_labels(row), axis = 1)

def response_labels(row):
    if row['status_group'] == 'non functional':
        return 0
    else:
        return 1
data['status_group'] = data.apply(lambda row: response_labels(row), axis = 1)

# Switch position of columns repair and status_group so status_group is last (it is the response variable) 
colList = list(data)
colList[37], colList[38] = colList[38], colList[37]
data.columns = colList

# Get number of levels of variable
len(data['wpt_name'].unique()) #37,400 -> this is too many so drop this variable 
data.drop('wpt_name', axis = 1, inplace = True)

######################### Convert categorical data to integer levels ####################
#newData, origLevels = pd.factorize(data[''])
# OrdinalEncoder





























