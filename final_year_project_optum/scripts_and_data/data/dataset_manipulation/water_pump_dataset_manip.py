# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:34:03 2021

@author: micha

description: This is where the orginal water pump dataset is read in and data manipulation occurs
"""

import pandas as pd 
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from fast_ml.model_development import train_valid_test_split
from sklearn.compose import ColumnTransformer


############################ Read in data #################################
# Change to correct directory 
correctDirPath = "C:/Users/micha/Documents/3rd year/Software Applications/PythonSpyder(Anaconda V)/final_year_project_optum"
os.chdir(correctDirPath)

# Append name of file you want to read to the current directory
datasFilePath = os.path.join(correctDirPath, "scripts_and_data\\data\\datasets\\water_pump_data\\training_set.csv")
# read in data (in this case the labels could only be downloaded seperately)
dataValues = pd.read_csv(datasFilePath, parse_dates = ['date_recorded'])  
# read in labels
labelsFilePath = os.path.join(correctDirPath, "scripts_and_data\\data\\datasets\\water_pump_data\\training_set_labels.csv")
dataLabels = pd.read_csv(labelsFilePath) # this contains the vars ID and the target variable status_group 


########################## Now merge the data and dataLabels (Joining them by ID) ###########################
## This will perform an inner join based on the common variable id (So if their is an id that is in one dataframe and not the 
# other this observation will be left out, as it is of no value)
data = pd.merge(left = dataValues, right = dataLabels, left_on = 'id', right_on = 'id')

# Set the vairable id to be the index of the dataframe
data = data.set_index('id')

# 59,000 rows, 40 columns 
#data.info()
#data.describe()

########################## Droping variables ###################################
# Check for missing data
#data.isnull().sum()
# drop the column scheme name as too many missing values and not useful
data.drop('scheme_name', axis = 1, inplace = True)

##### Remove duplicate and very similar variables
data.drop('payment', axis = 1, inplace = True) # Duplicate of payment_type
data.drop('quantity_group', axis = 1, inplace = True) # Duplicate of quantity
data.drop('extraction_type', axis = 1, inplace = True) # Duplicate of extraction_type_group

# The variable recorded_by only contains one value, so it is useless information 
data.drop('recorded_by', axis = 1, inplace = True)

#### Get number of levels of variable
# Cant have variables with large number of unique values as my laptop is not able to handle OneHotEncoding these vars
levels = {}
for i in data.columns:
    levels[i] = data[i].value_counts()
#print(levels)
data.drop('wpt_name', axis = 1, inplace = True) ##37,400 unique values -> this is too many so drop this variable 
data.drop('subvillage', axis = 1, inplace = True) #19,288 unique values
data.drop('num_private', axis = 1, inplace = True) # 98.7% are 0
data.drop('public_meeting', axis = 1, inplace = True) #2146 unique values

########################## Deal with missing data ######################################
# give nan values sensible values
data['funder'] = data['funder'].fillna('unknown') #unknown
data['installer'] = data['installer'].fillna('unknown') #unknown
data['scheme_management'] = data['scheme_management'].fillna('unknown') #unknown
data['permit'] = data['permit'].fillna(data['permit'].median()) #median

### Other instances of missing data
## The variable population has a number of observations with the value of zero. It is unclear whether this 
   # is missing data or is real data 
## The variable construction_year contains missing data with the value of zero. The first 'real' recording of construction year is 
   # 1960. So I will reset construction year so that 1960 has value of 1, 1961 has value of 2, etc and the missing data
   # has the median value of 27 
data['construction_year'].replace(0, data['construction_year'].median(), inplace = True) 
data['construction_year'] = data['construction_year'] - 1959
# now construction year has missing data with median value of 27 and all the other datapoints values range from 1-> 54

# Many of the variables contain the value 'unknown', but this is acceptable 

######################## Further data manipultaion ############################

# The target variable contrains 3 levels:P functional, non-functional and needs repair 
# I will create another variable called repair which will indicate if a water pump needs to be repaired
# 0 will indicate no repair needed, 1 will indicate repair needed. I will give non-functional water pumps 
# a repair value of 1 
## Think I may have made this too influential on the outcome so may need to be changed*****************************************
"""
### Will prob remove this section 
def repair_labels(row):
    if row['status_group'] == 'functional':
        return 0
    if row['status_group'] == 'functional needs repair':
        return 1
    if row['status_group'] == 'non functional':
        return 1
    return 'Other'
data['repair'] = data.apply(lambda row: repair_labels(row), axis = 1)
"""

def response_labels(row):
    if row['status_group'] == 'non functional':
        return 0
    else:
        return 1
data['status_group'] = data.apply(lambda row: response_labels(row), axis = 1)

"""
### Remove this if remove repair var 
# Switch position of columns repair and status_group so status_group is last (it is the response variable) 
colList = list(data)
colLength = len(colList)
colList[colLength - 2], colList[colLength - 1] = colList[colLength - 1], colList[colLength - 2]
data.columns = colList
"""

    

######## The location variables
# There are variables recording the longitude and latitude, these variables will likely provide enough information to 
# account for location of a water pump and so the variables district_code, region, region_code, subvillage and ward may 
# not be useful, but for now I will leave them in


######################### Convert categorical variables to be ordinal using OrdinalEncoder or nominal using OneHotEncoder ####################
#### using OneHotEncoder for Nominal data and OrdinalEncoddr for ordianl data

# While the variable quantity appears to be semi ordinal there is not enough information to warrent encoding
# it as ordinal as it is unclear what order to put 'seasonal', 'unknown', 'dry', 'enough' , 'insufficient' in

XNom = data.iloc[:, [2,4,7,8,9,10,11,12,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29]].astype(str)
XCont = data.iloc[:, [0,1,3,5,6,13,16]].astype(str)
y = pd.DataFrame(data.iloc[:, -1].astype(str))
#onehotEnc = preprocessing.OneHotEncoder()
#C = pd.DataFrame(onehotEnc.fit_transform(C).toarray()) # No column names 

XNomDumDf = pd.get_dummies(XNom, columns = XNom.columns)
# Now merge everything back together to get the final encoded dataset
encData = pd.concat([pd.concat([XCont, XNomDumDf], axis = 1), y], axis = 1) 


#################### Now save as a csv file ####################
# This csv file will be read in in water_pump_dataset.py 
csvFilePath = os.path.join(correctDirPath, "scripts_and_data\\data\\datasets\\water_pump_data\\processed_data.csv")
encData.to_csv(csvFilePath, header = True, index = True)




"""
    ###### Discarded code 
# First split into training set and then will split remaining terms again to get the test and validation sets
XTrain, XRem, yTrain, yRem = train_test_split(X,y, train_size=0.6)
XValid, XTest, yValid, yTest = train_test_split(XRem,yRem, test_size=0.5) #Test = 20% and validation = 20%

# Label Encoder
lblEnc = preprocessing.LabelEncoder()
lblEnc.fit(yTrain)
yTrain = lblEnc.transform(yTrain)
yTest = lblEnc.transform(yTest)
yValid = lblEnc.transform(yValid)

# Use ColumnTransformer to transform specific nominal columns using OneHotEncoder 
colTrans = ColumnTransformer([("categ", preprocessing.OneHotEncoder(), [2,4,7,8,9,10,11,12,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29] )])
colTrans.fit(XTrain.toarray())
XTrain = pd.DataFrame(colTrans.transform(XTrain.toarray()))
XTest = colTrans.transform(XTest)
XValid = colTrans.transform(XValid)



# Onehot Encode nominal variables ALL ATM*****************
onehotEnc = preprocessing.OneHotEncoder()
onehotEnc.fit(X)
encDf = pd.DataFrame(onehotEnc.transform(X).toarray())
a = onehotEnc.transform(X)

"""

























