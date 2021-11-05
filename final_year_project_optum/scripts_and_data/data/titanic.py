# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:01:31 2021

@author: micha
"""

"""
My idea is to create a number of dataset files where I read in the data from an online source or local folder, then manipulate the data
so it is ready for use in machine learning scripts where I will import these 'data' files.

Benefits:
    - Less code duplication making it easier to make changes to the code. 
    - Less files making it easier to work with 
    
Might then be able to make a master script where I can import the results from a number of machine learning scripts

Possible downfall:
    - Might not be possible to make universal machine learning scripts that will work for every dataset. Might be able to get around this using
    conditions, but it is a worry. 
"""

import pandas as pd 
import numpy as np
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt


"""
The titanic dataset on Kaggle is already broken up into a training and test set
"""
class Titanic():
    ############################ Read in data #################################
    #get the current working directory 
    currentDir = os.getcwd()
    ## Check if in correct w. directory (C:\Users\micha\Documents\3rd year\Software Applications\PythonSpyder(Anaconda V)\final_year_project_optum)
    
    # Append name of file you want to read to the current directory 
    testFilesDir = os.path.join(currentDir, "scripts_and_data\\data\\datasets\\titanic_data\\test.csv")
    test = pd.read_csv(testFilesDir)
    #"C:\Users\micha\Documents\3rd year\Software Applications\PythonSpyder(Anaconda V)\final_year_project_optum\data\datasets\titanic\test.csv"
    
    trainFilesDir = os.path.join(currentDir, 'scripts_and_data\\data\\datasets\\titanic_data\\train.csv')
    train = pd.read_csv(trainFilesDir)
    
    # Now get y_test (the targte var for the test data)
    yTestFilesDir = os.path.join(currentDir, 'scripts_and_data\\data\\datasets\\titanic_data\\y_test.csv')
    yTest = pd.read_csv(yTestFilesDir)
    
    ############################ Check data ###################################
    """
    train.head(3)
    train.info()  # Age, Embarked contain null values
    train.describe()
    
    test.head(3)
    test.info() # Age, Fare contain null values
    test.describe()

    yTest.head(3)
    """
   
    yTest.drop(['PassengerId'], axis=1, inplace=True)
    
    ############################ Manipulate Data #############################
    # Will convert the Cabin field to a binary field where 0 represents no cabin and 1 represents having a cabin.
    train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    train.drop(['Cabin'], axis=1, inplace=True)
    test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    test.drop(['Cabin'], axis=1, inplace=True)
    
    fullData = [train, test]
    ###### Remove all Null Values in
    ## Age and replace with random value in range of one std from av age
    for dataset in fullData:  #dataset is train first then test as (fullData = [train, test])
        avAge = dataset["Age"].mean()
        stdAge = dataset["Age"].std()
        ageNullCount = dataset['Age'].isnull().sum()
        ageNullRandomList = np.random.randint(avAge - stdAge, avAge + stdAge, size=ageNullCount)
        dataset['Age'][np.isnan(dataset['Age'])] = ageNullRandomList
        dataset['Age'] = dataset['Age'].astype(int)
        
    ## Embarked is null only in training dataset. Remove nulls 
    embNullCount = train['Embarked'].isnull().sum()
    embMajority = train['Embarked'].mode() ##  most frequent value in a pandas series is basically the
    # mode of the series. You can get the mode by using the pandas series mode() function
    # ******************* Remember to add [0] to the end of emnMajrity as it is a series not a string 
    # Or can use 
    a = train['Embarked'].value_counts().idxmax()
    train['Embarked'] = train['Embarked'].fillna("{}".format(embMajority[0]))
    
    # Define function to extract titles from passenger names
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""
    
    # Create a new feature Title, containing the titles of passenger names
    for dataset in fullData:
        dataset['Title'] = dataset['Name'].apply(get_title)
        
    # Group all non-common titles into one single grouping "Rare"
    for dataset in fullData:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    for dataset in fullData:
        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        
    ## drop Name and Ticket field, and mapping string values 
    for dataset in fullData:
        dataset.drop(['Name', 'Ticket'], axis=1, inplace=True)
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    
    ######################## Some interesting plots ###########################
    ######## Correaltion heatmap
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    columns = ["PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Has_Cabin", "Title"]
    # Can only have columns that can be converted to floats
    sns.heatmap(train[columns].astype(float).corr(),linewidths=0.1,vmax=1.0, 
                square=True, cmap=colormap, linecolor='white', annot=True)
    
    
    ######## Pairplot -> see dist from one feature to another
    g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'SibSp', u'Parch', u'Fare', u'Embarked', u'Has_Cabin', u'Title']],
                     hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
    g.set(xticklabels=[])
    
    
    
    ###################### Set vars that will be accessed later as constants #######################
    TARGET_VAR_NAME = "Survived"
    TRAIN = train
    TEST = test 
    YTEST = yTest





