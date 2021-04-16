# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:03:51 2021

@author: micha
"""
### Useful sources:
    # GUI in Python: https://realpython.com/pysimplegui-python/
    # https://www.youtube.com/watch?v=Aim_7fC-inw
    # https://www.geeksforgeeks.org/file-explorer-in-python-using-tkinter/

# Requirments for script 1: 
    # Part 1 (60%)
    # The aim of this assignment is to build a number of Python programmes that will allow a user to use
    # and interact with datasets on data.gov.ie.
    
    # Script 1
    # After a user has downloaded the files, create a Python script (in version 3.x) that can:
    # 1. Read in the files. The file formats that the programme should accept are:
    #• csv
    #• json
    #• json-stat
    #2. The user should be able, at minimum, to enter the file name. Ideally the user should be able
    #to select the file using a GUI.
    #3. Extract the data and save the data in a suitable format.
    #4. Provide some descriptive statistics on the dataset.
    #5. Export the data to an Excel spreadsheet

## Using Python V3


######################### Read in File
import tkinter as tk 
from tkinter import filedialog

root = tk.Tk() 
root.wm_attributes('-topmost', 1) 
## The above line and the parent = root argument in the
    # filedialog.askopenfilename() function (below) makes sure that the file explorer window 
    # opens on top of other running programmes. 
root.withdraw() #this prevents the tk pop up window from opening


filePath = filedialog.askopenfilename(parent = root,
                                       initialdir = "/Downloads", 
                                       title = "Select a file please. You can change file type in the bottom right. Only the following formats are valid: csv, json, and json-stat",
                                       filetypes = (("csv files", "*.csv"), ("json files", "*.json"), ("json-stat (All files incl.)", "*.*")) ) 
## This allows a user to select a file in a pop up file explorer 
    ## The parent = root argument ensures that the pop up window opens on top of other programs
    ## Initial Directory is set to open in the downloads (section of users device). 
    ## The file types that can be selected are csv, json and json-stat. It is not possible to individually
        ## define json-stat files so all files are displayed. Checks later on ensure that errors caused
        ## by selecting other file types are caught. When they are caught the program execution is terminated. 
    ## This will return something like: C:/Users/micha/Downloads/CovidStatisticsProfileHPSCIrelandOpenData.csv
    ## Atm it only selects one file, to select multiple change function to: .askopenfilenames()
   
   


######################## Check FilePath
validPath = True

# 1st check if filePath is empty
## The below if will set ValidPath to False if no file has been entered
if not filePath:
    validPath = False 

# 2nd check if file path is valid (i.e. opens a file)
## File path should always be valid as user is selecting a file from their own device, but should check anyway
try:
    f = open(filePath)
except FileNotFoundError:  
    validPath = False
else:
    f.close()



########################## Read in data into a Pandas Dataframe
import pandas as pd
import json
from pandas.io.json import json_normalize
import jsonstat   ## Requires: pip install jsonstat.py  (Do this in anaconda powershell (might be different for non-windows OS)) *********************
import sys ## For sys.exit()

df = [] ## initialize dataframe

if(validPath == True):
    ## Now will establish if the file is a csv, json, or json-stat 
    if(filePath.endswith('.csv')):
        ## Enters here if its a csv file
        chosenSep = input("Please enter a delimiter/separator for your chosen CSV file (Default is ','): ")
        if( chosenSep == ''): 
            chosenSep = ',' ## This will make sure that default is used if nothing is entered above
        df = pd.read_csv(filePath, sep=chosenSep, header = 'infer', encoding='latin-1')    
    elif(filePath.endswith('.json')):
        ## Enters here if its a JSON file
        with open(filePath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read() ## The json string
            data = json.loads(text) ##Creates a list with the json # https://www.freecodecamp.org/news/python-read-json-file-how-to-load-json-from-a-file-and-parse-dumps/
            # converting json dataset from dictionary to dataframe
            df = json_normalize(data['results'])
    else: 
        ## Enters here if Json-Stat or other file 
        # Initialize JsonStatCollection from the file
        try:
            collection = jsonstat.from_file(filePath)
        except:
            print("The file you have entered is not a valid CSV, JSON, or JSON-STAT file. Please re-run the program to try again")
            sys.exit() ## This will terminate the script execution
        else:
            data = collection.dataset(0) ## Should be only one dataset in the collection
            df = data.to_data_frame()
        

    ######################## Descriptive statistics on the dataset
    ## As dataset can be anything there is very little descriptive statistics that can be provided.
    import numpy as np
    print("\nDescriptive Statistics: ")
    
    ## count of observations (rows) and variables (columns)
    numRows = df.shape[0]
    numCols = df.shape[1]
    print("\nThis dataset has", numRows, "observations and", numCols, "columns.")
    
    ## Number of blanks or 'nan' entries
    numNanEntries = (df.astype(str) == 'nan').values.sum()
    numBlanks = (df.astype(str) == '').values.sum()
    if( numRows*numCols == numNanEntries):
        print("There was a problem reading in the file. If you selected a csv file this may be because" + 
               "you selected an incorrect seperator for your dataset. The seperator used was" + chosenSep)
    else:
        print("\nThe created dataframe contains", numNanEntries, "'nan' entries and", numBlanks, "blanks." )
    
    ## Ouptut the number of numerica nd non numeric columns
    is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    # is_number(df.dtypes) ## Outputs boolean array. True for numeric, False for non-numeric
    numNumericCols = np.count_nonzero( is_number(df.dtypes) == True)
    numNonNumericCols = np.count_nonzero( is_number(df.dtypes) == False)
    print("There are", numNumericCols, "numeric columns and", numNonNumericCols, "non-numeric columns.")
    
    ## Now will display the first 5 rows of the dataset
    print("\nThe first 5 rows of the dataset: \n", df.head())
    
    
    
    ######################## Export the data to an excel spreadsheet
    import os, time
    
    ## create excel file in local working directory
    fileName = input("Please enter a file name for the excel worksheet: ")
    ## Now make sure that the user has not entered .txt or .csv so take everyhting before the '.'(if there is a '.' in the string)
    partionedString = fileName.partition('.')
    fileName = partionedString[0]
    ## If user has entered nothing I will just call the file Data and append the UnixEpochTime to the name.
    ## This will ensure filenames are unique and no existing datasets are being accidently overwritten by this program
    if( fileName == ""):
        fileName = "Data"
        unixTimeStamp = round(time.time()) ## Need to round the unix time stamp as can't have a decimal place in a file name 
        fileName = fileName + str(unixTimeStamp) ## Will give something like: Data1617719371
    ## The file will be saved as an xlsx
    fileName = fileName + ".xlsx"
    
    # get the current working directory of user 
    currentDir = os.getcwd()
    # Append name of file with the current directory 
    filesDir = os.path.join(currentDir, fileName) 
    
    ## Now write data to the newly created excel file
    df.to_excel(filesDir)
    
    print("\nThe dataset has been saved as an xlsx file called", fileName, "in your local working directory." + "\n" +
          "Path: " + filesDir)
    
else:
    ## Enters here if file path is not valid. 
    print("The file you selected was not valid. Please re-run the program to try again.")
