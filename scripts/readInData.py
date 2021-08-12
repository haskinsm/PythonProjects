# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 12:02:04 2021

@author: micha
"""
import numpy as np ## Abbreviating import as np 
from pandas import Series, DataFrame ## The two main data structures in pandas
import pandas as pd   

## If you specify what is known as an absolute path when trying to read a file, 
# you might find that it
# runs fine on your machine but will fail when you try it with a different OS. 
# pd.read(`D:\\Documents\Python_Tutorial\Data\myFile.csv`)
## This directory may not exist on another machine

## Therefore you should obtain the directory you are currently in and tell python
## to look in that location for the file
## like so:
    
import os

#get the current working directory 
currentDir = os.getcwd()

# Append name of fiel you want to read to the current directory 
filesDir = os.path.join(currentDir, 'data\SciencePerformance.csv') ## obv replace the latter with the relevant file

# pd.read(filesDir)
from_csv = pd.read_csv(filesDir)
# Now if you ensure you always place the files you want to read in the same directory
# as your script,
# python will be able to read them no matter which machine the script is running.


## Could alternatively read in like this:
#from_csv = pd.read_csv('data\SciencePerformance.csv')

# Now check the data
from_csv.head()


## Now will write the first 10 rows of the subjecct, time, and value columns of the 
## SciencePerformance data as a HTML table
miniDataset = from_csv[['SUBJECT','TIME', 'Value']].head(10)
miniDataset
htmlCode = miniDataset.to_html()
print(htmlCode)

#write html to file
text_file = open("index.html", "w") ## Creates a file called index.html in wd 
text_file.write(htmlCode)
text_file.close()


## Now read in a second dataset called MathsPerformance.xslx
import os
#get the current working directory 
currentDir = os.getcwd()

# Append name of file you want to read to the current directory 
filesDir = os.path.join(currentDir, 'data\MathsPerformance.csv') ## obv replace the latter with the relevant file
## make sure to check that the file is actually as csv and not an xlsx
dataset = pd.read_csv(filesDir)



## It is sometimes useful to copy a table to the clipboard and read the data from 
# the clipboard. This is particularly useful if you want to copy a table #
# from a web page
mytable = pd.read_clipboard()
mytable.head()
## Its hard to find a table that reads in well by this method


# pandas provides support for a number of other filetypes, 
# see https://pandas.pydata.org/pandasdocs/stable/user_guide/io.html for further details.


# SQL
# It is possible to connect Python to a MySQL database. 
# MySQL Connector/Python enables Python
# programs to access MySQL databases. You will need to install the Python driver for
# communicating with MySQL servers.
# To install the driver, run the following code in the Anaconda terminal. (Start menu ➔ Anaconda
# ➔ Anaconda Prompt)
# conda install -c conda-forge mysql-connector-python
# The following site shows some sample code:
# https://dev.mysql.com/doc/connector-python/en/connector-python-examples.html

import mysql.connector
from mysql.connector import errorcode

config = {
    'user':'haskinsm',
    'password':'ziemae2Z',
    'database':'haskinsm_db',
    'host':'mysql.scss.tcd.ie'
    }
try:
  cnx = mysql.connector.connect(**config)
except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
else:
    ## Enters here if no errors 
    ## Now query data
    import mysql.connector
    cursor = cnx.cursor()    
    query = ("SELECT * from office")
    cursor.execute(query)   
    for (office, city, region, mgr, target, sales) in cursor:
        print("{}, {} , {}, {}, {}, {}".format( office, city, region, mgr, target, sales))
     
    ## Now will make second query
    query2 = ("SELECT * FROM salesrep")
    ## Now read in all data as a dataframe 
    import pandas as pd
    df = pd.read_sql(query2, cnx) # https://stackoverflow.com/questions/12047193/how-to-convert-sql-query-result-to-pandas-data-structure
    
    ## Close objects
    cursor.close()
    cnx.close()



## More simply could do:
from mysql.connector import (connection)

cnx = connection.MySQLConnection(user='haskinsm', password='ziemae2Z',
                                 host='mysql.scss.tcd.ie',
                                 database='haskinsm_db')
## Now query data
import mysql.connector

cursor = cnx.cursor()

query = ("SELECT * from office")

cursor.execute(query)

for (office, city, region, mgr, target, sales) in cursor:
    print("{}, {} , {}, {}, {}, {}".format( office, city, region, mgr, target, sales))
    
## Now will make second query
query2 = ("SELECT * FROM salesrep")
## Now read in all data as a dataframe 
import pandas as pd
df = pd.read_sql(query2, cnx)
    
cursor.close()

cnx.close()


