# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:42:38 2021

@author: micha
"""
## Below two lines are list comprehension
strings = ['a','as','bat', 'car', 'like', 'sweeet']
[newStrings.upper() for newStrings in strings if len(newStrings)>2 ] ### Note the if condition must contain the var after for 

# Next to Matplotlib and NumPy, pandas is one of the most widely used Python libraries in data 
# science. The pandas library contains high-level data structures and manipulation tools to make 
# data analysis fast and easy in Python. pandas is built on top of the NumPy library.

# There are two main data structures in pandas: Series and DataFrames. 
# To use these data structures you need to import both the NumPy and 
# pandas library as follows:

import numpy as np ## Abbreviating import as np 
from pandas import Series, DataFrame ## The two main data structures in pandas
import pandas as pd   

# A Series is a one dimensional array-like object containing an array of data and 
# an associated array of data labels, called its index. 
# The simplest Series is formed from a single array of data:
obj = Series([4,7,-5,3])
obj
# index is shown on the left and the values on the right. A default index from 0 
# to n-1 (where n is the length of the data) is created.

# get the array representation and index object of the Series via the value and index objects
obj.values
obj.index
# Often it will be desirable to create a Series with an index identifying each data point
obj2 = Series([4,7,-5,3], index=['d','a','b','c'])
obj2
# sorted(obj2)
obj2.sort_index() ##Sorts by index alphabetically
obj2.sort_index(ascending=False) ##Sorts by index in reverse
obj2.sort_values() ##Sorts by values low to high



# DataFrame
# One way to create a DataFrame is by using common Python data structures, for example it is 
# possible to pass a dictionary of lists to the DataFrame constructor.
data = {'years' : [2013, 2014, 2015, 2014],
        'team' : ['Chelsea', 'Chelsea', 'Chelsea', 'Liverpool'],
        'wins' : [11,8,10,15],
        'losses' : [5,8,6,1]}
football = pd.DataFrame(data)
print(football)

# It is more common that you will have a dataset that you want to read into a DataFrame
## Load the three datafiles into a pandas DataFrame object using pandas.read_csv.

# read in user details
u_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
users = pd.read_csv('data/ml-1m/users.dat', sep='::', names=u_cols, encoding='latin-1') 
 ##CTRL+I for help 

# read in ratings data
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', names=r_cols, encoding='latin-1')


# read in movie details
m_cols = ['movie_id', 'title', 'genres']
movies = pd.read_table("data/ml-1m/movies.dat", sep='::', names=m_cols, usecols=range(5),
                     encoding='latin-1')

## Now check if data has been read in correctly. May need to change sep if not
## Prob best to read in the data with an uncommon delimiter first to establish 
# what delimieter is reuqired or view in seperate window if possible 
## sep='\t'           # Tab-separated value file.  # For single char separators: [:,(,)]
users.head(5)
ratings.head(5)
movies.head(5) ##Not sure how to correctly sep this. 

movies.info()
# Each row was assigned an index of 0 to N-1, where N is the number of rows in the
# DataFrame. pandas will do this by default if an index is not specified.
# There are 3,883 rows.
# The dataset has three columns.
# The datatypes of each column.
# An approximate amount of RAM (memory_usage ) used to hold the DataFrame

users.describe()
## This displays basic statistics about the datasets numeric columns

#Selecting a single column from the DataFrame will return a Series object
movies['title'].head()

# Multiple columns can be selected by passing a list of column names to the DataFrame, 
# the output will be a DataFrame
users[['age','zip_code']].head()


# Row selection can be done multiple ways. Using an individual index or boolean 
# indexing are typically easiest.
print(users[users.age > 25].head(3))
print('\n')

print( users[ (users.age == 50) | (users.age < 30) ].head(3) )
print('\n')

print( users[(users.gender == 'F') | (users.age < 30)].head(3) )


# It is possible to set another field (for example user_id) as the index using the 
# set_index method. By default, set_index returns a new DataFrame, but you can 
# specify if you'd like the changes to occur in place.
print(users.set_index('user_id').head())
print('\n')
  ##Now show that it did not change orginal dataFrame:
print(users.head())
print("\n^^^It didn't actually change the DataFrame. ^^^\n") 


users_new_index = users.set_index('user_id')
print(users_new_index.head())
## Set index returns a new dataframe


# If you want to modify the existing DataFrame, you need to use the inplace parameter:
users.set_index('user_id', inplace = True)


# pandas.merge allows two DataFrames to be joined on one or more keys. This is similar to
# SQL's JOIN clause,
# By default, pandas.merge operates as an inner join, this can be changed using the how
# parameter. pandas infers which columns to use as the merge (or join) keys based on the
# overlapping names.

lens = pd.merge(users, ratings, on = 'user_id')
lens.head()
lens = pd.merge(lens, movies, on = 'movie_id')
lens.head()
## lens now contains all 3 datasets combined.


# There are numerous methods available for working with DataFrames. 
# Some useful functions are: pandas.groupby
# With queries you often have to divide the data into groups, perform some operations 
# on each of the groups, and then report the results for each group.
# pandas groupby is useful for these types of data analysis.
# This is similar to the SQL GROUP BY construct. 
# pandas.pivot_table
# The function pandas.pivot_table can be used to create spreadsheet-style pivot tables.
# A good introduction to pandas is available at
# https://pandas.pydata.org/docs/user_guide/10min.html
# Further detail is available at:
# https://pandas.pydata.org/docs/user_guide/index.html
# Way better: https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

## How many movies in each genre
lens.describe()
lens['genres'].value_counts()## Count the number of different genre combinations
len(lens.groupby("genres").count()) ## Number of categories
# or
lens['genres'].nunique() ## Counts number of non-null unique networks

averageRating = lens[['movie_id','rating']].groupby(['movie_id']).mean()
averageRating.tail(5)
averageRating.sort_values('rating', ascending = False).head(10)
## or
averageRating.nlargest(10, 'rating')
## The highest rated movies are: 3233, 3881,989,1830, 787    ........   

# Get the first rating for each film
lens.groupby('movie_id').first('rating')
# Get the count of ratings for each film
lens.groupby('movie_id')['rating'].count()
# Get the sum of ratings for each film
lens.groupby('movie_id')['rating'].sum()


pd.pivot_table(lens, values = 'rating', index = ['gender', 'movie_id'], aggfunc = np.mean)
## Above pivot table gets mean ratings for each film by gender

pd.pivot_table(lens, values = 'rating', index = ['age', 'movie_id'], aggfunc = np.mean)
    





## Practice with Series -> Get 2nd lowest scorer
import pandas as pd
name = ["Adam", "Charlie", "Tom", "Eileen", "Scott"]
score = [44.5, 44.5, 50, 50, 60]
      
ser = pd.Series(score, index = name)
z = ser.min()
while z == ser.min():
    ser.drop(ser[ser==z].index[0], inplace=True) ## Otherwise will loop infinitely, indiactes that I want to change existing series
    ## ser[ser==z].index[0] -> Gets index of first occurence of value == z
result = ""
x = ser.min()
while x == ser.min():
    result += "" + ser[ser==x].index[0] +"\n"
    ser.drop(ser[ser==x].index[0], inplace=True)
print(result)


## Also works (If you had a stream of input) 
d={} #Empty Dict
for _ in range(int(input())): #range for number of students
    Name=input() 
    Grade=float(input()) 
    d[Name]=Grade #Key = Name, Value = Grade
    
v=d.values() #Store values/test scores here
second=sorted(list(set(v)))[1] #Remove duplicate grades using set data type and take 2nd lowest grade from sorted list

second_lowest=[] 
for key,value in d.items():  ## Get names of students who scored 2nd lowest
    if value==second: 
        second_lowest.append(key) 
        
second_lowest.sort() 
for name in second_lowest: #Now print the names of the students 
    print (name) 


