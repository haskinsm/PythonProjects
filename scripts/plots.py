# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:11:38 2021

@author: micha
"""

# Python has many visualisation tools but we will mainly focus on matplotlib.
#  To see the kinds of visualizations you can make with matplotlib, 
# visit the sample gallery at http://matplotlib.org/. When you click on a visualization 
# in the gallery, you can see the code used to generate the plot.
 
# We will start by creating some common charts from the matplotlib.pyplot library. The
# library is imported as follows:
import matplotlib.pyplot as plt

squares = [1, 4, 9, 16, 25]
plt.plot(squares)
## Plot is outputted to the plot pane, beside the variable explorer and help buttons

# matplotlib allows you to adjust every feature of a visualization. 
# The following code increases the thickness of the line, adds a chart title
# and a title for each of the axes, and styles the tick marks

plt.plot(squares, linewidth = 5) # Set line thickness 

# set chart title and label axis
plt.title('Square Numbers', fontsize = 24)
plt.xlabel('Value', fontsize = 14)
plt.ylabel('Square of Value', fontsize = 14)

# set the size of the tick labels
plt.tick_params(axis='both', labelsize = 14)


# The chart is not plotting correctly, the square of 4.0 for example is shown as 25. 
# This is because plot() assumes that the first data point of the x coordinate is 0,
# but the first point corresponds to an x-value of 1. To fix this, plot() 
# requires both the input and output values used to calculate the squares:

input_values = [1,2,3,4,5]

# set the thickness of the line
plt.plot(input_values, squares, linewidth = 5)


# A scatterplot is also a suitable plot for the above data.
x_values = input_values
y_values = squares
plt.scatter(x_values, y_values, s=100)

## Now plot the squares of 1 to 1000
x_vals = []
y_vals = []
for i in range(1, 1001, 1):
    x_vals.append(i)
    y_vals.append(i*i)
plt.scatter(x_vals, y_vals, s=100)


## save the plot to a file with a call to plt.savefig()
plt.savefig('squares_plot.png', bbox_inches = 'tight')
# The first argument is the filename for the plot image, the second argument trims extra whitespace
# from the plot.





# Plots in matplotlib reside within a Figure object. 
# You can create a new figure with plt.figure:
fig = plt.figure()

# plt.figure has a number of options, notably figsize will guarantee that 
# figure has a certain size and aspect ratio.

# You can create a number of subplots in the figure using add_subplot:
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,2,3)
ax3 = fig.add_subplot(2,2,4)
# means that the figure should be 2x2 and we are selecting
# the first of the four subplots (numbered from 1). 

# When you issue a plotting command matplotlib draws on the last figure and 
# subplot used (creating one if necessary).

from numpy.random import randn 
ax1.plot(randn(50).cumsum(), 'k--')
# The ‘k--‘ is a style option instructing matplotlib to plot a black
# dashed line.
## This is displayed on the last plot if you run it all at once (ie. dont run line by line) 
# if you dont specify ax1

## It is possible to draw on the other two of the 3 plots as follows:
import numpy as np
ax2.hist(randn(100), bins=20, color = 'k', alpha = 0.3)
ax3.scatter(np.arange(30), np.arange(30) + 3 * randn(30))

# set chart titles
fig.suptitle('Example of plots using matplotlib in python', fontsize = 24)
ax1.set_title('Example of lineplot')
ax2.set_title('Example of histogram')
ax3.set_title('Example of scatterplot')

plt.tight_layout() # will also adjust spacing between subplots to minimize the overlaps.



## Now create plots using the Movies dataet from previous week
from pandas import Series, DataFrame ## The two main data structures in pandas
import pandas as pd   
# read in user details
u_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
users = pd.read_csv('data/ml-1m/users.dat', sep='::', names=u_cols, encoding='latin-1') 

# read in ratings data
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', names=r_cols, encoding='latin-1')

# read in movie details
m_cols = ['movie_id', 'title', 'genres']
movies = pd.read_table("data/ml-1m/movies.dat", sep='::', names=m_cols, usecols=range(5),
                     encoding='latin-1')

users.head(5)
ratings.head(5)
movies.head(5) ##Not sure how to correctly sep this. 

# If you want to modify the existing DataFrame, you need to use the inplace parameter:
users.set_index('user_id', inplace = True)

lens = pd.merge(users, ratings, on = 'user_id')
lens.head()
lens = pd.merge(lens, movies, on = 'movie_id')
lens.head()
## lens now contains all 3 datasets combined.



## Create a barcart of ratings
averageRating = lens[['movie_id','rating']].groupby(['movie_id']).mean()
plt.bar([1,2,3,4,5], averageRating['rating'].head(), color = 'r', width = 0.25)

## Create a boxplot of ratings by gender
dataM = lens[lens.gender == 'M']
dataF = lens[lens.gender == 'F']
genderRating = [dataM['rating'], dataF['rating']]
labels = list('MF')
plt.boxplot(genderRating, vert = True, patch_artist = True, labels = labels)
















