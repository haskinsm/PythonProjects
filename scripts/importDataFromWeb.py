# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:40:39 2021

@author: micha
"""
# https://towardsdatascience.com/direct-to-pandas-dataframe-ab2e97ae7574

# Importing Data from a URL using urllib

# The following code will download the “banknote authentication Data Set”
# (https://archive.ics.uci.edu/ml/machine-learningdatabases/00267/data_banknote_authentication.txt) 
# and save it locally

from urllib.request import urlretrieve

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

urlretrieve (url, "data_banknote_authentication.txt")   


## Import data as dataframe 
import pandas as pd    

df = pd.read_csv(url, header = None)



## Importing Data from a URL using requests
# The requests module lets you easily download files from the Web without having to worry
# about complicated issues such as network errors, connection problems etc. The requests
# module was written because the urllib module is complicated to use.

# The following code uses the requests.get() function to download a URL. You can see that
# the it returns a Response object. This contains the response that the web server gave 
# for your request. You can check if the request for this web page succeeded by checking the
# status_code attribute of the Response object. If the request succeeded, the downloaded
# web page is stored as a string in the Response object’s text variable.

import requests
res = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")

type(res) ## can see its a response object

res.status_code == requests.codes.ok ## See if status code is that of a succeeded request
## Returns true if so
print(res.text) ## Can see it prints all the data


import pandas as pd 
import io

urlData = res.content
myDF = pd.read_csv(io.StringIO(urlData.decode('utf-8')), header= None)



## Attempt at another option (IF WE WERE DEALING WITH API RESPONSE AND NOT A TXT FILE)
import json
import numpy as np
url = "https://wind-bow.glitch.me/twitch-api/channels/freecodecamp"
res = requests.get(url)
JSONContent = res.json()
content = json.dumps(JSONContent, indent = 4, sort_keys = True)
print(content)



# A simple way to check for errors is to call the raise_for_status() method on the response
# object. This will raise an exception if there was an error downloading the file and 
# will do nothing if the download succeeded.
import requests

res = requests.get('https://archive.ics.uci.edu/dataset.txt')

try:
    res.raise_for_status()
except Exception as exc:
    print('There was a problem: %s' % (exc))
  
# You should always call raise_for_status() after calling requests.get()



# Parsing HTML with the BeautifulSoup Module
# Good source of info: https://www.dataquest.io/blog/web-scraping-tutorial-python/


# Beautiful Soup is a module for extracting information from a HTML page. The
# BeautifulSoup module’s name is bs4 (for Beautiful Soup version 4).
# We are going to use Beautiful Soup to parse (i.e. to identify the parts) of the 
# HTML webpage at https://scss.tcd.ie/undergraduate

import requests 
import bs4 

res = requests.get('https://scss.tcd.ie/undergraduate')
res.raise_for_status()

ugSoup=bs4.BeautifulSoup(res.text, 'lxml')
type(ugSoup)
# The bs4.BeautifulSoup() function is called with a string containing the HTML to parse.
# The bs4.BeautifulSoup() function returns a BeautifulSoup() object. This code uses
# requests.get() to download the SCSS undergraduates page and then passes the text
# attribute of the response to bs4.BeautifulSoup(). The BeautifulSoup object that
# it returns is stored in a variable named ugSoup.

# Once you have a BeautifulSoup object, you can use its methods to locate specific parts of
# an HTML document.

print(ugSoup.prettify()) # We can now print out the HTML content of the page, formatted nicely

list(ugSoup.children) #We can first select all the elements at the top level of the 
# page using the children property of soup. Note that children returns a list generator,
# so we need to call the list function on it:
[type(item) for item in list(ugSoup.children)]
 

## Attempt 1 at Generate a script that will extract all the URLs that are listed in the
## https://scss.tcd.ie/undergraduate webpage. 


## Finding all instances of a tag at once 
ugSoup.find_all('a')
ugSoup.find_all('a')[0].get_text()
ugSoup.find_all('a')[0].has_attr('href')
print(ugSoup.find_all('a')[10]['href'])
#[a.get_text() for a in list(ugSoup.find_all('a'))]


## Attempt 2:  Below code will work for the google link but not scss one
import requests
from bs4 import BeautifulSoup, SoupStrainer

content = requests.get('https://www.google.com').content
hrefs = []

for link in BeautifulSoup(content, parse_only=SoupStrainer('a'), features='lxml'):
    if hasattr(link, 'href'):
        print(link['href'])
        hrefs.append(link['href'])



## Attempt 3:
from bs4 import BeautifulSoup
import urllib.request

parser = 'lxml'  # or 'lxml' (preferred) or 'html5lib', if installed
resp = urllib.request.urlopen('https://scss.tcd.ie/undergraduate')
soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))

for link in soup.find_all('a', href=True):
    print(link['href'])



# Write a Python Script to unzip the “Bike Sharing Data Set” from,
# https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset and load the data into
# Pandas DataFrames using the requests.get() function.

# https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip

import requests, zipfile, io, os

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
res = requests.get(url)

try:
    res.raise_for_status()
except Exception as exc:
    print('There was a problem: %s' % (exc))  
else:
    ## Save the zip file in current working directory
    open('Bike-Sharing-Dataset.zip', 'wb').write(res.content)
    #get the current working directory 
    currentDir = os.getcwd()
    # Append name of fiel you want to read to the current directory 
    filesDir = os.path.join(currentDir, 'Bike-Sharing-Dataset.zip') ## obv replace the latter with the relevant file

    zf = zipfile.ZipFile(filesDir) 
    dfDay = pd.read_csv(zf.open('day.csv')) 
    dfHour = pd.read_csv(zf.open('hour.csv'))