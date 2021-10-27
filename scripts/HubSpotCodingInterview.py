# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:31:22 2021

@author: micha
"""

import json
import requests
import sys ## For sys.exit()

#####################################################################################################
###########  Read in JSON using HTTP get request using the requests package in Python ###############
#####################################################################################################

urlAPI = ""
try:
    res = requests.get(urlAPI)
except requests.exceptions.RequestException as e:
    # This should catch ConnectionError, HTTPError, Timeout, TooManyRedirects, and every other kind of possible error 
    print('There was a problem: \n%s' % (e)) #\n is new line and %s is used to add string value (The error 'e') into the string 
    raise sys.exit() ## This will terminate the script execution


################ Parse JSON
### JSON is of form :{"partners":[{},..,{}]}
JSONContent = res.json() 
content = json.dumps(JSONContent, indent = 4, sort_keys = True)

### Now create dict from the JSON string using json.loads() fucntion 
jsonData = json.loads(content) 


