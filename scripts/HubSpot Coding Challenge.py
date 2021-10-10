# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:28:32 2021

@author: micha
"""
import json
import requests
import pandas as pd
import sys ## For sys.exit()
from collections import defaultdict
from datetime import datetime


#####################################################################################################
###########  Read in JSON using HTTP get request using the requests package in Python ###############
#####################################################################################################

urlAPI = "https://candidate.hubteam.com/candidateTest/v3/problem/dataset?userKey=d30928a61a998622889e1d58f45f"
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

# The above dict contains one key value pair. The key being 'partners'. 
# Create a partners list from JSON 
partners = jsonData['partners']





#########################################################################################################
########### Now create a dict where they key is a country and value is a list of partners ###############
#########################################################################################################

# Use default dict to avoid keyErrors when appending partners to list
partnersByCountry = defaultdict(list) # Create default dict of type list

# Now iterate through every partner dict in partners list. Append the partner dict to 
# partnersByCountry where the key of partnersByCountry is the partners country, which is gotten
# from the partner dict using partner['country']
for partner in partners: 
    country = partner['country'] 
    # Now add partner to country
    partnersByCountry['{}'.format(country)].append(partner)
    
    
    
    

##############################################################################################
######### Now for each country create a dict where the keys are dates available and the values 
######### will be the list of partners who can attend for two days from this date. Will then 
######### be able to form the answer to the coding test and store in a dict called countries 
##############################################################################################


# Create countries dict which will store results to be returned in HTTP Post request
countries = {'countries':[]}

def startDateAndMaxAttendees(d):
    ''' Function to find maxAttendees and start date when passed dict of datesAvailable, where
        the key is the date and the value is a list of partners available to attend were the 
        event to start on that date. If there is a multiple dates with the same number of max 
        attendees it will return the earliest'''
    maxAttendees = 0
    startDate = None
    for date, partners in d.items():
        if(len(d[date]) > maxAttendees):
            maxAttendees = len(d[date])
            startDate  = date
    return startDate, maxAttendees                    
        
    
for country, partners in partnersByCountry.items():
    # Use default dict to avoid keyErrors when appending
    datesAvailable = defaultdict(list) # This will also reset the list every loop
    # The values in this dict for say the 10/10/21 will be all the partners in this 
    # country who can attend the event if it were to start today
    # i.e. they are free for the next two days
    
    # Now iterate through the list of partners for this country 
    for partner in partners:
        availableDates = partner['availableDates']
        # This assumes that available dates are sorted, which they appear to be in every case
        for i in range(len(availableDates) - 1):  #-1 as dont need to check the last date, as there are no more dates after it
            # so it cant be a start date for the event. 
            
            # Need to convert the string dates to date objects in python use the datetime package to do this 
            date1 = datetime.strptime(availableDates[i], "%Y-%m-%d")
            date2 = datetime.strptime(availableDates[i+1], "%Y-%m-%d") ##This date should be after date1 as dates are sorted
            
            # Check if dates are consecutive (i.e. date2 is the next day after date1)
            if[(date1-date2).days == -1]: #Use .days to get the difference in days between the dates. Want this to equal -1
                # Enters here if partner is available to attend event when start date is date1. 
                # Now append partner dict to list of partner dicts in availableDates where the key (starting date)
                # is date1. 
                datesAvailable[availableDates[i]].append(partner)
                
    # Now calculate when the event should start by finding when the most partners can attend 
    # If draw chooses earliest start date 
    startDate, maxAttendees = startDateAndMaxAttendees(datesAvailable)
    
    # Create resultDict to be added to countries dict. This dict should contain the 
    # attendeeCount, the list of emails of attendees, the name of the country, and 
    # the startDate
    resultDict = {}
    resultDict['attendeeCount'] = maxAttendees
    resultDict['attendees'] = [partner['email'] for partner in datesAvailable[startDate]]
    resultDict['name'] = country
    resultDict['startDate'] = startDate
    # Now add resultDict to countries dict
    countries['countries'].append(resultDict)
    

    
#########################################################################################################
###################### Now Post the answer, being the countries dict, in JSON back to API ###############
#########################################################################################################

postUrl = "https://candidate.hubteam.com/candidateTest/v3/problem/result?userKey=d30928a61a998622889e1d58f45f"

# Convert back to JSON from Python Dict using json.dumps function
json_string = json.dumps(countries)
# Send post reuqest to API
r = requests.post(postUrl, data=json_string)

### Check if request was unsuccessful 
try:
    r.raise_for_status()
except Exception as exc:
    print('There was a problem: %s' % (exc))
    
### Now check if request was successful
if r.status_code == 200: #requests.codes.ok:
    print("All good")
    
    
    
    
    