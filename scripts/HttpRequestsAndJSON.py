# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 19:36:35 2021

@author: micha
"""

###################################### Read in the datasets from Data.Gov.ie #############################
import json
import requests
import pandas as pd
import sys ## For sys.exit()

url_API = "https://api.usmart.io/org/ae1d5c14-c392-4c3f-9705-537427eeb413/29df500e-877c-4146-9f9e-0182004bac7c/1/urql"
try:
    res = requests.get(url_API)
    # requests.get('https://api.github.com/user', auth=('user', 'pass'))
except requests.exceptions.RequestException as e:
    # This should catch ConnectionError, HTTPError, Timeout, TooManyRedirects, and every other kind of possible error 
    print('There was a problem: \n%s' % (e)) #\n is new line and %s is used to add string value (The error 'e') into the string 
    raise sys.exit() ## This will terminate the script execution
   
''' JSON of the form 
  [
    {.....
     }, ....
    {....
     }
  ]
     
'''     

### Now check if request was successful 
try:
    res.raise_for_status()
except Exception as exc:
    print('There was a problem: %s' % (exc))
    raise sys.exit() ## This will terminate the script execution
    
JSONContent = res.json()
content = json.dumps(JSONContent, indent = 4, sort_keys = True)
### Now create a Pandas df from the JSON
jsonData = json.loads(content) ## Change the (JSON) string into a JSON object that can be read into a pd df
test_df = pd.DataFrame.from_dict(jsonData)

# Can also see headers if you want
res.headers
res.headers['content-type']
res.encoding
res.text
res.json()




##################### Reading in JSON ####################
# json.loads() method can be used to parse a valid JSON string and convert it 
# into a Python Dictionary. It is mainly used for deserializing native string, 
# byte, or byte array which consists of JSON data into Python Dictionary.

######### As a Python Dict
import json
# JSON string:
# """ makes it a Multi-line string 
x = """{
    "Name": "Jennifer Smith",
    "Contact Number": 7867567898,
    "Email": "jen123@gmail.com",
    "Hobbies":["Reading", "Sketching", "Horse Riding"]
    }"""
# parse x:
y = json.loads(x)
# the result is a Python dictionary:
print(y)
# Can mess with dict
y['Hobbies']
y['Hobbies'].append('Racecar driving')  
y['Hobbies'].pop() ##Pops the itemthat was entred in last 

###### Convert to Pandas df
# Converting from dict made above 
my_df = pd.DataFrame.from_dict(y) 
my_df['Name']#[0]
for i in my_df['Name']:
    print(i)
    

####################### Reading in different kinds of JSON  ###############
# Very top example comfortably reads in JSON of format:  [{.....}, ....,{....}]
#JSON of format: { "partners": [ {...}, {....}, ....] } 
c = """{
    "partners":
        [   {"Name": "Jennifer Smith",
            "Contact Number": 7867567898,
            "Email": "jen123@gmail.com",
            "Hobbies":["Reading", "Sketching", "Horse Riding"]
            },
            {"Name": "Timo Smith",
            "Contact Number": 999911111,
            "Email": "timo123@gmail.com",
            "Hobbies":["Jumping", "Dancing", "Skiing"]
            }
            ]

    }"""
# parse c:
z = json.loads(c)
# the result is a Python dictionary:
print(z)
partners = z["partners"] ##This creates an List called partners in Python
print(partners)
print(partners[0])






###################### More on working with JSON #######################
import requests
import json

res = requests.get("https://jsonplaceholder.typicode.com/todos")
var = json.loads(res.text)
# To view your Json data, type var and hit enter
var
# Now our Goal is to find the User who has completed the most tasks!!
#Dict:  # {
            # "userId": 1,
            # "id": 1,
            # "title": "Hey",
            # "completed": false,  
        # }
# Note that there are multiple users with
# unique id, and their task have respective
# Boolean Values.

def find(d):
    ''' Fed in dictionary of tasks interatively from filter and returns true if task is completed '''
    check = d["completed"] #True/False
    return check #and max_var
 
Value = list(filter(find, var)) #Creates list of dictionaries contaning details of tasks that have been completed. 
# Each user can have multiple tasks completed stored in dfferent dictionaries 
# filter(function, sequence)
# function: function that tests if each element of a 
# sequence true or not. Returns an iterator that is already filtered.

# Alternatively could just use list comprehensions:
# Value=[item for item in var if item["completed"] == True]  #Works the same


############## Finds users with max completed tasks
count = {}
def userWithMaxCompleted(d):
    '''Warning: This function only finds one of the users with the max completed value when there can be multiple '''
    for user in d:
        count[user["userId"]] =  count.get(user["userId"], 0) + 1
    # Now get userID who has completed most 
    max_key = max(count, key=count.get) # max(iterable, key=dict.get)
    return max_key

######### Solution 1: Finds only one of the users with max completed tasks
userIdWithMax = userWithMaxCompleted(Value) 
print (userIdWithMax) 
# Get max number a user has completed
print(count[ userIdWithMax ])


########## Solution2 (Better): Finds all users who have max completed tasks 
all_values = count.values()
max_value = max(all_values)
print(max_value)
# Can form a list of users with 
[user for user, numCompletions in count.items() if int(numCompletions) == max_value]





##################### Converting back to JSON ##########################
# json.dumps() converts the Python objects into appropriate json objects.
# It can convert most python objects to JSON
#### From Dict
import json
Dictionary ={(1, 2, 3):'Welcome', 2:'to',
			3:'Geeks', 4:'for',
			5:'Geeks', 6:float('nan')}
# Indentation can be used
# for pretty-printing
json_string = json.dumps(Dictionary,
						skipkeys = True,
						allow_nan = True,
						indent = 6)
print('Equivalent json string of dictionary:\n' + json_string)





##################### HTTP POST Request ##################
r = requests.post('https://httpbin.org/post', data={'key': 'value'})

### Trivial Example
import json
url = 'https://api.github.com/some/endpoint'
payload = {'some': 'data'}
r = requests.post(url, data=json.dumps(payload))  # json.dumps() function converts a Python object into a json string.

### More useful example 
# Instead of encoding the dict yourself, you can also pass it directly using 
# the json parameter (added in version 2.4.2) and it will be encoded automatically:
url = 'https://api.github.com/some/endpoint'
payload = {'some': 'data'}

r = requests.post(url, json=payload)
# Note, the json parameter is ignored if either data or files is passed.
# Using the json parameter in the request will change the Content-Type in the
# header to application/json.





##################### More HTTP GET Request #################
'''
You often want to send some sort of data in the URL’s query string. If you were 
constructing the URL by hand, this data would be given as key/value pairs in the 
URL after a question mark, e.g. httpbin.org/get?key=val. Requests allows you to 
provide these arguments as a dictionary of strings, using the params keyword argument.
As an example, if you wanted to pass key1=value1 and key2=value2 to httpbin.org/get, 
you would use the following code:
'''

payload = {'key1': 'value1', 'key2': 'value2'}
r = requests.get('https://httpbin.org/get', params=payload)
# You can see that the URL has been correctly encoded by printing the URL:
print(r.url)
## ==> https://httpbin.org/get?key2=value2&key1=value1

# Note that any dictionary key whose value is None will not be added to the URL’s
# query string.

# You can also pass a list of items as a value:
payload = {'key1': 'value1', 'key2': ['value2', 'value3']}
r = requests.get('https://httpbin.org/get', params=payload)
print(r.url)

#### This stuff is important as often need to pass in things like API key
payload = {'userKey': 'ffce024f79c824a16a38260b5178'} ##i.e. The API key
r = requests.get('https://candidate.hubteam.com/candidateTest/v3/problem/dataset', params=payload)
print(r.url)
# ==> https://candidate.hubteam.com/candidateTest/v3/problem/dataset?userKey=ffce024f79c824a16a38260b5178



