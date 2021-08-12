# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:58:06 2021

@author: micha
"""
## In Python, a dictionary is wrapped in braces, {}, with a series of key-value 
## pairs inside the braces, as shown below:
car_details = {'model' : 'Volvo xc90', 'colour':'black', 'doors':5}
car_details
# Elements can be accessed and inserted using the same syntax as accessing 
# elements of a list or tuple.
# To get the value associated with a key, give the name of the dictionary and 
# then place the key inside a set of square brackets:
print(car_details['doors'])
# To add a new key-value pair, you give the name of the dictionary followed by 
# the new key in square brackets along with the new value.
car_details['engine'] = 'diesel'
car_details
# To modify a value in a dictionary, give the name of the dictionary with the 
# key in square brackets and then the new value you want associated with that key
car_details['colour'] = 'silver'
# use the del statement to completely remove a key-value pair. To remove the
# key ‘doors’ from car_details
del car_details['doors']
car_details
# You can check if a dictionary contains a key using the same syntax as with 
# checking whether a list or tuple contains a value:
'model' in car_details  ##Boolean returned



cust_details = {} ## initiate empty list
comp = input("Please enter your company: ")
name = input("Please enter your name: ")
email = input("Please enter your email: ")
cust_details['company'] = comp
cust_details['name'] = name
cust_details['email'] = email
cust_details


##loop through all key value pairs:
car_details
for key, value in car_details.items(): 
    ## Could have anything for key, value eg: for key, value in car_details.items():
    print("\nKey: " + key) ##\n is a line break
    print("Value: " + str(value))
    
## loop through all keys in a dictionary:
for key_name in sorted(car_details.keys()):
    print(key_name.capitalize())
    
## loop through all values in a dictionary:
for key_value in (car_details.values()):
    print(str(key_value))
 
## Can put lists in dictionarys:
pizza = {'crust' : 'thin', 'toppings' : ['shrooms', 'bacon']}
print("You ordered a " + pizza['crust'] + "crust pizza with the following toppings:")

for topping in pizza['toppings']:
    print("\t" + topping)    ## "\t" is a horizontal tab space => so indents the toppings list
    
    
    