# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 22:08:02 2021

@author: micha
"""

'''
Count char occurences in string. Case sensitive. 
'''
word = "mississippi"
counter = {}

for letter in word:
    if letter not in counter:
       counter[letter] = 0
    counter[letter] += 1
counter


'''
Case sensitive counter of words in a sentence. Can use to check if words in another 
string were in this string 
'''
string = "Hello how are you. Are you ok."
counter2 = {}
#Remove everything that is not an alphabet char
onlyAlpha = ""
for char in string:
    if char.isalpha() or char == " ":   #char.isspace() works either  
        onlyAlpha += char

for word in onlyAlpha.split():
    if word not in counter2:
        counter2[word] = 0
    counter2[word] += 1
counter2

#Check if below string was in other string
contains = True
note = "Hello you!"
onlyAlpha = ""
for char in string:
    if char.isalpha() or char == " ":   #char.isspace() works either  
        onlyAlpha += char

for word in onlyAlpha.split():
    if word in counter2:
        counter2[word] -= 1
    else: 
        contains = False
print(contains)
     

'''
Use Counter package to count the occurences of certain words 
'''
from collections import Counter 
magazine = "Hello how are you?"
note = "Hello you. Hello"
# Count occurences of chars
# Can use string or list as an arg
Counter(list(note))
Counter(note) 

def stripNonAlpha(string):
    onlyAlpha = ""
    for char in string:
        if char.isalpha() or char == " ":   #char.isspace() works either  
            onlyAlpha += char
    return onlyAlpha

# Count occurences of only words. Case sensitive. Get only alpha chars and then split on whitespace
note = stripNonAlpha(note).split() 
magazine = stripNonAlpha(magazine).split()
Counter(note) 
print ((Counter(list(note)) - Counter(list(magazine))) == {})
# prints false if words in note cannot be madue up from words in magazine 
# prints true if they can 
    

'''
Another cool use of counter 
'''
from collections import Counter

prices = {"course": 97.99, "book": 54.99, "wallpaper": 4.99}
cart = Counter(course=1, book=3, wallpaper=2)

for product, units in cart.items():
    subtotal = units * prices[product]
    price = prices[product]
    print(f"{product:9}: ${price:7.2f} × {units} = ${subtotal:7.2f}")