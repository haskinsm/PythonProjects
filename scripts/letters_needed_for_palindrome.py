# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 21:14:11 2021

@author: micha
"""

def solution(word):
    """
    Thinking: For a palindrome it appears that there needs to be only one odd amount of a 
    char and all the other chars need to have an even count
    e.g. abcb => b = 2  a,c =1 so only need to add one more letter of either a or c
    """
    counter = {}
    numLettersNeeded = 0
    
    # Count occurences of each letter
    for char in word:
        if char in counter:
            counter[char] += 1
        else:
            counter[char] = 1
    
    # The below loop will count all the letters that do not have even counts 
    # will need to decrement this by one as want to have one letter with an odd count. 
    for char, count in counter.items():
        if( count % 2 != 0):
            numLettersNeeded += 1
            
    # Decrement numLettersNeeded as have over counted by one the letters I 
    # need to form a palindrome         
    if(numLettersNeeded > 0):
        numLettersNeeded -= 1
            
    return numLettersNeeded
    
a = "abcb" # need one letter for palindrome -> abcba
print(solution(a))            
