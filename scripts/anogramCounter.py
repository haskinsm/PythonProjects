# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:01:37 2021

@author: micha
"""
'''
Function to return the count of the number of anograms in a string 
'''
from collections import Counter
def sherlockAndAnagrams(s):
    # Write your code here
    # form a list of all possible substrings
    buckets = {}
    for i in range(len(s)):
        for j in range(1,len(s) - i + 1):
            key = frozenset(Counter(s[i:i+j]).items()) # O(N) time key extract
            # frozenset operation generate an unordered sequence. 
            # For example is frozenset({('a', 1), ('b', 2)})  i.e. like abb
            # equals to frozenset({('b', 2), ('a', 1)})       i.e. like bba
            # This is good as means you dont have to sort. Will be able to indentify 
            # that anogram keys should be the same and they 
            # should be incremented by one which is done below.
            # The above example represents the anogram abb and bba. These must be put 
            # in the same key as dict that is why frozen sets are used cause
            # above frozen sets are equal 
            buckets[key] = buckets.get(key, 0) + 1 # Gets the value at index key, 
            # if nothing there gives it a default value of 0 and then add 1 
    count = 0
    for key in buckets:
        count += buckets[key] * (buckets[key]-1) // 2
        # If you have 3 instances of 'ab', the total number of anagram
        # pairs is 2+1 = 3. In general, if you have n instances the 
        # number of anagram pairs is n-1 + n-2 + ... + 1. That simplifies 
        # to n*(n-1)/2  -> the combination formula 
        # n" is the total number of an specific anagram in the String, 
        # and the second argument is "2" because we group them in pairs. 
        # Mathematical simplification: C(n, 2) = n! / 2! * (n - 2)! => 
        # n * (n - 1) * (n - 2)! / (n - 2)! * 2 => n * (n - 1) / 2
    return count

sherlockAndAnagrams("abba") ##==4 [a,a]  [ab, ba]   [b,b] [abb,bba]
sherlockAndAnagrams("cdcd") ##=5
