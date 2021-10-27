# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:11:42 2021

@author: micha
"""

def twoSum(nums, target):
        
    ## use a dict, known as a map (or hash tables) in other languages
    ## As this is the best way to check we have a num 
    ## dicts are hash tables. No tree searching is used. Looking up a key 
    ## is a nearly constant time operation, regardless of the size of the dict. O(1)
    numsNeeded = {}
    for index, num in enumerate(nums):
        # Check if the current num is a number needed to get the target and if it is not needed
        # then add the number that one be needed to sum with this one to reach the target to the dict
        # where the value is the index of a number and the key is the number 
        if num in numsNeeded:
            return [numsNeeded[num],index]
        else:
            numsNeeded[target-num] = index
            
    ## Told valid solution exists but if not output -1 -1
    return [-1,-1]

twoSum([2,11,15,7], 9)