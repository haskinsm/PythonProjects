# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:27:18 2021

@author: micha
"""

"""
Given a method which takes two unsorted arrays (A & B) and a number x, create an algorithm 
to merge and fetch the first x numbers of the merged array
"""
import heapq

def mergeSortK (arr1, arr2, k):
    arr1.extend(arr2)
    return heapq.nsmallest(k, arr1, key = lambda value : (value))
    
    
    
    
    
    
    
    
    
    
lis1 = [2,4,6,7,8,10]
lis2 = [1,3,5,6,9]
k = 6
print(mergeSortK(lis1, lis2, k))
lis1.extend(lis2)
