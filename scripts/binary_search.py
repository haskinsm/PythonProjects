# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 19:28:10 2021

@author: micha
"""

def binarySearch(arr, left, right, num):
    """
    Takes in sorted array, the startign left index and the ending right index to search between, and
    the number to search for and returns index of that number in arr
    If can't find the number returns -1
    """
    while left <= right:
        middleIndex = (left + right)//2
        middleElem = arr[middleIndex] 
        if(num == middleElem):
            return middleIndex
        elif(num < middleElem):
            return binarySearch(arr, left, middleIndex - 1, num)
        else:
            return binarySearch(arr, middleIndex + 1, right, num)
           
    return -1

    
    
    
    
myArr = [2,3,4,5,6,7,8,9,10,20,30] 
getIndexOf = 8
low, high = 0, len(myArr) - 1
index = binarySearch(myArr, low, high, getIndexOf)
print(myArr[index])
