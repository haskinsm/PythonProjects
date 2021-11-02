# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:21:49 2021

@author: micha
"""

"""
Non-recursive
Stable
In place
O(n²)  when the array is not sorted
Best case is O(n) when the array is sorted already
"""

def insertionSort(array):
    for i in range(1, len(array)):
        key = array[i]
        j = i-1
        while array[j] > key and j >= 0:
            array[j+1] = array[j]
            j -= 1
        array[j+1] = key
    return array

aList = [2,4,1,12,4,19,1,3,8,40]
sortedArr = insertionSort(aList)
print(sortedArr)



for i in range(1, len(aList)):
    print(i)
print(len(aList))

def myInsertionSort(array):
    for i in range(1, len(array)):
        j = i - 1
        curVal = array[i]
        while(curVal < array[j] and  j>=0):
            array[j+1] = array[j]
            j -= 1
        array[j+1] = curVal
    return array
myInsertionSort(aList)
