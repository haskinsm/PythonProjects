# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:18:43 2021

@author: micha
"""

"""
Non recursive
Stable
In place
O(n²)
"""

def bubbleSort(array):
    swapped = False
    for i in range(len(array)-1,0,-1):
        for j in range(i):
            if array[j]>array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
                swapped= True
        if swapped:
            swapped=False
        else:
            break
    return array

aList = [2,4,1,12,4,19,1,3,8,40]
sortedArr = bubbleSort(aList)
print(sortedArr)