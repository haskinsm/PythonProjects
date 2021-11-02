# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:16:37 2021

@author: micha
"""
"""
Non recursive
Unstable
In place
O(n²)
src: https://towardsdatascience.com/sorting-algorithms-with-python-4ec7081d78a1
"""
def selectionSort(array):
    for i in range(len(array)):
        min_idx = i
        for idx in range(i + 1, len(array)):
            if array[idx] < array[min_idx]:
                min_idx = idx
        array[i], array[min_idx] = array[min_idx], array[i]
    return array

aList = [2,4,1,12,4,19,1,3,8,40,2]
sortedArr = selectionSort(aList)
print(sortedArr)

