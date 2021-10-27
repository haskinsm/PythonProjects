# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:29:03 2021

@author: micha
"""

"""
Non-recursive
Unstable
In place
O(nlogn)

Heap Sort
Like last two previous algorithms we create two segments of the list one sorted and one unsorted. 
In this we use heap data structure to efficiently get the max element from the unsorted segment of
the list. Heapify method uses recursion to get the max element at the top.

"""

def heapify(array, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    
    if l < n and array[i] < array[l]:
        largest = l
    if r < n and array[largest] < array[r]:
        largest = r
    
    if largest != i:
        array[i], array[largest] = array[largest], array[i]
        heapify(array, n, largest)
        
def heapSort(array):
    n = len(array)
    for i in range(n//2, -1, -1):
        heapify(array, n, i)
    for i in range(n-1, 0, -1):
        array[i], array[0] = array[0], array[i]
        heapify(array, i, 0)
    return array

aList = [2,4,1,12,4,19,1,3,8,40,90]
sortedArr = heapSort(aList)
print(sortedArr)