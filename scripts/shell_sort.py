# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:26:08 2021

@author: micha
"""

"""
Non-recursive
Stable
In place
0(n²) also depends on interval choice

Shell sort is an optimization over insertion sort. This is achieved by
repeatedly doing insertion sort on all elements at fixed, decreasing intervals.
Last iteration the interval is 1. Here it becomes a regular insertion sort and 
it guarantees that the array will be sorted. But to note one point is that by the
time we do that array is almost sorted, hence the iteration is very fast.
"""

import math 

def shellSort(array):
    n = len(array)
    k = int(math.log2(n))
    interval = 2**k -1
    while interval > 0:
        for i in range(interval, n):
            temp = array[i]
            j = i
            while j >= interval and array[j - interval] > temp:
                array[j] = array[j - interval]
                j -= interval
            array[j] = temp
        k -= 1
        interval = 2**k -1
    return array

aList = [2,4,1,12,4,19,1,3,8,40,90]
sortedArr = shellSort(aList)
print(sortedArr)