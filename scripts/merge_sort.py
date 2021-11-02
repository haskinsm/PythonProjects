# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:31:09 2021

@author: micha
"""

"""
Recursive
Stable
Needs extra space
O(nlogn)


This is a divide and conquer algorithm. In this algorithm we split a list in half, and 
keeps splitting the list by 2 until it only has single element. Then we merge the sorted sorted list.
We keep doing this until we get a sorted list with all the elements of the unsorted input list.
"""

def mergeSort(nums):
    if len(nums)==1:
        return nums
    mid = (len(nums)-1) // 2
    lst1 = mergeSort(nums[:mid+1])
    lst2 = mergeSort(nums[mid+1:])
    result = merge(lst1, lst2)
    return result

def merge(lst1, lst2):
    lst = []
    i = 0
    j = 0
    while(i<=len(lst1)-1 and j<=len(lst2)-1):
        if lst1[i]<lst2[j]:
            lst.append(lst1[i])
            i+=1
        else:
            lst.append(lst2[j])
            j+=1
    if i>len(lst1)-1:
        while(j<=len(lst2)-1):
            lst.append(lst2[j])
            j+=1
    else:
        while(i<=len(lst1)-1):
            lst.append(lst1[i])
            i+=1
    return lst

aList = [2,4,1,12,4,19,1,3,8,40,90]
sortedArr = mergeSort(aList)
print(sortedArr)