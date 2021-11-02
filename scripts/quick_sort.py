# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:33:00 2021

@author: micha
"""

"""
Recursive
In place
Unstable
O(nlogn)

In this algorithm we partition the list around a pivot element, sorting values around the pivot.
In my solution I used the the last element from the list as pivot value. Best performance is
achieved when the pivot value splits the list in two almost equal halves.

src: https://towardsdatascience.com/sorting-algorithms-with-python-4ec7081d78a1
"""
import random 

def quickSort(array):
    if len(array)> 1:
        random.shuffle(array)  ## Shuffle in place, reduces chance of worst case 
        pivot=array.pop()
        grtr_lst, equal_lst, smlr_lst = [], [pivot], []  ## Note this is setting equal_lst = [pivot]   So the piv element is not lost 
        for item in array:
            if item == pivot:
                equal_lst.append(item)
            elif item > pivot:
                grtr_lst.append(item)
            else:
                smlr_lst.append(item)
        return (quickSort(smlr_lst) + equal_lst + quickSort(grtr_lst))
    else:
        return array
    
aList = [2,4,1,12,4,19,1,3,8,40,90,1,0]
sortedArr = quickSort(aList)
print(sortedArr)


