# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 21:24:16 2021

@author: micha
"""

def minimumSwaps(arr):
    ''' Just go through the array iteratively putting each element in its right position
        If think about it every out of position number is going to require one 
        swap unless the out of position number is in the position of the number 
        that is in its correct position. In which case this algorithm would be 
        able to deal with it and only one stop would be recorded.  '''
    swaps = 0
    i = 0
    #Cant use for i in range(len(arr)) as I resets each time, i.e. cant decrement it 
    while i < len(arr) - 1:   
        i += 1
        num = i+1 #As index starts at 0 need to add 1
        #Check if not in right position 
        if num != arr[i]:
            swaps += 1
            # Now swap the numbers so arr[i] value is in its correct postion at 
            # index temp-1
            temp = arr[i]
            arr[i] = arr[temp-1] #temp-1 as need to adjust since index starts at 0
            arr[temp-1] = temp 
            # Decrement i 
            i -= 1 #as need to make sure the other number that you 
            # just swapped to index i is in the right position so decrement i 
    return swaps

minimumSwaps([4,3,1,2])
minimumSwaps([1,3,5,2,4,6,7])