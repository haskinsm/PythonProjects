# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:31:22 2021

@author: micha
"""

def twoSum(arr, sum):
    numsNeeded = {}
    output = []
    for num in arr:
        if num in numsNeeded:
            output.append([sum-num,num])
            
        numberNeed = sum - num
        numsNeeded[numberNeed] = num
        
    return output


input = [2,9,-1,42,6,4]
sum = 8
output = [2,6], [9,-1]

print(twoSum(input, sum))


######################## Nathan I did the below bit in a sort of weird way, sort of forgot how to do 
# binary search and needed a bit of help from the guy
# Normal binary search looks a lil diff

def twoSum2(arr, sum):
    arr.sort()
    output = []
    
    for index, num in enumerate(arr):
        numberNeed = sum - num 
        output.extend(recursiveFunc(arr, numberNeed))   
    return output
                
def recursiveFunc(arr, numberNeed):
    """
    Binary search function, will return a list od pairs that sum to a passed target called numberNeed
    """
    middleInd = len(arr)//2
    middleElem = arr[middleInd]
    output = []
    
    if(middleElem == numberNeed):
        output.append(sum-numberNeed, middleElem) 
    if(middleElem < numberNeed):
        recursiveFunc(arr[middleInd+1:], numberNeed)  
    else:
        recursiveFunc(arr[:middleElem-1], numberNeed)
    return output
    
    

    
    
    
    
                

    