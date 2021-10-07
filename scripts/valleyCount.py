# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 21:42:18 2021

@author: micha
"""

# Count number of valleys
path = 'UDDDUDUU'
height = 0
valleys = 0
notBelowS = True; 
for step in path:
    #Change height
    if step == 'U':
        height +=1
    elif step == 'D':
        height -= 1
        
    #If height is 0 and notBelowS is false youve just come up from a valley so increment valleys variable
    if( notBelowS == False and height == 0):
        valleys += 1
        
    # Change notBelowS var to True if above or at sea level..etc
    if height >= 0: 
        notBelowS = True
    else:
        notBelowS = False
  
   

## Another practice test stuff
test = [1,4,2,3,3,4,5]
a = set(test)   #Set sorts and removes duplicates, need to convert back to list using list func tho
b = list(a)


## Fucked Amazon Q 
def rev(arr):
    result = []
    for i in arr:
        result.insert(0,i)
    return result

l=[1,2,3,4,5,6]
a=0
b=4
c=[*l[0:a],*rev(l[a:b+1]),*l[b+1:]]
print(c)

rev(l[1:4])


