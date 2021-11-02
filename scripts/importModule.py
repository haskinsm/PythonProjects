# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:12:17 2021

@author: micha
"""

import random
import selection_sort as ss ## I have a selection_sort class in this repo. 

a = [3,1,5,9,4,2,1]
ss.selectionSort(a)

random.shuffle(a)
print(a)
sorted(a) ## returns a
random.shuffle(a)
a.sort() ## sorts a in place 
print(a)
