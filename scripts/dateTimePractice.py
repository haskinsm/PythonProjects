# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:12:13 2021

@author: micha
"""

from datetime import datetime 
## Could do from datetime import datetime as dt 
## Could then call datetime by dt
import calendar

## Click on datetime and then ctrl + i to get the help file
now = datetime.now()
now
## Stores date and time down to milisecond

print("Today is", now.strftime("%A, %d %B %Y") )
## Also outputs "Tuesday": calendar.day_name[ now.weekday() ]



from dateutil import parser 

birthdate = parser.parse(input("Enter date: "))
birthdate ## = datetime.datetime(1999, 10, 24, 0, 0)
age = ((now - birthdate).days)/365
age

import math
print(math.floor(age),  "years old")


?age
age?

# ?parser
# ??parser ## double ?? shows src code
