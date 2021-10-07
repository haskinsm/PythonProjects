# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:12:17 2021

@author: micha
"""
x = 3
if x <0:
    print ('it is less than zero')
elif x == 0:
    print('iszero')
elif 0 < x < 5:
    print('is between 1 and 4')
else:
    print('is positive and greater than or equal to 5')
  
    
  
    
    
sequence = [1, 2, None, 4, None, 5]
total = 0
for value in sequence:
    if value is None:
        continue
    total += value

print(total)






quit = False

while quit == False:
    age = input("Please enter your age: ")
    
    
    if  age.isnumeric() == False:
        quit = True
        print("Thank you! Goodbye.")
    else:
        age = int(age) ## Need to cast from string to int
        if age < 3:
            print("Your ticket is free. Enjoy!")
        elif age <= 12:
            print("Your ticket is €9")
        elif age >= 65:
            print("Your ticket is €10. Enjoy!")
        else:
            print("Your ticket is €15. Enjoy!")
        
        

for value in range(2,20,2):
    print(value)
  
    
sum_even = 0
for x in range(2, 100, 2):
    sum_even += x
print(sum_even)




a_list = [2, 12, 7, None]
veggies = ('carrots', 'onions', 'turnips') ## Create a tuple using brackets, cant change items in a tuple


print(a_list)

a_list.append(4)

a_list.insert(0, 33)

print(a_list)

a_list.pop(3)
a_list.remove(33)
print(a_list)

4 in a_list ## Checks if in list

a_list.extend([55, 67])

veggies.sort()

veggies.sort(reverse = True)

a_list[0:2]

a_list[:4]
a_list[2:]
a_list[-4:] ## Negative slices relative to end of list
a_list[:-2] ## Everything but the last two elements


strings = ['a', 'as', 'bat', 'car', 'like', 'sweet']

[newStrings.upper() for newStrings in strings 
    if len(newStrings) > 2]
        


dimensions = (200,50) ## Tuple cant alter once created
dimensions[0]


exit = False
num = int( input("Please enter the number of daily temperatures you want to record: ") )
i = int(0)
temp = []
while (exit == False):
    i += int(1)
    if( i == num):
        exit = True
    a = input("Please enter temperature or quit to exit: ")
    if a.isnumeric() == False:
        exit = True
    else:
        temp.append( int(a) ) ## Cast string to int
    

answer  = sum(temp) / len(temp)
print( "Avergae temp: ", answer )
