# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:38:43 2021

@author: micha
"""

## Debug mode in python is the big play & pause symbol
    ## Can press ctrl+F5 either 
# starts the IPython debugger (ipdb). After doing that, the Editor pane will 
# highlight the line that is about to be executed, and the Variable Explorer 
# (top right of IDE) will display variables in the current context of the 
# point of program execution. 

## Step button is beside the debug mode button or press ctrl+F10

## Inspect how a particular function is working by stepping into it with the Step into
## button or the shortcut Ctrl+F11

## To get out of a function and continue with the next
## line you need to use the Step return button or the shortcut Ctrl+Shift+F11.

## Set a breakpoint by pressing F12 or double-clicking in the grey area to the left
## of a line of code as shown below. The breakpoint is indicated by a red dot
## the code prior to the breakpoint will run and debugging will only begin
## at the breakpoint.

# Simple code to practice debug:
age = int(input("Please enter your age: "))
    
if age > 13 and age < 20:
    print("You are a teenager")
else:
    print("You are not a teenager")
        
    
## Error Handling:
# You can use error handlers in any situation where an action (either expected 
# or unexpected) has the potential to produce an error that stops program execution. 

# Exceptions are handled with try-except blocks. A try-except block tells Python 
# what to do if an exception is raised. When you use try-except blocks, your 
# programs will continue running even if things start to go wrong. 
# The code to catch the ZeroDivisionError is as follows:
try: 
    print(5/0) ##Obv can't divide by zero so expect error traceback but 
    # with excpetion handling this wont kill the entire program
except ZeroDivisionError: ## I believe there are a number of named errors that you can handle
    print("You can't divide by zero!")
    
    
quit = False
while (quit == False):
    num1 = input("Please enter your numerator (or q to quit): ")
    if( num1 == 'q'):
        quit = True
    else:
        num2 = input("Please enter your denominator (or q to quit): ")
        if( num1 == 'q'):
            quit = True
        else:
            try: 
                num1 = int(num1) ## cast to int
                num2 = int(num2)
                print(num1/num2) 
            except ZeroDivisionError: 
                print("Your denominator can't be zero. Its impossible to divide by zero!")
            except ValueError:
               print("Non-numeric data entered. Please try again.")
          
## Exception handling in python:
    ## https://docs.python.org/3/tutorial/errors.html
    
    
    
    
