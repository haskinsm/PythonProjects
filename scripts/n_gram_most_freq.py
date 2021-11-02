# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:39:33 2021

@author: micha
"""

"""
An N-gram is a sequence of N consecutive characters from a given word. For the word "pilot" there are 
three 3-grams: "pil", "ilo" and "lot". For a given set of words and an n-gram length Your task is to
• write a function that finds the n-gram that is the most frequent one among all the words
• print the result to the standard output (stdout)
• if there are multiple n-grams having the same maximum frequency please print the one that is the 
smallest lexicographically (the first one according to the dictionary sorting order)

Note that your function will receive the following arguments:
• text
    ○ which is a string containing words separated by whitespaces
• ngramLength
    ○ which is an integer value giving the length of the n-gram
Data constraints

• the length of the text string will not exceed 250,000 characters
• all words are alphanumeric (they contain only English letters a-z, A-Z and numbers 0-9)
"""
import heapq
from collections import defaultdict

def nGram (myStr, n):
        words = myStr.split(' ')
        counter = defaultdict(int)
        for word in words:
            numNGrams = len(word) - n + 1
            if( numNGrams <= 0):
                continue 
            for i in range(0, numNGrams):
                j = i + n
                nGram = word[i:j]
                counter[nGram] += 1
                
        maxNGram = heapq.nsmallest(1, counter, key = lambda count : (~counter[count], count))[0]
        maxCount = counter[maxNGram]
        return ("The max ngram is {} with a frequency of {}".format(maxNGram, maxCount))
            #tom   1
            #toms->tom oms   2
            #fives->fiv ive ves      3
            #tenner->ten enn nne ner        4
            #tenners-> ... ers                     5 n-grams so numNGrams = len(word) - n + 1  where n is the length of the nGrams
            
        # Worst case is we have one huge word, with no spaces. This is going to lead to a lot of ngrams
        # O((m-n)log(m-n))   #where n is the length of ngrams and m is the length of the string (i.e. the big word)
        # m - n - 1 is the number of n-grams but we ignore the -1
        # Heapsort has a worst case run time of nlogn so replace n with m-n here. This is the case where we have all unique n-grams
        # 
 
nGramLength = 3
myStr = "aaaab a0a baaab c"  #Output aaa ngramLength: 3
a= nGram(myStr, nGramLength)
