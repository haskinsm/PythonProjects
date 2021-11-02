# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:37:13 2021

@author: micha
"""

import collections, heapq


def topKFrequent(words, k):
    Freqs = collections.Counter(words)
    return heapq.nsmallest(k, Freqs,
        key=lambda word:(~Freqs[word], word)
    )
# nsmallest() returns the [:k] of the result.
# key is a callable function that determines how elements are compared.
    # word could be replaced by x (is just more descirptive)
    # ~Freqs[word] means 'rank the frequencies of words in descending order'. Remove the ~ and it will sort in asc order
    # word means 'rank the words with the highest frequencies in their alphabetical order' (the 
    #   second sorting criteria in lambda function)

"""
Think about this example: (1, 'b'), (2, 'a'), (3, 'c'), (1, 'd'), (4, 'c') # (Freq, word)
heapq.nlargest(k, Freqs, key=lambda word:(Freqs[word], word))
-- > (1, 'b'), (1, 'd'), (2, 'a'), (3, 'c'), (4, 'c')
--> (3, 'c'), (4, 'c') (wrong order)

heapq.nsmallest(k, Freqs, key=lambda word:(~Freqs[word], word)
-- > (4, 'c'), (3, 'c'), (2, 'a'), (1, 'b'), (1, 'd'),
--> (4, 'c'), (3, 'c') (correct order)

The time complexity of heapsort is O(nlogn)
"""
words = ["the","day","is","sunny","the","the","the","sunny","is","is"] 
k = 4

ans = topKFrequent(words, k)
print(ans)


def topKFrequent2(words, k):
    counts = collections.Counter(words)
    items = list(counts.items()) ##[('the', 4), ('is', 3), ('sunny', 2), ('day', 1)]
    items.sort(key=lambda item:(-item[1],item[0]))
    ## return heapq.nsmallest(k, items, key = lambda item: (-item[1], item[0])) ##Also workds
    return [item[0] for item in items[0:k]]
"""
(1) Use collections.Counter to collate
(2) create list of touples containing (word,count)
(3) Sort by the count and secondary by the word. Note that by negating the count we sort from highest count to lowest instead of the other way around. (Note also that you can't just do a reverse sort or the words themselves would be the wrong way around.)
(4) Strip off and return a list of the first K words.
"""

words = ["the","day","is","sunny","the","the","the","sunny","is","is"] 
k = 4

ans = topKFrequent2(words, k)
print(ans)


a = [3, 5, 1, 2, 6, 8, 7]
heapq.heapify(a)
a

import heapq
results="""\
Christania Williams      11.80
Marie-Josee Ta Lou       10.86
Elaine Thompson          10.71
Tori Bowie               10.83
Shelly-Ann Fraser-Pryce  10.86
English Gardner          10.94
Michelle-Lee Ahye        10.92
Dafne Schippers          10.90
"""
top_3 = heapq.nsmallest(
    3, results.splitlines(), key=lambda x: float(x.split()[-1])
)
print("\n".join(top_3))
"""
Here, the key function splits the line by whitespace, takes the last element, and converts it to a 
floating-point number. This means the code will sort the lines by running time and return the three
lines with the smallest running times. These correspond to the three fastest runners, which gives you the
gold, silver, and bronze medal winners.
"""