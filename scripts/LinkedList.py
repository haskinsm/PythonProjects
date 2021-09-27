# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:54:51 2021

@author: micha
"""

# Node class
class Node:
      # Function to initialize the node object
    def __init__(self, data):
        self.data = data  # Assign data
        self.next = None  # Initialize
                          # next as null
  
# Linked List class
class LinkedList:
    # Function to initialize the Linked
    # List object
    def __init__(self):
        self.head = None
        
    def printList(self):
        ''' Function to print contents of linked list starting from head'''
        temp = self.head
        while (temp):
            print (temp.data)  ## print (" %d" %(temp.data))  #Will print it on the same line
            temp = temp.next
            
    def push(self, newData):
        ''' Function to insert new node at beginning'''
        newNode = Node(newData)
        newNode.next = self.head 
        # Now need to change the head to point to new Node
        self.head = newNode

    def insertAfter(self, prev_node, new_data):
        ''' Inserts a new node after the given prev_node'''
        # check if the given prev_node exists
        if prev_node is None:
            print ("The given previous node must be in LinkedList.")
            return
        # Create new node & Put in the data
        new_node = Node(new_data)
        # Make next of new Node as next of prev_node
        new_node.next = prev_node.next
        # Make next of prev_node as new_node
        prev_node.next = new_node
     
    def append(self, newData):
        '''Add node to end of list'''
        newNode = Node(newData)
        # Check if empty list
        if self.head is None:
            self.head = newNode
        # Otherwise Traverse the list
        last = self.head
        while(last.next):
            last = last.next
        #Now should be at last so change lasts pointer to new node
        last.next = newNode
        

    def deleteNode(self, key):
        ''' Given a reference to the head of a list and a key,
         delete the first occurrence of key in linked list '''
        # Store head node
        temp = self.head
        # If head node itself holds the key to be deleted
        if (temp is not None):
            if (temp.data == key):
                self.head = temp.next
                temp = None
                return
        # Search for the key to be deleted, keep track of the
        # previous node as we need to change 'prev.next'
        while(temp is not None):
            if temp.data == key:
                break
            prev = temp
            temp = temp.next
        # if key was not present in linked list
        if(temp == None):
            return
        # Unlink the node from linked list
        prev.next = temp.next
        temp = None
        
    def deleteNode(self, position):
        '''  delete the node at a given position '''
        # If linked list is empty
        if self.head == None:
            return
        # Store head node
        temp = self.head
        # If head needs to be removed
        if position == 0:
            self.head = temp.next
            temp = None
            return
        # Find previous node of the node to be deleted
        for i in range(position -1 ):
            temp = temp.next
            if temp is None:
                break
        # If position is more than number of nodes
        if temp is None:
            return
        if temp.next is None:
            return
        # Node temp.next is the node to be deleted
        # store pointer to the next of node to be deleted
        next = temp.next.next
        # Unlink the node from linked list
        temp.next = None
        temp.next = next
     
    def search(self, li, key):
        ''' Checks whether the value key is present in linked list. Takes in head of list, and key youre looking for'''
        # Base case
        if(not li):
            return False
        # If key is present in 
        # current node, return true
        if(li.data == key):
            return True
        # Recur for remaining list
        return self.search(li.next, key)
    
# Code execution starts here
if __name__=='__main__':
    # Start with the empty list
    llist = LinkedList()
    llist.head = Node(1)
    llist.head.next = Node(2)
    
    #Can also do like
    third = Node(3)
    llist.head.next.next = third 
    third.next = Node(4)
    
    llist.printList()
    llist.push(6)
    llist.append(5)
    llist.insertAfter(llist.head.next, 8)
    llist.printList()

    if llist.search(llist.head,4):
        print("Yes")
    else:
        print("No")