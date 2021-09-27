# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:26:50 2021

@author: micha
"""
##Should prob also create a BST class but not rly necessary 
class BSTNode:
    def __init__(self, key=None): ##If no key is provided set it to None (i.e. Null)
        self.val = key
        self.left = None
        self.right = None

    def insert(self, val):
        if not self.val:
            self.val = val
            return
    
        if self.val == val:
            return
    
        if val < self.val:
            if self.left:
                self.left.insert(val)
                return
            self.left = BSTNode(val)
            return
    
        if self.right:
            self.right.insert(val)
            return
        self.right = BSTNode(val)
        
    def delete(self, key):
        """Delete a node with value `key`."""
        if key < self.val: 
            # Find and delete the value in the left subtree.
            if self.left is None:
                # There's no left subtree; the value does not exist.
                raise ValueError("Value not found in tree")
            self.left = self.left.delete(key)
            return self  # current node not deleted, just return
        elif key > self.val: 
            # Find and delete the value in the right subtree.
            if self.right is None:
                # There's no right subtree; the value does not exist.
                raise ValueError("Value not found in tree")
            self.right = self.right.delete(key)
            return self  # current node not deleted, just return
        else:
            # The current node should be deleted.
            if self.left is None and self.right is None:
                # The node has no children -- it is a leaf node. Just delete.
                return None

            # If the node has only one children, simply return that child.
            if self.left is None:
                return self.right
            if self.right is None:
                return self.left

            # The node has both left and right subtrees, and they should be merged.
            # Following your implementation, we find the rightmost node in the
            # left subtree and replace the current node with it.
            parent, node = self, self.left
            while node.right is not None:
                parent, node = node, node.right
            # Now, `node` is the rightmost node in the left subtree, and
            # `parent` its parent node. Instead of replacing `self`, we change
            # its attributes to match the value of `node`.
            if parent.left is node:
                # This check is necessary, because if the left subtree has only
                # node, `node` would be `self.left`.
                parent.left = None
            else:
                parent.right = None
            self.val = node.val
            return self
    
    def search(self, key):
        if self is None: return None  # key not found
        if key< self.key: return self.search(self.left, key)
        elif key> self.key: return self.search(self.right, key)
        else: return self.value  # found key
        
    def get_min(self):
        current = self
        while current.left is not None:
            current = current.left
        return current.val
    
    def get_max(self):
        current = self
        while current.right is not None:
            current = current.right
        return current.val
    
    def exists(self, val):
        if val == self.val:
            return True

        if val < self.val:
            if self.left == None:
                return False
            return self.left.exists(val)

        if self.right == None:
            return False
        return self.right.exists(val)
    
    def preorder(self, vals):
        if self.val is not None:
            vals.append(self.val)
        if self.left is not None:
            self.left.preorder(vals)
        if self.right is not None:
            self.right.preorder(vals)
        return vals

    def inorder(self, vals):
        if self.left is not None:
            self.left.inorder(vals)
        if self.val is not None:
            vals.append(self.val)
        if self.right is not None:
            self.right.inorder(vals)
        return vals

    def postorder(self, vals):
        if self.left is not None:
            self.left.postorder(vals)
        if self.right is not None:
            self.right.postorder(vals)
        if self.val is not None:
            vals.append(self.val)
        return vals
    
# Code execution starts here
if __name__=='__main__':
    nums = [12, 6, 18, 19, 21, 11, 3, 5, 4, 24, 18]
    bst = BSTNode()
    for num in nums:
        bst.insert(num)
    print("preorder:")
    print(bst.preorder([]))
    print("#")

    print("postorder:")
    print(bst.postorder([]))
    print("#")

    print("inorder:")
    print(bst.inorder([]))
    print("#")

    nums = [2, 6, 20]
    print("deleting " + str(nums))
    for num in nums:
        bst.delete(num)
    print("#")

    print("4 exists:")
    print(bst.exists(4))
    print("2 exists:")
    print(bst.exists(2))
    print("12 exists:")
    print(bst.exists(12))
    print("18 exists:")
    print(bst.exists(18))
    