# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:46:22 2021

@author: micha
"""

"""
class SklearnHelper that allows one to extend the inbuilt methods (such as train, predict and fit) 
common to all the Sklearn classifiers. Therefore this cuts out redundancy as won't need to write the
same methods five times if we wanted to invoke five different classifiers.
"""
import sklearn

class SklearnHelper(object):
    """
    Class to extend the Sklearn classifier
    """
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return (self.clf.fit(x,y).feature_importances_)