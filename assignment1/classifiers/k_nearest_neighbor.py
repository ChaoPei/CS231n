# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np  
  
class KNearestNeighbor(object):  
  """ a kNN classifiers with L2 distance """  
  
  def __init__(self):  
    pass  
  
  def train(self, X, y):  
    """ 
    Train the classifiers. For k-nearest neighbors this is just  
    memorizing the training data. 
 
    Inputs: 
    - X: A numpy array of shape (num_train, D) containing the training data 
      consisting of num_train samples each of dimension D. 
    - y: A numpy array of shape (N,) containing the training labels, where 
         y[i] is the label for X[i]. 
    """  
    self.X_train = X  
    self.y_train = y  
   
    
    
  def predict(self, X, k=1, num_loops=0):  
    """ 
    Predict labels for test data using this classifiers. 
 
    Inputs: 
    - X: A numpy array of shape (num_test, D) containing test data consisting 
         of num_test samples each of dimension D. 
    - k: The number of nearest neighbors that vote for the predicted labels. 
    - num_loops: Determines which implementation to use to compute distances 
      between training points and testing points. 
 
    Returns: 
    - y: A numpy array of shape (num_test,) containing predicted labels for the 
      test data, where y[i] is the predicted label for the test point X[i].   
    """  
    if num_loops == 0:  
      dists = self.compute_distances_no_loops(X)  
    elif num_loops == 1:  
      dists = self.compute_distances_one_loop(X)  
    elif num_loops == 2:  
      dists = self.compute_distances_two_loops(X)  
    else:  
      raise ValueError('Invalid value %d for num_loops' % num_loops)  
  
    return self.predict_labels(dists, k=k)  
  
    
    
    
  def compute_distances_two_loops(self, X):  
    """ 
    Compute the distance between each test point in X and each training point 
    in self.X_train using a nested loop over both the training data and the  
    test data. 
 
    Inputs: 
    - X: A numpy array of shape (num_test, D) containing test data. 
 
    Returns: 
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j] 
      is the Euclidean distance between the ith test point and the jth training 
      point. 
    """  
    num_test = X.shape[0]  
    #print 'X.shape :',X.shape  
    num_train = self.X_train.shape[0]  
    #print 'self.X_train.shape :', self.X_train.shape  
      
    dists = np.zeros((num_test, num_train))  
    for i in xrange(num_test):  
        for j in xrange(num_train):             
            dists[i,j] = np.sqrt(np.sum(np.square(self.X_train[j,:]-X[i,:])))  
    
            return dists  
  
  def compute_distances_one_loop(self, X):  
    """ 
    Compute the distance between each test point in X and each training point 
    in self.X_train using a single loop over the test data. 
 
    Input / Output: Same as compute_distances_two_loops 
    """  
    num_test = X.shape[0]  
    num_train = self.X_train.shape[0]  
    dists = np.zeros((num_test, num_train))  
    for i in xrange(num_test):  
        #print (self.X_train - X[i,:]).shape  
        dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i,:]),axis = 1))  
        
    return dists  
  
  def compute_distances_no_loops(self, X):  
    """ 
    Compute the distance between each test point in X and each training point 
    in self.X_train using no explicit loops. 
 
    Input / Output: Same as compute_distances_two_loops 
    """  
    num_test = X.shape[0]  
    num_train = self.X_train.shape[0]  
    dists = np.zeros((num_test, num_train))   
  
    dists = np.multiply(np.dot(X,self.X_train.T),-2)    
    sq1 = np.sum(np.square(X),axis=1,keepdims = True)    
    sq2 = np.sum(np.square(self.X_train),axis=1)    
    dists = np.add(dists,sq1)    
    dists = np.add(dists,sq2)    
    dists = np.sqrt(dists)    
      
    return dists  
  
    
    
    
  def predict_labels(self, dists, k=1):  
    """ 
    Given a matrix of distances between test points and training points, 
    predict a label for each test point. 
 
    Inputs: 
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j] 
      gives the distance betwen the ith test point and the jth training point. 
 
    Returns: 
    - y: A numpy array of shape (num_test,) containing predicted labels for the 
      test data, where y[i] is the predicted label for the test point X[i].   
    """  
    num_test = dists.shape[0]  
    y_pred = np.zeros(num_test)  
    for i in xrange(num_test):  
        # A list of length k storing the labels of the k nearest neighbors to  
        # the ith test point.  
        closest_y = []  
        
        closest_y = self.y_train[np.argsort(dists[i,:])[:k]]  
      
        y_pred[i] = np.argmax(np.bincount(closest_y))    
 
  
    return y_pred  