# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import cPickle as pickle
import numpy as np
import os
from pprint import pprint
import matplotlib.pyplot as plt

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    # pprint(datadict.__doc__)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y


def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  num_batch = 5
  for b in range(1, num_batch+1):       # 50000
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))     # 10000
  return Xtr, Ytr, Xte, Yte

  
def prepare_data_CIFAR10(ROOT, num_training=49000, num_validation=1000, num_test=1000):
    
    # Load the raw CIFAR-10 data.
    X_train, y_train, X_test, y_test = load_CIFAR10(ROOT)
    
    # As a sanity check, we print out the size of the training and test data.
    print 'Training data shape: ', X_train.shape     # (50000,32,32,3)
    print 'Training labels shape: ', y_train.shape   # (50000L,)
    print 'Test data shape: ', X_test.shape          # (10000,32,32,3)
    print 'Test labels shape: ', y_test.shape        # (10000L,)
    print
    
    # show_train_CIFAR10(X_train, y_train)

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]                  # (1000,32,32,3)
    y_val = y_train[mask]                  # (1,1000)
    mask = range(num_training)
    X_train = X_train[mask]                # (49000,32,32,3)
    y_train = y_train[mask]                # (1,49000)
    mask = range(num_test)
    X_test = X_test[mask]                  # (1000,32,32,3)
    y_test = y_test[mask]                  # (1,1000)

    # Preprocessing1: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))    # (49000,3072)
    X_val = np.reshape(X_val, (X_val.shape[0], -1))          # (1000,3072)
    X_test = np.reshape(X_test, (X_test.shape[0], -1))       # (1000,3072)
    
    # Preprocessing2: subtract the mean image
    mean_image = np.mean(X_train, axis=0)       # (1,3072)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # show_mean_image(mean_image)
    
    # Bias trick, extending the data
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])    # (49000,3073)
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])          # (1000,3073)
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])       # (1000,3073)
    
    return X_train, y_train, X_val, y_val, X_test, y_test
    
def show_train_CIFAR10(X_train, y_train):
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 
                    'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):    
        idxs = np.flatnonzero(y_train == y)    
        idxs = np.random.choice(idxs, samples_per_class, replace=False) 
        for i, idx in enumerate(idxs):        
            plt_idx = i * num_classes + y + 1 
            plt.subplot(samples_per_class, num_classes, plt_idx)   
            plt.imshow(X_train[idx].astype('uint8'))        
            plt.axis('off')       
            if i == 0:            
                plt.title(cls)
    plt.show()

    
def show_mean_image(mean_image):
    # Visualize the mean image
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
    plt.show()
    