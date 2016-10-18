# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:26:59 2016

@author: peic
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import prepare_data_CIFAR10
from classifiers.linear_classifier import LinearSVM

if __name__ == "__main__":
        
    # Load the raw CIFAR-10 data 
    cifar10_dir = os.path.join(os.getcwd(), 'cifar10')
    X_train, y_train, X_val, y_val, X_test, y_test= prepare_data_CIFAR10(cifar10_dir)

    # Use the validation set to tune hyperparameters (regularization strength 
    # and learning rate).
    learning_rates = [1e-5, 1e-6]
    regularization_strengths = [5e4, 1e5]
    results = {}
    best_val = -1    # The highest validation accuracy that we have seen so far.
    best_svm = None   # The LinearSVM object that achieved the highest validation rate.
    iters = 2000
    for lr in learning_rates:
        for rs in regularization_strengths:    
            svm = LinearSVM()    
            svm.train(X_train.T, y_train, learning_rate=lr, reg=rs, num_iters=iters)    
            Tr_pred = svm.predict(X_train.T)    
            acc_train = np.mean(y_train == Tr_pred)    
            Val_pred = svm.predict(X_val.T)    
            acc_val = np.mean(y_val == Val_pred)    
            results[(lr, rs)] = (acc_train, acc_val)    
            if best_val < acc_val:
                best_val = acc_val
                best_svm = svm
    
    # print results
    for lr, reg in sorted(results):    
        train_accuracy, val_accuracy = results[(lr, reg)]    
        print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy)
    print 'Best validation accuracy achieved during validation: %f' % best_val    # about 38.2%
    
    # Visualize the learned weights for each class
    print best_svm.W.shape      # 10 x 3073
    w = best_svm.W.T[:-1, :]    # strip out the bias
    print w.shape               # 3072 x 10
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in xrange(10):    
        plt.subplot(2, 5, i + 1)
        # Rescale the weights to be between 0 and 255    
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)    
        plt.imshow(wimg.astype('uint8'))    
        plt.axis('off')    
        plt.title(classes[i])
    plt.show()
    
    # Evaluate the best svm on test set
    Ts_pred = best_svm.predict(X_test.T)
    test_accuracy = np.mean(y_test == Ts_pred)     # about 37.1%
    print 'LinearSVM on raw pixels of CIFAR-10 final test set accuracy: %f' % test_accuracy
