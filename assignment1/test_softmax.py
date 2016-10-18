# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import os
import numpy as np
from utils.data_utils import prepare_data_CIFAR10
from classifiers.linear_classifier import Softmax

if __name__ == "__main__":
        
    # Load the raw CIFAR-10 data 
    cifar10_dir = os.path.join(os.getcwd(), 'cifar10')
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_CIFAR10(cifar10_dir)

    # Use the validation set to tune hyperparameters (regularization strength 
    # and learning rate).
    results = {}
    best_val = -1
    best_softmax = None
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [5e4, 1e4]
    iters = 2000
    for lr in learning_rates:    
        for rs in regularization_strengths:        
            softmax = Softmax()       
            softmax.train(X_train.T, y_train, learning_rate=lr, reg=rs, num_iters=iters)      # X_train.T: 3073(D) x 49000(N)  
            Tr_pred = softmax.predict(X_train.T)       
            acc_train = np.mean(y_train == Tr_pred)       
            Val_pred = softmax.predict(X_val.T)        
            acc_val = np.mean(y_val == Val_pred)       
            results[(lr, rs)] = (acc_train, acc_val)       
            if best_val < acc_val:           
                best_val = acc_val            
                best_softmax = softmax

    # Print out results.
    for lr, reg in sorted(results):    
        train_accuracy, val_accuracy = results[(lr, reg)]    
        print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy)
            # about 38.9%                     
    print 'best validation accuracy achieved during cross-validation: %f' % best_val

    # Evaluate the best softmax on test set.
    Ts_pred = best_softmax.predict(X_test.T)
    test_accuracy = np.mean(y_test == Ts_pred)       # about 37.4%
    print 'Softmax on raw pixels of CIFAR-10 final test set accuracy: %f' % test_accuracy