# -*- coding:utf-8 -*-


import numpy as np
from linear_svm import *
from softmax import *

class LinearClassifier:

    def __init__(self):
        '''
        - W: C x D array of weights.   # weights and bias
        '''
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
                        batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: D x N array of training data. Each training point is a D-dimensional
                 column.
        - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        dim, num_train = X.shape
        num_classes = np.max(y) + 1     # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = np.random.randn(num_classes, dim) * 0.001  # weights(dim) and bias(+1)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in xrange(num_iters):    # 每次随机取batch的数据来进行梯度下降
            X_batch = None
            y_batch = None
            
            # Sampling with replacement is faster than sampling without replacement.
            sample_index = np.random.choice(num_train, batch_size, replace=False)   # 从训练集中一次选取batch_size张图片
            X_batch = X[:, sample_index]   # batch_size by D
            y_batch = y[sample_index]      # 1 by batch_size
            
            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            # print grad.shape        # 10x3073
            loss_history.append(loss)
            
            # 更新权重
            self.W += -learning_rate * grad

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
            array of length N, and each element is an integer giving the predicted
            class.
        """
        y_pred = np.zeros(X.shape[1]) # y_pred: 1 x N

        # 预测直接找到最后y最大的那个值下标(行坐标)
        y_pred = np.argmax((self.W).dot(X), axis=0) 
        
        return y_pred
    


    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs:
        - X_batch: D x N array of data; each column is a data point.
        - y_batch: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

