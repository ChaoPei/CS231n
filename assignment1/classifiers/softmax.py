# -*- coding: utf-8 -*-


import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    dW_each = np.zeros_like(W)
    dim, num_train = X.shape
    num_classes = W.shape[0]
    f = W.dot(X)    # C x N
    # 每张图片的所有类别评分是一列, 所以设置axis=0求每一列的最大值
    f_max = np.reshape(np.max(f, axis=0), (1, num_train))
    # softmax函数
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=0, keepdims=True) # C x N
    y_trueClass = np.zeros_like(prob)
    # 将每张图标(n)签的类别(c)的值设置为1
    y_trueClass[y, np.arange(num_train)] = 1.0    # y_trueClass[c, n] = 1.0

    for i in xrange(num_train):
        for j in xrange(num_classes):
            # 损失函数的公式: L = -(1/N)∑i∑j1(k=yi)log(exp(fk)/∑j exp(fj)) + λR(W)
            loss += -(y_trueClass[j, i] * np.log(prob[j, i]))    
            
            #梯度的公式 Wk L = -(1/N)∑i xiT(pi,m-Pm) + 2λWk, where Pk = exp(fk)/∑j exp(fj    
            dW_each[j, :] = -(y_trueClass[j, i] - prob[j, i]) * X[:, i]
    
        # 将所有的类放在一起    
        dW += dW_each        # C*D

    # 正则化
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)    
    dW /= num_train 
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    dim, num_train = X.shape

    f = W.dot(X)    # C x N
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=0), (1, num_train))        # N x 1
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=0, keepdims=True) # C x N
    
    y_trueClass = np.zeros_like(prob)    # C x N
    y_trueClass[y, range(num_train)] = 1.0    # N x C
    
    #向量化直接操作即可
    loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)
    dW += -np.dot(y_trueClass - prob, X.T) / num_train + reg * W         # D x C

    return loss, dW
