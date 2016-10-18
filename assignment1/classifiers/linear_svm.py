# -*- coding:utf-8 -*-


import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights		# 权重参数
  - X: D x N array of data. Data are D-dimensional columns 	# N张图片的数据
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes  # N张图片的标签
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]		# 总的类别
  num_train = X.shape[1]		# 训练图片的数量
  loss = 0.0
  delta = 1.0 					# 设置边界delta = 1
  for i in xrange(num_train):
    scores = W.dot(X[:, i])		# W和X做矩阵乘法，得到每个标签的得分
    correct_class_score = scores[y[i]]		# 正确标签的得分
    
    # 对所有类别分别求Loss
    for j in xrange(num_classes):
      if j == y[i]:				# 跳过正确类别(不做损失)
        continue
      margin = scores[j] - correct_class_score + delta		# 边界损失
      if margin > 0:
        loss += margin			# max(0, margin)
        
        # 梯度的计算
        dW[y[i], :]	-= X[:, i].T	# 根据公式，L对W求偏导的结果就是X
        dW[j, :] += X[:, i].T


  loss /= num_train
  #	参数正则化
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  # 导数正则化
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # Implement a vectorized version of the structured SVM loss, storing the result in loss.                                                           
  WX = W.dot(X)
  num_train = X.shape[1]
  
  # 每张图片正确标签的得分  
  Sy = np.zeros(num_train)
  for i in xrange(num_train):
    Sy[i] = WX[y[i], i]		# WX.shape: C x N

  WX = WX - Sy + 1

  for i in xrange(num_train):
    WX[y[i], i] -= 1 

  loss = np.sum( WX[WX > 0] )
  loss /= num_train 
  
  # 计算偏导数
  num_classes = W.shape[0]
  for i in xrange(num_train):
    for j in xrange(num_classes):
      if (WX[j, i] > 0):
        # 梯度的计算
        dW[y[i], :]	-= X[:, i].T	# 根据公式，L对W求偏导的结果就是X
        dW[j, :] += X[:, i].T  

  dW /= num_train
  dW += reg * W

  return loss, dW
