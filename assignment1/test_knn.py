# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import os
from CS231n.data_utils import load_CIFAR10
from CS231n.classifiers.k_nearest_neighbor import KNearestNeighbor
from matplotlib import pyplot as plt


if __name__ == "__main__":
    
    k_choices = [1,3,5,10,20,50,100]
    
    # 加载测试和训练的数据
    # 内存有限，制度取部分数据
    Xtr, Ytr, Xte, Yte = load_CIFAR10(os.path.join(os.getcwd(), 'CS231n\cifar10')) # a magic function we provide
    # flatten out all images to be one-dimensional
    
    # 训练有50000张图片
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
    
    # 取1000张作为验证集，用来调整超参数
    Xval_rows = Xtr_rows[:1000, :]
    Yval = Ytr[:1000]
    Xtr_rows = Xtr_rows[1000:, :]
    Ytr = Ytr[1000:]

    # 测试有10000张图片
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
    
    validation_accuracies = {}
    
    for k in k_choices:
    
        knn = KNearestNeighbor() # create a Nearest Neighbor classifier class
        knn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels 
        
        # k=5: 默认是最好邻近分类器
        Yval_predict = knn.predict(Xval_rows, k=k) # predict labels on the test images  
        # and now print the classification accuracy, which is the average number
        # of examples that are correctly predicted (i.e. label matches)
        accuracy = np.mean(Yval_predict == Yval)
        validation_accuracies[k] = accuracy   
        print 'k = %d, accuracy = %f' % (k, accuracy)    
  
  
    # plot the raw observations  
     
    accuracies = validation_accuracies.values()
    plt.plot(k_choices, accuracies)
  
    # plot the trend line with error bars that correspond to standard deviation  
    accuracies_mean = np.array([np.mean(v) for k,v in sorted(validation_accuracies.items())])  
    accuracies_std = np.array([np.std(v) for k,v in sorted(validation_accuracies.items())])  
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)  
    plt.title('Cross-validation on k')  
    plt.xlabel('k')  
    plt.ylabel('Cross-validation accuracy')  
    plt.show()  