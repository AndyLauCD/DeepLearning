# -*- coding: utf-8 -*-
# @StartTime : 2017/12/25 10:37
# @EndTime : 2017/12/25 10:37
# @Author  : Andy
# @Site    : 
# @File    : test.py
# @Software: PyCharm

import read_mnist_data


mnist = read_mnist_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)