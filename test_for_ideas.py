# -*- coding: utf-8 -*-
# @StartTime : 10/11/2017 23:21
# @EndTime   : 10/11/2017 23:21
# @Author    : Andy
# @Site      : 
# @File      : test_for_ideas.py
# @Software  : PyCharm

from tensorflow.examples.tutorials.mnist import input_data
minist = input_data.read_data_sets("MINIST_data/",one_hot=True)