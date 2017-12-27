# -*- coding: utf-8 -*-
# @StartTime : 2017/12/25 13:41
# @EndTime : 2017/12/25 13:41
# @Author  : Andy
# @Site    : 
# @File    : mnist_for_ml_beginners.py
# @Software: PyCharm

"""
Actually this is a program in Chinese version of the official document.
In the meantime is book <tensorflow practice> 3.2 's program.
Site: http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html
"""
import tensorflow as tf
import read_mnist_data


mnist = read_mnist_data.read_data_sets("MNIST_data/", one_hot=True)
# make data placeholder and init Variable
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
# y_ is true number of mnist data sets
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# this has some compile problem that need to solve
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# evaluate the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))







