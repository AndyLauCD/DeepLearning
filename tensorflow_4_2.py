# -*- coding: utf-8 -*-
# @StartTime : 2017/12/26 14:35
# @EndTime : 2017/12/26 14:35
# @Author  : Andy
# @Site    : 
# @File    : tensorflow_4_2.py
# @Software: PyCharm


import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import read_mnist_data


# 参数初始化
def xavier_init(fan_in, fan_out, constant=1):
    """
    :param fan_in: 输入节点的数量
    :param fan_out:输出节点的数量
    :param constant:
    :return: 均匀分布
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = - constant * np.sqrt((6.0 / (fan_in + fan_out)))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high,
                             dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 定义网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.n_hidden = self.transfer(tf.add(tf.matmul(
            self.x + scale * tf.random_normal((n_input,)), self.weights['w1']),
            self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.n_hidden,
                                     self.weights['w2']), self.weights['b2'])

        # 定义自编码器的损失函数
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weigthts = dict()
        all_weigthts['w1'] = tf.Variable(
            xavier_init(self.n_input, self.n_hidden)
        )
        all_weigthts['b1'] = tf.Variable(
            tf.zeros([self.n_hidden]), dtype=tf.float32
        )
        all_weigthts['w2'] = tf.Variable(
            tf.zeros([self.n_hidden, self.n_input]), dtype=tf.float32
        )
        all_weigthts['b2'] = tf.Variable(
            tf.zeros([self.n_input]), dtype=tf.float32
        )
        return all_weigthts

    # 拟合优化
    def partial_fit(self, X):
        cost, opt = self.sess.run(
            (self.cost, self.optimizer),
            feed_dict={self.x: X, self.scale: self.training_scale}
        )
        return cost

    # 只求损失不优化(测试集上用到)
    def calc_total_cost(self, X):
        return self.sess.run(
            self.cost,  feed_dict={self.x: X, self.scale: self.training_scale}
        )

    # 返回自编码器隐含层的输出结果
    def transform(self, X):
        return self.sess.run(
            self.n_hidden,
            feed_dict={self.x: X, self.scale: self.training_scale}
        )

    # 将隐含层的输出作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(
            self.reconstruction, feed_dict={self.hidden: hidden}
        )

    # 整体运行一遍复原过程
    def reconstruct(self, X):
        return self.sess.run(
            self.reconstruction,
            feed_dict={self.x: X, self.scale: self.training_scale},
        )

    def get_weight(self):
        return self.sess.run(self.weights['w1'])

    def get_bias(self):
        return self.sess.run(self.weights['b1'])

















