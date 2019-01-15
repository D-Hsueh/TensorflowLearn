# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/15 16:46
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : LinearRegressionWithEagerAPI.py
@Software    : PyCharm
@introduction: 这个部分主要介绍用动态图的方式实现线性回归模型，因此对于与之前重复的部分就不再赘述了
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def run():
    # 动态图模式
    tf.enable_eager_execution()
    tfe = tf.contrib.eager

    # 训练数据
    train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
               7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
    train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
               2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
    n_samples = len(train_X)

    # 超参数设置
    learning_rate = 0.01
    display_step = 100
    num_steps = 1000

    # W 和 b
    W = tfe.Variable(np.random.randn())
    b = tfe.Variable(np.random.randn())

    # 线性回归模型 (Wx + b)
    def linear_regression(inputs):
        return inputs * W + b

    # 均方误差
    def mean_square_fn(model_fn, inputs, labels):
        return tf.reduce_sum(tf.pow(model_fn(inputs) - labels, 2)) / (2 * n_samples)

    # 梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # 计算梯度
    grad = tfe.implicit_gradients(mean_square_fn)

    # 再优化之前初始化 cost
    print("初始化 cost= {:.9f}".format(
        mean_square_fn(linear_regression, train_X, train_Y)),
        "W=", W.numpy(), "b=", b.numpy())

    # 训练
    for step in range(num_steps):

        optimizer.apply_gradients(grad(linear_regression, train_X, train_Y))

        if (step + 1) % display_step == 0 or step == 0:
            print("步数:", '%04d' % (step + 1), "cost=",
                  "{:.9f}".format(mean_square_fn(linear_regression, train_X, train_Y)),
                  "W=", W.numpy(), "b=", b.numpy())

    # 展示结果
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, np.array(W * train_X + b), label='Fitted line')
    plt.legend()
    plt.show()
    return

