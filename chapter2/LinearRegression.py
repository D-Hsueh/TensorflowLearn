# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/15 16:03
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : LinearRegression.py
@Software    : PyCharm
@introduction: 我们将利用 tensorflow 实现一个简单的线性回归
"""

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

def run():
    rng = numpy.random

    # 首先我们需要预定义一些超参数
    # 我们在此处定义三个超参数，分别为：
    # learning_rate     学习率
    # training_epochs   训练次数
    # display_step      展示步
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 50

    # 我们通过直接输入的方式来定义训练数据 x 和 y
    train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                             7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                             2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    n_samples = train_X.shape[0]

    # 采用占位符的方式定义线性回归的 X 与 Y ，他们的类型都为float
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # tf.Variable
    # 利用此函数可以定义 tensorflow 的图变量，其构造方法如下
    # tf.Variable.init(
    # initial_value,        变量的初始值
    # trainable=True,       如果为True，会把它加入到GraphKeys.TRAINABLE_VARIABLES，才能对它使用Optimizer
    # collections=None,     指定该图变量的类型、默认为[GraphKeys.GLOBAL_VARIABLES]
    # validate_shape=True,  如果为False，则不进行类型和维度检查
    # name=None)            变量的名称，如果没有指定则系统会自动分配一个唯一的值
    # 在此我们采用随机值来初始化线性模型的 w 和 b 变量
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    # 实现线性模型 y = wx + b
    pred = tf.add(tf.multiply(X, W), b)

    # 我们采用梯度下降的方式来训练模型，损失函数则使用均方误差
    # tf.reduce_sum(                是求和函数，通过该函数可以进行简单的降维求和
    #     input_tensor,             表示输入
    #     axis=None,                表示在哪个维度进行sum操作，默认值则对所有维度进行降维求和
    #     keep_dims=False,          表示是否保留原始数据的维度，False相当于执行完后原始数据就会少一个维度。
    #     name=None,
    #     reduction_indices=None    为了跟旧版本的兼容，现在已经不使用了。
    # )
    # tf.train.GradientDescentOptimizer 表示梯度下降优化器 ,通过设置 learning_rate 的值来自定义学习率
    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # 初始化所有的变量，给他们分配其默认值
    # tf.global_variables_initializer() 方法可以给之前定义的变量全部初始化赋值
    # 注意，TF的变量需要初始化才有值，而常量和占位符则不需要
    init = tf.global_variables_initializer()

    # 定义所有所需要的操作之后，我们就可以开始训练我们的模型了
    with tf.Session() as sess:
        # 首先初始化模型
        sess.run(init)
        for epoch in range(training_epochs):
            # 在每一次的训练过程中，我们把全部的数据给模型供其训练
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # 每训练 display_step 次，我们就展示当前训练的状态
            if (epoch + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("步数:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(c),
                      "W=", sess.run(W),
                      "b=", sess.run(b))


        print("训练完成!")
        # 再执行一次训练，获取最终的训练状态
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("最终的训练cost为=", training_cost,
              "W=", sess.run(W),
              "b=", sess.run(b), '\n')


        # 把训练结果画出来
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.plot(train_X, train_Y, 'ro', label='训练数据')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='拟合曲线')
        plt.legend()
        plt.show()

    return