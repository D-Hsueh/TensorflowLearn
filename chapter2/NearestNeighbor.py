# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/16 13:51
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : NearestNeighbor.py
@Software    : PyCharm
@introduction: 利用 tensorflow 实现一个最近邻算法
"""


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def run():
    # 加载数据
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    # 在本次训练中，我们选择5000组数据用于训练，200组数据用于测试
    Xtr, Ytr = mnist.train.next_batch(5000)
    Xte, Yte = mnist.test.next_batch(200)

    # 声明训练数据和测试数据的输入维度
    # [None, 784]表示一共有 784个特征值（列），但是行不确定
    # [784]表示一位数据，有784个特征点
    # 为什么这个地方要采取这样一种方式对变量进行复制呢，我的理解是：
    #   tensorflow 提供的 add 方法面对矩阵加法时如果张量的维度不相同则会自动把低维矩阵对应加到高维矩阵上去。
    #   如当一个二维矩阵于一维矩阵运算时，二维矩阵中的每一个一维向量会与一维向量去运算，并且最终的结果为一个高维矩阵。
    #   但是需要注意的时，该自动映射运算仅限于形如[n,y] + [y]这种维度不同的运算映射，
    #   [2n,y] + [n,y]这种同维度的就会因为形状不同而无法运算。
    xtr = tf.placeholder("float", [None,784])
    xte = tf.placeholder("float", [784])

    # 使用L1距离计算最近邻
    # 计算L1距离
    # tf.add(a, tf.negative(b)) == a - b, 其中 tf.negative被用来取反square
    # distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    # 或使用L2距离计算最近邻
    distance = tf.sqrt(tf.reduce_sum(tf.square(tf.add(xtr, tf.negative(xte))), reduction_indices=1))
    # 预测结果：获取最小的距离指数
    pred = tf.argmin(distance, 0)

    # 初始化准确性
    accuracy = 0.

    # 初始化所有变量
    init = tf.global_variables_initializer()

    # 开始训练
    with tf.Session() as sess:
        sess.run(init)
        # 直接对测试数据进行操作
        for i in range(len(Xte)):
            # 通过运行模型，计算测试数据于所有训练数据的距离，找到最近的邻居
            nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
            # 打印出预测数据
            print("测试", i, "预测结果:", np.argmax(Ytr[nn_index]),
                  "真正的结果:", np.argmax(Yte[i]))
            # 计算准确性
            if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
                accuracy += 1. / len(Xte)
        print("完成!")
        print("准确性:", accuracy)

    return
