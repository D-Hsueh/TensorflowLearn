# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/21 14:57
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : AutoEncoder.py
@Software    : PyCharm
@introduction: 实现一个自动编码器
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def run():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 超参数设置
    learning_rate = 0.01    # 学习率
    num_steps = 30000       # 训练步数
    batch_size = 256        # 单批训练大小

    display_step = 1000     # 展示间隔
    examples_to_show = 10   # 展示例子数量

    # 网络参数
    num_hidden_1 = 256      # 第一层特征数
    num_hidden_2 = 128      # 第二层特征数
    num_input = 784         # 数据集输入

    # 输入数据定义
    X = tf.placeholder("float", [None, num_input])

    # 权重和偏置值定义
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([num_input])),
    }

    # 建立编码器
    def encoder(x):
        # 带有 sigmoid 的第一层隐藏层编码器
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # 带有 sigmoid 的第二层隐藏层编码器
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        return layer_2

    # 建立解码器
    def decoder(x):
        # 带有 sigmoid 的第一层隐藏层解码器
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # 带有 sigmoid 的第二层隐藏层解码器
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        return layer_2

    # 建立模型
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # 预测
    y_pred = decoder_op
    # 预测的结果就是输入数据
    y_true = X

    # 定义损失函数和优化器, 最小化平方误差
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    # 初始化
    init = tf.global_variables_initializer()

    # 开始训练
    sess = tf.Session()
    sess.run(init)
    for i in range(1, num_steps + 1):
        # 准备数据
        # 获取下一批
        batch_x, _ = mnist.train.next_batch(batch_size)

        # 运行损失函数优化
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # 展示log
        if i % display_step == 0 or i == 1:
            print('步数 %i: 当前批次损失: %f' % (i, l))

    # 测试
    # 从测试集编码和解码图片
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        batch_x, _ = mnist.test.next_batch(n)
        # 编码和解码
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # 展示原始图
        for j in range(n):
            # 绘制图片
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
        # 展示重构的图片
        for j in range(n):
            # 绘制图片
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    print("原始图片")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("重构图片")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

    return