# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/17 16:06
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : LSTM.py
@Software    : PyCharm
@introduction: 利用 Tensorflow 实现一个 LSTM 网络
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

def run():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 训练参数
    learning_rate = 0.001
    training_steps = 10000
    batch_size = 128
    display_step = 200

    # 网络参数
    # 原图是 28 * 28 的输入数据，我们理解为一次送一行数据，一共送28次(时间)
    num_input = 28      # 输入数据 （28 * 28）
    timesteps = 28      # 时间步
    num_hidden = 128    # 隐藏单元数量
    num_classes = 10    # 最终分类要求

    # 定义输入
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # 定义权重和偏置值
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    def RNN(x, weights, biases):
        # 为了匹配 RNN 的数据需求来准备数据
        # 当前输入形状: (批量大小, 时间步, 输入)
        # 需要的形状: 在每一个时间步里 (批量大小, 输入)

        # 取消堆栈以获取形状的时间步张量列表 (批量大小, 输入)
        x = tf.unstack(x, timesteps, 1)

        # 定义一个 LSTM Cell
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # 获得 LSTM 的输出
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # 线性激活，使用rnn内循环最后输出
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    logits = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # 定义损失函数和优化方法
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # 评价模型
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 初始化模型
    init = tf.global_variables_initializer()

    # 开始训练
    with tf.Session() as sess:
        sess.run(init)
        for step in range(1, training_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 把输入数据处理成需要的形状
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # 运行优化方法
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("步数 " + str(step) + ", 该组数据的损失= " + \
                      "{:.4f}".format(loss) + ", 训练准确度= " + \
                      "{:.3f}".format(acc))

        print("优化完成!")

        # 计算在128个测试数据上的测试准确度
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        test_label = mnist.test.labels[:test_len]
        print("测试准确度:", \
              sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    return