# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/21 14:20
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : bidirectionalLSTM.py
@Software    : PyCharm
@introduction: 我们借助 Tesorflow 实现一个双向递归训练网络
"""
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def run():
    # 加载数据集
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 设置超参数
    learning_rate = 0.001   # 学习率
    training_steps = 10000  # 训练次数
    batch_size = 128        # 单次训练数据集大小
    display_step = 200      # 展示间隔

    # 网络超参数
    num_input = 28          # 每一行输入
    timesteps = 28          # 时间步（共28行）
    num_hidden = 128        # 隐藏单元的数量
    num_classes = 10        # 最终分类的数目

    # 构造输入数据
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # 定义权重和偏置值
    weights = {
        # 隐藏层权重 => 2*隐藏单元数量，由于前向单元+反向单元的关系（双向循环）
        'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    def BiRNN(x, weights, biases):

        # 准备数据格式来匹配 RNN 功能要求
        # 当前输入的数据格式: (batch_size, timesteps, n_input)
        # 需要的格式是一个以 timesteps 的张量列表 (batch_size, num_input)

        # 利用 tf.unstack 来把矩阵分解成按 timesteps 切片(batch_size, num_input)
        x = tf.unstack(x, timesteps, 1)

        # 定义 lstm 神经元
        # 前向方向单元
        lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # 反向方向单元
        lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # 获得输出
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # 老版本的 tensorflow 只有 outputs 返回值
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

        # 线性拟合，最后使用 rnn 内循环输出
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    # 生成模型，获取预测
    logits = BiRNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # 定义损失函数，优化去
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # 预测模型
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 初始化
    init = tf.global_variables_initializer()

    # 开始训练
    with tf.Session() as sess:
        sess.run(init)
        for step in range(1, training_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 把数据变形成指定的格式
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # 运行优化操作
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("步数 " + str(step) + ", 当前批次损失= " + \
                      "{:.4f}".format(loss) + ", 训练准确度= " + \
                      "{:.3f}".format(acc))

        print("优化完成!")

        # 在 128 大小的测试集上计算准确度
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        test_label = mnist.test.labels[:test_len]
        print("测试准确度:", \
              sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    return