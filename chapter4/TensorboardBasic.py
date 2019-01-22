# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/22 10:31
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : TensorboardBasic.py
@Software    : PyCharm
@introduction: tensorflow 提供的可视化功能
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
def run():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 超参数
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_epoch = 1
    logs_path = './tmp/tensorflow_logs/example/'    # 日志位置

    # 定义输入输出
    x = tf.placeholder(tf.float32, [None, 784], name='InputData')
    y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

    # 权重和偏置值
    W = tf.Variable(tf.zeros([784, 10]), name='Weights')
    b = tf.Variable(tf.zeros([10]), name='Bias')

    # 构建模型并将所有操作封装起来供可视化使用
    with tf.name_scope('Model'):
        pred = tf.nn.softmax(tf.matmul(x, W) + b)
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    with tf.name_scope('SGD'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    with tf.name_scope('Accuracy'):
        # 准确率
        acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    init = tf.global_variables_initializer()
    # 创建一个 summary 来监控损失
    tf.summary.scalar("loss", cost)
    # 创建一个 summary 来监控准确度
    tf.summary.scalar("accuracy", acc)
    # 把所有的 summary 合并成一个操作
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        # 记录日志的操作
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                         feed_dict={x: batch_xs, y: batch_ys})
                # 每次迭代都记录日志
                summary_writer.add_summary(summary, epoch * total_batch + i)
                avg_cost += c / total_batch
            if (epoch + 1) % display_epoch == 0:
                print("代数:", '%04d' % (epoch + 1), "损失=", "{:.9f}".format(avg_cost))

        print("优化完成!")

        print("准确度:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))

        print("命令行运行:\n" \
              "--> tensorboard --logdir=./tmp/tensorflow_logs " \
              "\n然后在你的浏览器打开 http://0.0.0.0:6006/")
    return