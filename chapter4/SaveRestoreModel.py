# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/22 9:59
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : SaveRestoreModel.py
@Software    : PyCharm
@introduction: 这是一个如何保存训练好的模型和加载训练好的模型的例子
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def run():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 训练参数
    learning_rate = 0.001           # 学习率
    batch_size = 100                # 每次训练的数据大小
    display_step = 1                # 展示间隔
    model_path = "./tmp/model.ckpt"  # 模型保存路径

    # 网络参数
    n_hidden_1 = 256
    n_hidden_2 = 256
    n_input = 784
    n_classes = 10

    # 模型输入
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # 创建模型
    def multilayer_perceptron(x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # 权重和偏置值
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = multilayer_perceptron(x, weights, biases)
    # 定义损失和优化器
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # 我们利用 saver 来存储和恢复所有的变量
    saver = tf.train.Saver()

    # 运行第一个 session
    print("开始第一个 session...")
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(3):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # 分批压入所有的数据
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # 计算平均损失
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("代数:", '%04d' % (epoch + 1), "损失=", \
                "{:.9f}".format(avg_cost))
        print("第一次优化结束!")

        # 测试模型
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # 计算准确度
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("准确度:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        # 保存模型
        save_path = saver.save(sess, model_path)
        print("模型保存在: %s" % save_path)

    # 开始第二个 session
    print("开始运行第二个session...")
    with tf.Session() as sess:
        sess.run(init)
        # 从已存在的路径中加载模型
        load_path = saver.restore(sess, model_path)
        print("模型从下列路径中恢复: %s" % save_path)
        for epoch in range(7):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("代数:", '%04d' % (epoch + 1), "损失=", \
                      "{:.9f}".format(avg_cost))
        print("第二次优化完成!")

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("准确度:", accuracy.eval(
            {x: mnist.test.images, y: mnist.test.labels}))

    return