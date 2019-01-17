# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/17 15:07
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : ConvolutionalNeuralNetwork.py
@Software    : PyCharm
@introduction: 利用 Tensorflow 实现一个卷积神经网络
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def run():
    # 导入数据
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 训练参数
    learning_rate = 0.001
    num_steps = 500
    batch_size = 128
    display_step = 10

    # 网络参数
    num_input = 784     # 网络的输入
    num_classes = 10    # 网络的输出
    dropout = 0.75      # Dropout的概率

    # 定义输入数据
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # 为了代码简洁，我们创建一些包装器
    def conv2d(x, W, b, strides=1):
        # Conv2D 包装器, 具有偏置和激活功能
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D 包装器
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # 创建模型
    def conv_net(x, weights, biases, dropout):
        # 输入是一个 1 维的 28 * 28 的图片，总共 784 个特征。
        # 把输入变形来匹配格式 [高 x 宽 x 通道]
        # Tensor的输入是四维的: [批量大小, 高, 宽, 通道]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # 卷积层1
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # 最大池1
        conv1 = maxpool2d(conv1, k=2)

        # 卷积层2
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # 最大池2
        conv2 = maxpool2d(conv2, k=2)

        # 全连接层
        # 把卷积层2的输出变形来适应真正的输出
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # 应用 Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # 输出是类预测
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # 保存权重和偏置值
    weights = {
        # 5x5 卷积层, 1 输入, 32 输出
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 卷积层, 32 输入, 64 输出
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # 全连接层, 7*7*64 输入, 1024 输出
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        # 1024 输入, 10 输出
        'out': tf.Variable(tf.random_normal([1024, num_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # 构建模型
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    # 定义损失函数和优化器
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # 评估模型
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 初始化
    init = tf.global_variables_initializer()

    # 开始训练
    with tf.Session() as sess:
        sess.run(init)
        for step in range(1, num_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 运行优化函数（反向传播）
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})
                print("步数 " + str(step) + ", 该批的损失= " + \
                      "{:.4f}".format(loss) + ", 训练准确度= " + \
                      "{:.3f}".format(acc))

        print("优化完成!")

        print("测试准确度:", \
              sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                            Y: mnist.test.labels[:256],
                                            keep_prob: 1.0}))

    return