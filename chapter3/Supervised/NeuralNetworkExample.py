# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/17 10:20
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : NeuralNetworkExample.py
@Software    : PyCharm
@introduction: 实现一个简单的神经网络
"""


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def run():
    # 导入 MNIST 数据集
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 超参数
    learning_rate = 0.1     # 学习率
    num_steps = 500         # 学习次数
    batch_size = 128        # 批次大小
    display_step = 100      # 展示间隔

    # 神经网络参数
    n_hidden_1 = 256        # 神经网络第一层的隐藏单元数量
    n_hidden_2 = 256        # 神经网络第二层的隐藏单元数量
    num_input = 784         # 输入尺度的大小 （28 * 28的图片）
    num_classes = 10        # 输出分类的数量

    # 定义输入输出，None表示第一个维度（行）的数量不确定
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # 定义每一层的权重和偏置值
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # 创建模型
    def neural_net(x):
        # 第一层与输入全连接
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # 第二层与第一层全连接
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # 输出层与第二层全连接
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # 构建模型
    logits = neural_net(X)

    # 定义损失函数和优化方法
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # 带有测试日志的评估模型
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开始训练
    with tf.Session() as sess:
        # 运行初始化变量的方法
        sess.run(init)
        for step in range(1, num_steps + 1):
            # 获取下一批训练数据
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 运行反向传播训练
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            # 达到展示步限制后展示当前训练结果
            if step % display_step == 0 or step == 1:
                # 计算当前批次的准确度和损失
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("步数 " + str(step) + ", 最小损失= " + \
                      "{:.4f}".format(loss) + ", 训练准确度= " + \
                      "{:.3f}".format(acc))

        print("训练完成!")

        # 计算测试集上的准确性
        print("在测试集上的准确度为:", \
              sess.run(accuracy, feed_dict={X: mnist.test.images,
                                            Y: mnist.test.labels}))
    return