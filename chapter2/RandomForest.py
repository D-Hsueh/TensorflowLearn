# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/16 16:24
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : RandomForest.py
@Software    : PyCharm
@introduction: 在本部分，我们将实现一个随机森林算法
"""
import tensorflow as tf
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.examples.tutorials.mnist import input_data
import os

def run():
    # 忽略所有的 GPU, 因为随机森林不会从中受益
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # 导入数据集
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

    # 超参数
    # num_steps     训练的次数
    # batch_size    每次喂给模型的批次大小
    # num_classes   目标 class 的数量
    # num_features  特征值的数量
    # num_trees     数的数量
    # max_nodes     最大节点数
    num_steps = 500
    batch_size = 1024
    num_classes = 10
    num_features = 784
    num_trees = 10
    max_nodes = 1000

    # 输入数据占位符
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # 对于随机森林来说，标签必须是int
    Y = tf.placeholder(tf.int32, shape=[None])

    # 设置随机森林超参数
    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                          num_features=num_features,
                                          num_trees=num_trees,
                                          max_nodes=max_nodes).fill()

    # 构建随机森林
    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    # 获取训练操作和损失函数
    train_op = forest_graph.training_graph(X, Y)
    loss_op = forest_graph.training_loss(X, Y)
    # 计算准确度
    infer_op, _, _ = forest_graph.inference_graph(X)
    correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化变量和随机森林
    init_vars = tf.group(tf.global_variables_initializer(),resources.initialize_resources(resources.shared_resources()))

    # 获取训练图
    sess = tf.train.MonitoredSession()

    # 初始化
    sess.run(init_vars)

    # 训练
    for i in range(1, num_steps + 1):
        # 准备数据
        # 获取下一批
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        if i % 50 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
            print('步数 %i, 损失: %f, 准确度: %f' % (i, l, acc))

    # 测试模型
    test_x, test_y = mnist.test.images, mnist.test.labels
    print("测试集准确度:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
    return