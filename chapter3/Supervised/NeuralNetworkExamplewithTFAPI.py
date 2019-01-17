# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/17 10:36
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : NeuralNetworkExamplewithTFAPI.py
@Software    : PyCharm
@introduction: 利用 Tensorflow 提供的 API 更方便的实现全连接神经网络
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def run():
    # 导入数据集
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

    # 超参数
    learning_rate = 0.1     # 学习率
    num_steps = 1000        # 学习次数
    batch_size = 128        # 批次大小
    display_step = 100      # 展示间隔

    # 神经网络参数
    n_hidden_1 = 256        # 神经网络第一层的隐藏单元数量
    n_hidden_2 = 256        # 神经网络第二层的隐藏单元数量
    num_input = 784         # 输入尺度的大小 （28 * 28的图片）
    num_classes = 10        # 输出分类的数量

    # 定义用于训练的输入方法
    # numpy_input_fn(   x,                      numpy数组对象或numpy数组对象的dict。
    #                                           如果是数组，则该数组将被视为单个特征。
    #                   y=None,                 numpy数组对象或numpy数组对象的dict。 如果不存在，则为None。
    #                   batch_size=128,         int，批量的大小。
    #                   num_epochs=1,           迭代数据的次数。如果为 None 将一直运行下去
    #                   shuffle=None,           如果为 True 将随机打乱顺序
    #                   queue_capacity=1000,    累计队列的大小
    #                   num_threads=1):         用于读取和排队的线程数。
    #                                           为了具有预测和可重复的读取和排队顺序，
    #                                           例如在预测和评估模式中，应该是1。
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images}, y=mnist.train.labels,
        batch_size=batch_size, num_epochs=None, shuffle=True)

    # 定义神经网络
    # dense(                                            用来定义一个神经网络层
    #     inputs, units,                                输入数据，输出维度
    #     activation=None,                              激活功能。 将其设置为 None 以保持线性激活。
    #     use_bias=True,                                该层是否使用偏置值
    #     kernel_initializer=None,                      权重矩阵的初始化函数。
    #                                                   如果是 None
    #                                                   则使用 tf.get_variable 使用的默认初始化程序初始化权重。
    #     bias_initializer=init_ops.zeros_initializer(),用于初始化偏置值的初始化函数
    #     kernel_regularizer=None,                      用于权重矩阵的正则化函数
    #     bias_regularizer=None,                        用于偏置值的初始化函数
    #     activity_regularizer=None,                    用于输出的正在化函数
    #     kernel_constraint=None,                       用于权重的约束项
    #     bias_constraint=None,                         用于偏置值的约束项
    #     trainable=True,                               如果为 True 那么变量也会加到
    #                                                   GraphKeys.TRAINABLE_VARIABLES 中
    #     name=None,                                    该层的名称
    #     reuse=None)                                   是否重用前一层同名的变量
    def neural_net(x_dict):
        # 在多个输入的时候，TF Estimator输入是一个字典
        x = x_dict['images']
        # 第一层与输入全连接
        layer_1 = tf.layers.dense(x, n_hidden_1)
        # 第二层与第一层全连接
        layer_2 = tf.layers.dense(layer_1, n_hidden_2)
        # 输出层与第二层全连接
        out_layer = tf.layers.dense(layer_2, num_classes)
        return out_layer

    # 定义模型（基于 TF 分类器）
    def model_fn(features, labels, mode):
        # 建立神经网络
        logits = neural_net(features)

        # 进行预测
        pred_classes = tf.argmax(logits, axis=1)
        pred_probas = tf.nn.softmax(logits)

        # 如果实在预测模式下，那么就提早返回
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # 定义损失函数和优化器
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # 评价模型的准确性
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF分类器需要返回EstimatorSpec，它指定用于训练，评估，...的不同操作。
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})
        return estim_specs

    # 建立分类器
    model = tf.estimator.Estimator(model_fn)

    # 训练模型
    model.train(input_fn, steps=num_steps)

    # 评估模型
    # 定义用于评估的输入
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images}, y=mnist.test.labels,
        batch_size=batch_size, shuffle=False)
    # 采用分类器的评估方法进行评估模型
    model.evaluate(input_fn)

    # 预测图片
    n_images = 4
    # 从测试集中获取部分图片用于预测
    test_images = mnist.test.images[:n_images]
    # 准备输入数据
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': test_images}, shuffle=False)
    # 利用模型进行预测
    preds = list(model.predict(input_fn))

    # 展示
    for i in range(n_images):
        plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
        plt.show()
        print("预测结果:", preds[i])

    return