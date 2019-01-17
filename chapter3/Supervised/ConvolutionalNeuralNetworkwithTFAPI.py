# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/17 15:28
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : ConvolutionalNeuralNetworkwithTFAPI.py
@Software    : PyCharm
@introduction: 利用 Tensorflow 提供的 API 更快的实现卷积神经网络
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
def run():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

    # 训练参数设置
    learning_rate = 0.001
    num_steps = 2000
    batch_size = 128

    # 神经网络参数
    num_input = 784     # 数据输入
    num_classes = 10    # 网络输出
    dropout = 0.25      # Dropout

    # 创建神经网络
    def conv_net(x_dict, n_classes, dropout, reuse, is_training):
        # 定义重用变量的范围
        with tf.variable_scope('ConvNet', reuse=reuse):
            # 输入是一个字典
            x = x_dict['images']

            # 输入是一个 1 维的 28 * 28 的图片，总共 784 个特征。
            # 把输入变形来匹配格式 [高 x 宽 x 通道]
            # Tensor的输入是四维的: [批量大小, 高, 宽, 通道]
            x = tf.reshape(x, shape=[-1, 28, 28, 1])

            # 卷积层有32个过滤器，内核大小为5
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            # 最大池（下采样），步长为2，核大小为2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            # 卷积层有64个过滤器，核大小为3
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            # 最大池（下采样），步长为2，核大小为2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # 将数据展平为完全连接层的1-D向量
            fc1 = tf.contrib.layers.flatten(conv2)

            # 完全连接的图层
            fc1 = tf.layers.dense(fc1, 1024)
            # 应用Dropout（如果is_training为False，则不应用dropout）
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # 输出层，类预测
            out = tf.layers.dense(fc1, n_classes)
        return out

    # 定义模型
    def model_fn(features, labels, mode):
        # 建立神经网络
        # 由于Dropout在训练和预测时具有不同的行为，我们需要创建两个仍然共享相同权重的不同计算图。
        logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
        logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

        # 预测
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_probas = tf.nn.softmax(logits_test)

        # 如果在预测模式下，就提前 return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # 定义损失函数和优化器
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # 评价模型
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators需要返回EstimatorSpec，它指定用于训练，评估，...的不同操作。
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs

    # 构建分类器
    model = tf.estimator.Estimator(model_fn)

    # 定义用于训练的输入数据
    input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'images': mnist.train.images}, y = mnist.train.labels,
    batch_size = batch_size, num_epochs = None, shuffle = True)
    # 训练模型
    model.train(input_fn, steps=num_steps)

    # 测试模型
    # 定义用于测试的输入数据
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images}, y=mnist.test.labels,
        batch_size=batch_size, shuffle=False)
    # 测试模型
    model.evaluate(input_fn)

    # 对单张图片进行预测
    n_images = 4
    # 获取测试图
    test_images = mnist.test.images[:n_images]
    # 准备输入数据
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': test_images}, shuffle=False)
    # 利用模型预测
    preds = list(model.predict(input_fn))

    # 展示
    for i in range(n_images):
        plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
        plt.show()
        print("预测结果是:", preds[i])
    return